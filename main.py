import argparse
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Antibody Epitope Clustering')
parser.add_argument('antibody_file', type=str, help='Path to antibody data file')
parser.add_argument('antigen_file', type=str, help='Path to antigen data file')
parser.add_argument('--antigens', nargs='+', help='List of antigens to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
args = parser.parse_args()


# Load and preprocess data
def load_data(antibody_file, antigen_file, antigens):
    ab_data = pl.read_csv(antibody_file)
    ag_data = pl.read_csv(antigen_file)

    # Filter antigens if specified
    if antigens:
        ag_data = ag_data.filter(pl.col('HA').is_in(antigens))
        ab_data = ab_data.select([col for col in ab_data.columns if col in antigens or not col.startswith('H')])

    return ab_data, ag_data


ab_data, ag_data = load_data(args.antibody_file, args.antigen_file, args.antigens)


# Preprocess antibody sequences
def preprocess_ab_sequence(seq, cdr_positions):
    for i, (start, end) in enumerate(cdr_positions, 1):
        seq = seq[:start] + f"<CDR{i}>" + seq[start:end] + f"</CDR{i}>" + seq[end:]
    return seq

def create_mutation_features(seq_length, mut_positions):
    features = [0] * seq_length
    for pos in mut_positions:
        if 0 <= pos < seq_length:
            features[pos] = 1
    return features


ab_data = ab_data.with_columns([
    # ... (previous columns remain the same)
    pl.struct(['aaVDJRegion_h', 'cdr_positions_h'])
    .apply(lambda x: preprocess_ab_sequence(x['aaVDJRegion_h'], x['cdr_positions_h']))
    .alias('processed_h'),
    pl.struct(['aaVDJRegion_l', 'cdr_positions_l'])
    .apply(lambda x: preprocess_ab_sequence(x['aaVDJRegion_l'], x['cdr_positions_l']))
    .alias('processed_l'),
    pl.struct(['aaVDJRegion_h', 'mut_positions_h'])
    .apply(lambda x: create_mutation_features(len(x['aaVDJRegion_h']), x['mut_positions_h']))
    .alias('mutation_features_h'),
    pl.struct(['aaVDJRegion_l', 'mut_positions_l'])
    .apply(lambda x: create_mutation_features(len(x['aaVDJRegion_l']), x['mut_positions_l']))
    .alias('mutation_features_l')
])

# Load ESM-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Update tokenizer with new special tokens
special_tokens = ["<CDR1>", "</CDR1>", "<CDR2>", "</CDR2>", "<CDR3>", "</CDR3>"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
esm_model.resize_token_embeddings(len(tokenizer))


# Prepare dataset
class AntibodyAntigenDataset(Dataset):
    def __init__(self, ab_data, ag_data, antigens):
        self.ab_data = ab_data
        self.ag_data = ag_data
        self.antigens = antigens

    def __len__(self):
        return len(self.ab_data)

    def __getitem__(self, idx):
        ab = self.ab_data.row(idx)

        ab_h = tokenizer(ab['processed_h'], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        ab_l = tokenizer(ab['processed_l'], return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        mut_features_h = torch.tensor(ab['mutation_features_h'][:512], dtype=torch.float)
        mut_features_l = torch.tensor(ab['mutation_features_l'][:512], dtype=torch.float)

        ag_seqs = []
        binding_scores = []
        for antigen in self.antigens:
            ag_row = self.ag_data.filter(pl.col('HA') == antigen).row(0)
            ag_seq = tokenizer(ag_row['seq'], return_tensors='pt', padding='max_length', truncation=True,
                               max_length=512)
            ag_seqs.append(ag_seq)

            binding_score = ab[antigen] if antigen in ab.keys() else 'NA'
            binding_score = float(binding_score) if binding_score != 'NA' else np.nan
            binding_scores.append(binding_score)

        return ab_h, ab_l, ag_seqs, mut_features_h, mut_features_l, torch.tensor(binding_scores), ab['run'], ab['pid']


# Split data into train and validation sets
train_data, val_data = train_test_split(ab_data, test_size=0.2, random_state=42)

train_dataset = AntibodyAntigenDataset(train_data, ag_data, args.antigens)
val_dataset = AntibodyAntigenDataset(val_data, ag_data, args.antigens)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)


# Define model architecture
class AntibodyAntigenModel(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.esm_model = esm_model
        self.projection = nn.Linear(esm_model.config.hidden_size, 256)
        self.refinement = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ab_h, ab_l, ag):
        ab_h_emb = self.esm_model(**ab_h).last_hidden_state[:, 0, :]
        ab_l_emb = self.esm_model(**ab_l).last_hidden_state[:, 0, :]
        ag_emb = self.esm_model(**ag).last_hidden_state[:, 0, :]

        ab_emb = self.projection(ab_h_emb + ab_l_emb)
        ag_emb = self.projection(ag_emb)

        cosine_sim = nn.functional.cosine_similarity(ab_emb, ag_emb)
        refined_score = self.refinement(ab_emb * ag_emb).squeeze()

        return cosine_sim, refined_score


model = AntibodyAntigenModel(esm_model)


# Define loss function
def loss_function(cosine_sim, refined_score, target, temperature=0.07):
    mse_loss = nn.MSELoss(reduction='none')(refined_score, target)
    contrastive_loss = -torch.log(torch.exp(cosine_sim / temperature) /
                                  (torch.exp(cosine_sim / temperature) + torch.exp(target / temperature)))
    return mse_loss + 0.1 * contrastive_loss


# Training loop
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for ab_h, ab_l, ag, target, run, pid in tqdm(train_loader):
        optimizer.zero_grad()
        cosine_sim, refined_score = model(ab_h, ab_l, ag)
        loss = loss_function(cosine_sim, refined_score, target[~torch.isnan(target)]).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ab_h, ab_l, ag, target, run, pid in val_loader:
            cosine_sim, refined_score = model(ab_h, ab_l, ag)
            loss = loss_function(cosine_sim, refined_score, target[~torch.isnan(target)]).mean()
            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

print("Training completed")