import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import numpy as np

import argparse
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast
import numpy as np


class AntibodyAntigenDataset(Dataset):
    def __init__(self, antibody_df, antigen_df, antigen_list):
        self.antibody_df = antibody_df
        self.antigen_df = antigen_df
        self.antigen_list = antigen_list

        # Load the ESM-2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        # Add CDR tokens to the tokenizer
        new_tokens = ["<CDR1>", "</CDR1>", "<CDR2>", "</CDR2>", "<CDR3>", "</CDR3>"]
        self.tokenizer.add_tokens(new_tokens)

    def __len__(self):
        return len(self.antibody_df)

    def __getitem__(self, idx):
        ab_row = self.antibody_df.row(idx)

        # Prepare antibody sequence
        heavy_seq = ab_row['aaVDJRegion_h']
        light_seq = ab_row['aaVDJRegion_l']

        # Add CDR separators
        heavy_seq = self.add_cdr_separators(heavy_seq, ab_row, '_h')
        light_seq = self.add_cdr_separators(light_seq, ab_row, '_l')

        # Combine heavy and light chains
        ab_seq = heavy_seq + " " + light_seq

        # Tokenize antibody sequence
        ab_tokens = self.tokenizer(ab_seq, return_tensors="pt", padding=True, truncation=True)

        # Prepare antigen sequences and binding scores
        ag_seqs = []
        binding_scores = []
        for ag in self.antigen_list:
            ag_seq = self.antigen_df.filter(pl.col('HA') == ag).select('seq').item()
            ag_tokens = self.tokenizer(ag_seq, return_tensors="pt", padding=True, truncation=True)
            ag_seqs.append(ag_tokens)
            binding_scores.append(ab_row[ag] if ag in ab_row.keys() else 0)

        # Prepare additional features
        additional_features = self.get_additional_features(ab_row)

        return ab_tokens, ag_seqs, torch.tensor(binding_scores, dtype=torch.float32), additional_features

    def add_cdr_separators(self, seq, row, chain):
        cdr1_start = row[f'aaCDR1StartPosition{chain}']
        cdr1_end = row[f'aaCDR1EndPosition{chain}']
        cdr2_start = row[f'aaCDR2StartPosition{chain}']
        cdr2_end = row[f'aaCDR2EndPosition{chain}']
        cdr3_start = row[f'aaCDR3StartPosition{chain}']
        cdr3_end = row[f'aaCDR3EndPosition{chain}']

        seq = (seq[:cdr1_start] + "<CDR1>" + seq[cdr1_start:cdr1_end] + "</CDR1>" +
               seq[cdr1_end:cdr2_start] + "<CDR2>" + seq[cdr2_start:cdr2_end] + "</CDR2>" +
               seq[cdr2_end:cdr3_start] + "<CDR3>" + seq[cdr3_start:cdr3_end] + "</CDR3>" +
               seq[cdr3_end:])
        return seq

    def get_additional_features(self, row):
        features = []
        for chain in ['_h', '_l']:
            mut_pos = [int(pos) for pos in row[f'aaMutPos{chain}'].split(',') if pos]
            features.extend([1 if i in mut_pos else 0 for i in range(len(row[f'aaVDJRegion{chain}']))])
        return torch.tensor(features, dtype=torch.float32)


class ContrastiveModel(nn.Module):
    def __init__(self, esm_model, feature_dim, output_dim):
        super().__init__()
        self.esm_model = esm_model
        self.ab_proj = nn.Linear(esm_model.config.hidden_size + feature_dim, output_dim)
        self.ag_proj = nn.Linear(esm_model.config.hidden_size, output_dim)
        self.predictor = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ab_tokens, ag_tokens, additional_features):
        ab_emb = self.esm_model(**ab_tokens).last_hidden_state[:, 0, :]
        ab_emb = torch.cat([ab_emb, additional_features], dim=1)
        ab_proj = self.ab_proj(ab_emb)

        ag_emb = self.esm_model(**ag_tokens).last_hidden_state[:, 0, :]
        ag_proj = self.ag_proj(ag_emb)

        combined = torch.cat([ab_proj, ag_proj], dim=1)
        prediction = self.predictor(combined)
        return prediction.squeeze()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for ab_tokens, ag_seqs, binding_scores, additional_features in train_loader:
        ab_tokens = {k: v.to(device) for k, v in ab_tokens.items()}
        additional_features = additional_features.to(device)
        binding_scores = binding_scores.to(device)

        optimizer.zero_grad()
        loss = 0
        for i in range(len(ag_seqs)):
            ag_tokens = {k: v.to(device) for k, v in ag_seqs[i].items()}
            predictions = model(ab_tokens, ag_tokens, additional_features)
            loss += criterion(predictions, binding_scores[:, i])
        loss /= len(ag_seqs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for ab_tokens, ag_seqs, binding_scores, additional_features in val_loader:
            ab_tokens = {k: v.to(device) for k, v in ab_tokens.items()}
            additional_features = additional_features.to(device)
            binding_scores = binding_scores.to(device)

            loss = 0
            for i in range(len(ag_seqs)):
                ag_tokens = {k: v.to(device) for k, v in ag_seqs[i].items()}
                predictions = model(ab_tokens, ag_tokens, additional_features)
                loss += criterion(predictions, binding_scores[:, i])
            loss /= len(ag_seqs)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(antibody_file, antigen_file, antigen_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data using Polars
    antibody_df = pl.read_csv(antibody_file)
    antigen_df = pl.read_csv(antigen_file)

    # Create dataset
    dataset = AntibodyAntigenDataset(antibody_df, antigen_df, antigen_list)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Resize token embeddings to account for new CDR tokens
    esm_model.resize_token_embeddings(len(dataset.tokenizer))

    feature_dim = len(dataset[0][3])
    model = ContrastiveModel(esm_model, feature_dim, output_dim=128).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_antibody_clustering_model.pth")
            print("New best model saved!")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antibody Clustering Model")
    parser.add_argument("antibody_file", type=str, help="Path to the antibody CSV file")
    parser.add_argument("antigen_file", type=str, help="Path to the antigen CSV file")
    parser.add_argument("antigen_list", nargs="+", help="List of antigens to consider")
    args = parser.parse_args()

    main(args.antibody_file, args.antigen_file, args.antigen_list)