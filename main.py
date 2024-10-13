import argparse
import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
from esm import pretrained


def get_esm_embedding(sequence):
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("", sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    embeddings = results["representations"][33].numpy()
    return embeddings.mean(axis=1)  # Average pooling over sequence length


def insert_separator_tokens(sequence, positions):
    for pos in sorted(positions, reverse=True):
        sequence = sequence[:pos] + "[SEP]" + sequence[pos:]
    return sequence


def create_position_mask(sequence, positions):
    mask = np.zeros(len(sequence))
    mask[positions] = 1
    return mask


def main(antibody_file, antigen_file, ha_names, output_file):
    # Load ESM-2 model
    global model, alphabet
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()  # Set the model to evaluation mode

    # Load data
    antibody_df = pl.read_csv(antibody_file)
    antigen_df = pl.read_csv(antigen_file)

    # Process antibody sequences
    for chain in ['h', 'l']:
        cdr_positions = [f"aaCDR{i}StartPosition_{chain}" for i in range(1, 4)] + [f"aaCDR{i}EndPosition_{chain}" for i
                                                                                   in range(1, 4)]
        antibody_df = antibody_df.with_column(
            pl.col(f"aaVDJRegion_{chain}").apply(lambda seq, pos: insert_separator_tokens(seq, pos)).alias(
                f"processed_sequence_{chain}")
            .arr.eval(pl.col(*cdr_positions), parallel=True)
        )

    antibody_df = antibody_df.with_column(
        pl.concat_str([f"processed_sequence_{chain}" for chain in ['h', 'l']], separator="[SEP]").alias(
            "combined_sequence")
    )

    # Generate embeddings
    antibody_embeddings = np.vstack(antibody_df["combined_sequence"].apply(get_esm_embedding).to_numpy())
    antigen_embeddings = np.vstack(antigen_df["seq"].apply(get_esm_embedding).to_numpy())

    # Process binding scores
    binding_scores = antibody_df.select(ha_names).to_numpy()
    missing_mask = np.isnan(binding_scores)

    # Normalize scores within each run
    for run in antibody_df["run"].unique():
        run_mask = antibody_df["run"] == run
        run_scores = binding_scores[run_mask.to_numpy()]
        run_mean = np.nanmean(run_scores, axis=0)
        run_std = np.nanstd(run_scores, axis=0)
        binding_scores[run_mask.to_numpy()] = (run_scores - run_mean) / run_std

    binding_scores = np.nan_to_num(binding_scores, nan=0.0)

    # Process additional features
    mutation_masks = {}
    for chain in ['h', 'l']:
        mutation_masks[chain] = antibody_df.select(
            pl.col(f"aaMutPos_{chain}").str.split(',').apply(
                lambda pos: create_position_mask(antibody_df[f"aaVDJRegion_{chain}"][0], [int(p) for p in pos if p]))
        ).to_numpy()

    # One-hot encode V and J genes
    onehot_encoder = OneHotEncoder(sparse=False)
    v_genes_encoded = onehot_encoder.fit_transform(antibody_df[["v_gene_h", "v_gene_l"]].to_numpy())
    j_genes_encoded = onehot_encoder.fit_transform(antibody_df[["j_gene_h", "j_gene_l"]].to_numpy())

    additional_features = np.hstack([
        pad_sequence([torch.tensor(m) for m in mutation_masks['h']], batch_first=True).numpy(),
        pad_sequence([torch.tensor(m) for m in mutation_masks['l']], batch_first=True).numpy(),
        v_genes_encoded,
        j_genes_encoded
    ])

    # Encode metadata
    metadata_encoded = pl.get_dummies(antibody_df.select(["run", "pid"])).to_numpy()

    # Prepare final data
    preprocessed_data = {
        'antibody_embeddings': antibody_embeddings,
        'antigen_embeddings': antigen_embeddings,
        'binding_scores': binding_scores,
        'missing_mask': missing_mask,
        'additional_features': additional_features,
        'metadata': metadata_encoded
    }

    # Save preprocessed data
    np.savez_compressed(output_file, **preprocessed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess antibody and antigen data")
    parser.add_argument("antibody_file", help="Path to the antibody data CSV file")
    parser.add_argument("antigen_file", help="Path to the antigen data CSV file")
    parser.add_argument("--ha_names", nargs="+",
                        default=["H1vic", "H1cal", "H3taz", "H3dar", "H4", "H5", "H6", "H8", "H9", "H10", "H13", "H17",
                                 "H18", "HBwash"],
                        help="List of HA names to process")
    parser.add_argument("--output_file", default="preprocessed_data.npz", help="Path to save the preprocessed data")

    args = parser.parse_args()

    main(args.antibody_file, args.antigen_file, args.ha_names, args.output_file)