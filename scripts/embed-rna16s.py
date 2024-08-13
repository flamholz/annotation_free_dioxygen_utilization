import argparse
from aerobot.features import rna16s
from aerobot.utils import DATA_PATH
import os

# NOTE: This script must be run in the aerobot-16s environment due to dependency issues when using the genslm model. 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None, help='The path to the input FASTA file containing 16S RNA sequences.')
    parser.add_argument('--output-path', type=str, default=None, help='The path of the file where the embeddings will be written.')

    args = parser.parse_args()

    embeddings_df = rna16s.from_fasta(args.input_path) # The index is already the sequence ID. 
    embeddings_df.to_csv(args.output_path)