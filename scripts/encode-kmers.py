import pandas as pd
import numpy as np
from aerobot.utils import FEATURE_TYPES, AMINO_ACIDS, NUCLEOTIDES
import os
import argparse
from Bio import SeqIO, Seq, SeqRecord
from aerobot.features import kmers
import itertools


FASTA_EXTENSIONS = ['.fa', '.fn', '.fasta']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str)
    parser.add_argument('--output-path', '-o', type=str, default=None)
    parser.add_argument('--kmer-size', type=int, default=3)
    parser.add_argument('--kmer-type', choices=['nt', 'aa'], default='aa')
    # parser.add_argument('--genome-id', type=str, default=None)
    args = parser.parse_args()


    # First, parse the FASTA file. 
    records = list(SeqIO.parse(args.input_path, 'fasta'))
    alphabet = NUCLEOTIDES if (args.kmer_type == 'nt') else AMINO_ACIDS
    allowed_kmers = keywords = [''.join(i) for i in itertools.product(alphabet, repeat=args.kmer_size)]

    genome_id = os.path.basename(args.input_path)
    genome_id, _ = os.path.splitext(genome_id)

    output_path = args.output_path
    if output_path is None: # Create a default output path if none is specified. 
        output_path = os.path.dirname(args.input_path)
        output_file_name = f'{genome_id}_{args.kmer_type}_{args.kmer_size}mer.csv'
        output_path = os.path.join(output_path, output_file_name)

    # The DataFrame produced by this function already has the genome
    kmer_df = kmer.from_records(records, k=args.kmer_size, allowed_kmers=allowed_kmers, genome_id=genome_id, ignore_file_ids=True)
    print(f'Saved k-mers of size {args.kmer_size} to {args.output_path}')
    kmer.to_csv(args.output_path)

