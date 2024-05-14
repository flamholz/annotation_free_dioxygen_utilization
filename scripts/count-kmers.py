'''A script for generating amino-acid k-mer counts from a FASTA file produced by Prodigal.'''
import re
import pandas as pd 
import numpy as np 
from aerobot.io import RESULTS_PATH, load_fasta
# from aerobot.dataset import dataset_load_feature_order
from aerobot.kmer import kmer_sequence_to_kmers
import os 
from tqdm import tqdm
import argparse



def get_contig_kmers(path:str, k:int=3) -> pd.DataFrame:
    '''Read in information from a Prodigal output file and group the predicted genes by contig. Then, generate k-mer count
    data for each contig.

    :param path: The path to the Prodigal output file. 
    :param k: The size of the amino acid k-mers. 
    :return: A DataFrame for each contig, mapping the contig number to the count of each k-mer. The DataFrame
        also has a column for the original genome ID. 
    '''
    df = load_fasta(path)

    # def combine_contig(df:pd.DataFrame):
    #     seqs = df.seq.values.tolist()
    #     return ''.join(seqs)

    # NOTE: Should I process k-mers for each gene individually? Or can I combine sequences first?
    # I think the former is a better idea, but might want to check. 
    kmer_df = []
    for row in tqdm(df.itertuples(), desc='get_contig_kmers', total=len(df)):
        kmer_row = dict()
        kmer_row = kmer_sequence_to_kmers(row.seq, kmer_row, k=k)
        kmer_row['contig_id'] = row.header
        kmer_df.append(kmer_row)
    kmer_df = pd.DataFrame(kmer_df)

    return kmer_df.groupby('contig_id').sum() # Sum k-mer counts across contigs. 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str, help='Location of the FASTA file containing contigs, or a complete genome.')
    parser.add_argument('--output-path', '-o', type=str, help='Location to which to write the k-mer count CSV.')
    parser.add_argument('-k', '--kmer-size', type=int, default=3, help='The size of the k-mers. ')

    args = parser.parse_args()

    # aa_3mer_features = dataset_load_feature_order('aa_3mer')
    kmer_df = get_contig_kmers(args.input_path, k=args.kmer_size)
    # kmer_df = kmer_df[['contig_id'] + aa_3mer_features] # contig_id is already the index after merging. 
    # kmer_df = kmer_df.set_index('contig_id')
    kmer_df.to_csv(args.output_path)