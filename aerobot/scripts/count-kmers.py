import re
import pandas as pd 
import numpy as np 
from aerobot.io import RESULTS_PATH
from aerobot.dataset import dataset_load_feature_order
from aerobot.kmer import kmer_sequence_to_kmers
import os 
from tqdm import tqdm

BS_PATH = os.path.join(RESULTS_PATH, 'black_sea')


def parse_prodigal_output(path:str) -> pd.DataFrame:
    '''Read in Prodigal output, and generate a DataFrame mapping each automatically-generated gene ID to its
    corresponding amino acid sequence.'''
    # Read in the FASTA file. 
    with open(path, 'r') as f:
        lines = f.readlines()
    
    df, i = [], 0
    while i < len(lines):
        row = {}
        if '>' in lines[i]:
            match = re.search('(GCA_\d+\.\d+\.\d+)_\d+', lines[i])
            # Extract the contig ID, which is the {genome_id}.{contig} (removing the gene index).
            row['contig_id'] = match.group(1)
            i += 1
        
        seq = ''
        while (i < len(lines)) and ('>' not in lines[i]):
            seq += lines[i].strip().replace('*', '') # Make sure to remove whitespace and newlines.
            i += 1
        row['seq'] = seq
        df.append(row)

    return pd.DataFrame(df)


def get_contig_kmers(path:str, k:int=3) -> pd.DataFrame:
    '''Read in information from a Prodigal output file and group the predicted genes by contig. Then, generate k-mer count
    data for each contig.

    :param path: The path to the Prodigal output file. 
    :param k: The size of the amino acid k-mers. 
    :return: A DataFrame for each contig, mapping the contig number to the count of each k-mer. The DataFrame
        also has a column for the original genome ID. 
    '''
    df = parse_prodigal_output(path)
    # df = df.iloc[:1000]

    # def combine_contig(df:pd.DataFrame):
    #     seqs = df.seq.values.tolist()
    #     return ''.join(seqs)

    # NOTE: Should I process k-mers for each gene individually? Or can I combine sequences first?
    # I think the former is a better idea, but might want to check. 
    kmer_df = []
    for row in tqdm(df.itertuples(), desc='get_contig_kmers', total=len(df)):
        kmer_row = dict()
        kmer_row['contig_id'] = row.contig_id
        kmer_row.update(kmer_sequence_to_kmers(row.seq, dict(), k=k))
        kmer_df.append(kmer_row)
    kmer_df = pd.DataFrame(kmer_df)
    return kmer_df.groupby('contig_id').sum() # Sum k-mer counts across contigs. 


if __name__ == '__main__':

    aa_3mer_features = dataset_load_feature_order('aa_3mer')
    kmer_df = get_contig_kmers(os.path.join(BS_PATH, 'bs_contigs.faa'))
    kmer_df = kmer_df[['contig_id'] + aa_3mer_features]
    kmer_df = kmer_df.set_index('contig_id')

    kmer.to_csv(os.path.join(BS_PATH, 'bs_aa_3mer_from_contigs.csv'))