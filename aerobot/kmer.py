
'''Code for generating files containing k-mer counts from FASTA files.'''

import pandas as pd
import numpy as np
import gzip
from Bio import Entrez, SeqIO
from typing import Dict

def kmer_sequence_to_kmers(seq:str, kmers:Dict[str, int], k:int=3) -> Dict[str, int]:
    '''Count all k-mers in the input sequence, updating the dictionary with counts.
    
    :param seq: A sequence of nucleotides or amino acids for which to compute k-mer counts.
    :param kmers: A dictionary storing the k-mer counts for a FASTA file.
    :param k: The k-mer length. 
    :return: An updated dictionary of k-mer counts.
    '''
    # Iterate through the sequence to generate kmers.
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k] # Extract the k-mer from the sequence. 
        if kmer not in kmers:
            kmers[kmer] = 0 # Add to the dictionary if it's not there already. 
        kmers[kmer] += 1 # Increment the k-mer's count.
    return kmers


def kmer_count_gz(path, k:int=3) -> Dict[str, int]:
    '''Count k-mers stored in a zipped file (with a .gz extension).

    :param path: The path to the compressed FASTA file to read. 
    :param k: The k-mer length. 
    :return: A dictionary mapping k-mers to their counts in the FASTA file.
    '''
    kmers = dict()
    with gzip.open(path, 'r') as f:
        # Parse the fasta file and iterate through each record.
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq) 
            kmers = kmer_sequence_to_kmers(seq, kmer, k=k)
    return kmers
    
def kmer_count_dataframe(df:pd.DataFrame, k:int=3) -> Dict[str, int]:
    '''Count the k-mers for sequences stored in a pandas DataFrame. 

    :param df: A DataFrame containing sequences over which k-mers will be counted. K-mer counts are accumulated
        over all sequences in the DataFrame. 
    :param k: The k-mer length. 
    :return: A dictionary mapping k-mers to their counts in the DataFrame.
    '''
    kmers = dict()
    for row in df.itertuples():
        kmers = kmer_sequence_to_kmers(row.seq, kmers, k=k)
    return kmers


def kmer_count_fasta(path:str, k:int=3) -> Dict[str, int]:
    '''Count k-mers stored in a non-compressed FASTA file (with a .fa, .fn, etc. extension).
    
    :param path: The path to the FASTA file to read. 
    :param k: The k-mer length. 
    :return: A dictionary mapping k-mers to their counts in the FASTA file.
    '''
    kmers = dict() # Initialize a dictionary to store the k-mers. 
    with open(path, 'r') as f:
        # Parse the FASTA file and iterate through each record.
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq) 
            kmers = kmer_sequence_to_kmers(seq, kmers, k=k)
    return kmers



