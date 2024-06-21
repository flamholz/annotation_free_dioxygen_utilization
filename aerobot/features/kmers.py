import pandas as pd 
import numpy as np 
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import gzip
from typing import Dict, List
from tqdm import tqdm

def from_sequence(seq:str, kmers:Dict[str, int], k:int=3, allowed_kmers:List[str]=None) -> Dict[str, int]:
    '''Count all k-mers in the input sequence, updating the dictionary with counts.
    
    :param seq: A sequence of nucleotides or amino acids for which to compute k-mer counts.
    :param kmers: A dictionary storing the k-mer counts for a FASTA file.
    :param k: The k-mer length. 
    :return: An updated dictionary of k-mer counts.
    '''
    # Iterate through the sequence to generate kmers.
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k] # Extract the k-mer from the sequence. 
        if kmer in allowed_kmers:
            if kmer not in kmers:
                kmers[kmer] = 0 # Add to the dictionary if it's not there already. 
            kmers[kmer] += 1 # Increment the k-mer's count.
    return kmers


def from_gzip(path, k:int=3) -> Dict[str, int]:
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
            kmers = from_sequence(seq, kmer, k=k)
    return kmers

    
def from_dataframe(df:pd.DataFrame, k:int=3) -> Dict[str, int]:
    '''Count the k-mers for sequences stored in a pandas DataFrame. 

    :param df: A DataFrame containing sequences over which k-mers will be counted. K-mer counts are accumulated
        over all sequences in the DataFrame. 
    :param k: The k-mer length. 
    :return: A dictionary mapping k-mers to their counts in the DataFrame.
    '''
    kmers = dict()
    for row in df.itertuples():
        kmers = from_sequence(row.seq, kmers, k=k)
    return kmers



def from_records(records:List[SeqRecord], k:int=3, allowed_kmers:List[str]=None):
    '''Takes a list of SeqRecords as input, which have been read in form a FASTA file. 
    This function does not assume that all sequences contained in the records belong to the same
    sequence object (i.e. genome, contig), so first groups the sequences by ID.'''

    # Group the sequences represented by the records according to their ID.
    seqs_by_id = dict()
    for record in records:
        if record.id not in seqs_by_id:
            seqs_by_id[record.id] = []
        seqs_by_id[record.id].append(str(record.seq))

    kmers, ids = [], []
    for id_, seqs in tqdm(seqs_by_id.items(), desc='kmers.from_records'):
        id_kmers = dict()
        for seq in seqs:
            id_kmers = from_sequence(seq, id_kmers, k=k, allowed_kmers=allowed_kmers)
        ids.append(id)
        kmers.append(id_kmers)
    return pd.DataFrame(kmers, index=ids)


