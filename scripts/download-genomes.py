from Bio import Entrez, SeqIO
import pandas as pd 
import gzip
import time 
from aerobot.io import RESULTS_PATH
import numpy as np 
import os
import argparse

# NOTE: Do you actually need an API key?
# First, you need to create an NCBI account and generate an API key. 
# Go to Account settings > API Key Management and create a key. 

# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.chapter2_table1


def fix_genome_id(genome_id:str) -> str:
    '''Some of the genome IDs in the dataset are GenBank accessions, not RefSeq accesssions. Also, they are missing the .{int} 
    extension, which specifies the version of the genome. This function changes a GenBank prefix to a RefSeq prefix 
    (under the assumption that the numerical portion is the same), and adds a '.1' extension if not present.'''
    original_genome_id = genome_id
    genome_id = genome_id.replace('GCA', 'GCF') # Modify the prefix to align with RefSeq convention. 
    if '.' not in genome_id:
        genome_id = genome_id + '.1'
    
    if original_genome_id != genome_id:
        print(f'fix_genome_id: Modified input genome ID to match RefSeq conventions, {original_genome_id} > {genome_id}.')
    return genome_id 


def get_nt_ids(genome_id:str):
    '''Get the search ID for the nucleotide database using a RefSeq genome ID. '''
    genome_id = fix_genome_id(genome_id)

    handle = Entrez.esearch(db='nucleotide', term=genome_id)
    results = Entrez.read(handle)
    handle.close()

    if len(results['IdList']) < 1:
        print(f'get_nt_id: No search hits for genome {genome_id}.')
        return None

    return results['IdList']


def download_genomes(genome_ids, output_path:str=None, complete_only:bool=False):

    for genome_id in genome_ids:
        # Retrieve the nucleotide accession for the genomic sequence
        nt_ids = get_nt_ids(genome_id) # Can be multiple IDs if the genome is in the form of contigs.
        if nt_ids is None:
            continue
        if (len(nt_ids)) > 1 and (complete_only): # If more than one nucleotide ID is found, then the genome is incomplete.
            print(f'download_genomes: Skipping {genome_id}, as genome is incomplete.')
            continue

        # Fetch the genomic sequence(s). 
        records = []
        for nt_id in nt_ids:
            handle = Entrez.efetch(db='nucleotide', id=nt_id, rettype='fasta')
            records.append(handle.read().strip()) # Remove trailing whitespace

        # Write the obtained sequences as records in a FASTA file. 
        filename =  f'{genome_id}.fasta'
        with open(os.path.join(output_path, filename), 'w') as f:
            f.write('\n'.join(records))
        handle.close()

        print(f'download_genomes: Downloaded {genome_id} to {os.path.join(output_path, filename)}.')
        
        time.sleep(1)  # Respectful delay so as not to get blocked by NCBI. 
        

if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('genomes', type=str, nargs='+', help='One or more genome IDs to look up.')
    parser.add_argument('--email', type=str, default='prichter@caltech.edu')
    parser.add_argument('--api-key', type=str, default='2ff07cb20e93ddb8b358f92f91cae939e209')
    parser.add_argument('--output-path', type=str, default=os.path.join(RESULTS_PATH, 'contigs', 'genomes'))
    parser.add_argument('--complete-only', type=bool, default=0, help='Whether or not to skip non-complete genomes.')

    args = parser.parse_args()
    Entrez.email = args.email
    Entrez.api_key = args.api_key

    download_genomes(args.genomes, output_path=args.output_path, complete_only=args.complete_only)

