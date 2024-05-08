from Bio import Entrez, SeqIO
import pandas as pd 
import gzip
import time 
from aerobot.io import ASSET_PATH
import numpy as np 
import argparse

# NOTE: Do you actually need an API key?
# First, you need to create an NCBI account and generate an API key. 
# Go to Account settings > API Key Management and create a key. 

# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.chapter2_table1

def get_nt_id(genome_id:str):
    '''Get the search ID for the nucleotide database using a RefSeq genome ID. '''

    handle = Entrez.esearch(db='nucleotide', term=genome_id)
    results = Entrez.read(handle)
    handle.close()

    return results['IdList'][0]['Id']


def download_genomes(genome_ids, output_dir:str=None):

    for genome_id in genome_ids:
        # Retrieve the nucleotide accession for the genomic sequence
        nt_id = get_nt_id(genome_id)

        # Fetch the genomic sequence.
        handle = Entrez.efetch(db='nucleotide', id=nt_id, rettype='fasta', retmode='text')
        file_path = os.path.join(output_dir, f"{genome_id}.fasta")
        
        with open(file_path, 'w') as f:
            f.write(handle.read())
        handle.close()

        print(f'download_genomes: Downloaded {genome_id} to {file_path}.')
        
        time.sleep(1)  # Respectful delay so as not to get blocked by NCBI. 
        

if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default='prichter@caltech.edu')
    parser.add_argument('--api-key', type=str, default='2ff07cb20e93ddb8b358f92f91cae939e209')
    parser.add_argument('--output-dir', type=str, default=os.path.join(ASSET_PATH, 'data', 'genomes'))

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    Entrez.email = args.email
    Entrez.api_key = args.api_key