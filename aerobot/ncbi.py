'''Code for downloading genomes from the NCBI database. This is used primarily in the predict-contigs.py script.'''
from Bio import Entrez, SeqIO
import pandas as pd 
import gzip
import time 
from aerobot.io import RESULTS_PATH, DATA_PATH
import numpy as np 
import os
import argparse
from typing import List 

# NOTE: Do you actually need an API key?
# First, you need to create an NCBI account and generate an API key. 
# Go to Account settings > API Key Management and create a key. 

# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.chapter2_table1

GENOMES_PATH = os.path.join(DATA_PATH, 'contigs', 'genomes')

Entrez.email = 'prichter@caltech.edu'
Entrez.api_key = '2ff07cb20e93ddb8b358f92f91cae939e209'

def ncbi_genome_id_to_refseq(genome_id:str) -> str:
    '''Some of the genome IDs in the dataset are GenBank accessions, not RefSeq accesssions. Also, they are missing the .{int} 
    extension, which specifies the version of the genome. This function changes a GenBank prefix to a RefSeq prefix 
    (under the assumption that the numerical portion is the same), and adds a '.1' extension if not present.'''
    original_genome_id = genome_id
    genome_id = genome_id.replace('GCA', 'GCF') # Modify the prefix to align with RefSeq convention. 
    if '.' not in genome_id:
        genome_id = genome_id + '.1'
    
    if original_genome_id != genome_id:
        print(f'ncbi_genome_id_to_refseq: Modified input genome ID to match RefSeq conventions, {original_genome_id} > {genome_id}.')
    return genome_id 


def ncbi_get_16s_seqs(gene_ids:List[str]) -> List[str]:
    records = []
    for accession in accession_list:
        try:
            handle = Entrez.efetch(db='nucleotide', id=accession, rettype='gb', retmode='text')
            record = SeqIO.read(handle, 'genbank')
            handle.close()
            records.append(record)
        except Exception as e:
            print(f"Error fetching accession {accession}: {e}")
    return records



def ncbi_get_nt_ids(genome_id:str) -> List[str]:
    '''Get the search ID for the nucleotide database using a RefSeq genome ID.

    :param genome_id: An RefSeq genome ID. If the genome ID does not conform to RefSeq convention, it is 
        modified using the genome_id_to_refseq function.
    :return: A list of integer identifiers for the nucleotide sequences attached to the genome ID in the NCBI
        nucleotide database. 
    '''
    genome_id = ncbi_genome_id_to_refseq(genome_id)

    handle = Entrez.esearch(db='nucleotide', term=genome_id)
    results = Entrez.read(handle)
    handle.close()

    if len(results['IdList']) == 0:
        return None

    return results['IdList']


def ncbi_download_genomes(genome_ids:List[str]) -> List[str]:
    '''Download the genomes with the specified IDs from the NCBI database.

    :param genome_ids: A list of genome IDs to download.
    :param complete_genomes_only: Whether or not to skip a download if the genome is not complete (i.e. is present
        in contigs, as opposed to a closed genome).
    :return: A list of the genome IDs which were successfully downloaded, or were already present in the GENOMES_PATH. 
    ''' 
    successfully_downloaded = []

    for genome_id in genome_ids:
        # Write the obtained sequences as records in a FASTA file. 
        filename =  f'{genome_id}.fasta'

        if filename in os.listdir(GENOMES_PATH):
            # print(f'ncbi_download_genomes: Skipping {genome_id}, as it is already present in the output directory.')
            successfully_downloaded.append(genome_id)
            continue

        # Retrieve the nucleotide accession for the genomic sequence
        nt_ids = ncbi_get_nt_ids(genome_id) # Can be multiple IDs if the genome is in the form of contigs.

        if nt_ids is None:
            print(f'ncbi_download_genomes: Skipping {genome_id}, as no search hits were found in the NCBI nucleotide database.')
            continue
        # This doesn't work, as there is also a case where the genome is complete, but spread across multiple chromosomes. 
        # About 10 percent of bacterial species have more than one chromosome. These genomes are called multipartite genomes. 
        # if (len(nt_ids)) > 1 and (complete_genomes_only): # If more than one nucleotide ID is found, then the genome is incomplete.
        #     print(f'ncbi_download_genomes: Skipping {genome_id}, as genome is incomplete.')
        #     continue

        # Fetch the genomic sequence(s). 
        records = []
        for nt_id in nt_ids:
            handle = Entrez.efetch(db='nucleotide', id=nt_id, rettype='fasta')
            records.append(handle.read().strip()) # Remove trailing whitespace

        # Write the sequence(s) to a FASTA file. 
        with open(os.path.join(GENOMES_PATH, filename), 'w') as f:
            f.write('\n'.join(records))
        handle.close()

        print(f'ncbi_download_genomes: Downloaded {genome_id} to {os.path.join(GENOMES_PATH, filename)}.')
        successfully_downloaded.append(genome_id)
        time.sleep(1)  # Respectful delay so as not to get blocked by NCBI. 

    return successfully_downloaded # Return the list of completed downloads.
        