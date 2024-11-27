from Bio import Entrez, SeqIO
import pandas as pd 
import gzip
import time 
import numpy as np 
import os
from tqdm import tqdm
import argparse
from typing import List 
from bs4 import BeautifulSoup
import zipfile 
import subprocess
import shutil


# NOTE: Do you actually need an API key?
# First, you need to create an NCBI account and generate an API key. 
# Go to Account settings > API Key Management and create a key. 

# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.chapter2_table1


Entrez.email = 'prichter@caltech.edu'
Entrez.api_key = '2ff07cb20e93ddb8b358f92f91cae939e209'


def download_rna16s_seqs(ids:List[str]) -> List[str]:
    '''Function copied over from Josh's code for setting up the 16S classifier training and validation datasets. Downloads sequences
    using accessions which are contained in the Mark_Westoby_Organism_Metadata_Export_02152018.tsv file.
    '''
    records = []
    for id_ in tqdm(ids, desc='download_rna16s_seqs'):
        try:
            handle = Entrez.efetch(db='nucleotide', id=id_, rettype='gb', retmode='text')
            record = SeqIO.read(handle, 'genbank')
            handle.close()
            records.append(record)
            # print(f'rna16s_get_seqs: Successfully obtained sequence for ID {id_}.')
        except Exception as e:
            print(f'download_rna16s_seqs: Error fetching sequence for ID {id_}.')
    return records


def download_taxonomy(ids:List[str]):
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    rows = []
    for id_ in tqdm(ids, desc='download_taxonomy'):
        try:
            handle = Entrez.efetch(db='taxonomy', id=id_, rettype='text')
            soup = BeautifulSoup(handle.read(), features='xml')
            lineage = soup.Lineage.text.split(';')[1:] # First entry in the taxonomy string is pointless. 
            rows.append({level:taxonomy for level, taxonomy in zip(levels, lineage)})
            # time.sleep(1)
        except Exception as e:
            print(f'download_rna16s_seqs: Error fetching taxonomy for ID {id_}.')
            rows.append({level:'no rank' for level in levels})
    df = pd.DataFrame(rows)
    df.index = ids 
    return df


def download_genomes(genome_ids, path:str=None):

    archive_path = os.path.join(path, 'ncbi_dataset.zip')

    def extract_genome_from_archive(genome_id:str):
        # https://stackoverflow.com/questions/4917284/extract-files-from-zip-without-keeping-the-structure-using-python-zipfile
        archive = zipfile.ZipFile(archive_path)
        for member in archive.namelist():
            if member.startswith(f'ncbi_dataset/data/{genome_id}'):
                source = archive.open(member)
                # NOTE: Why does wb not result in another zipped file being created?
                with open(os.path.join(path, f'{genome_id}.fna'), 'wb') as target:
                    shutil.copyfileobj(source, target)

    for genome_id in tqdm(genome_ids, desc='download_genomes'):
        if not os.path.exists(os.path.join(path, f'{genome_id}.fna')):
            # Make sure to add a .1 back to the genome accession (removed while removing duplicates).
            cmd = f'datasets download genome accession {genome_id}.1 --filename {archive_path} --include genome --no-progressbar'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            extract_genome_from_archive(genome_id)
    
            os.remove(archive_path)

