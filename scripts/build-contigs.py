from typing import List
import pandas as pd 
import numpy as np
import argparse
import os
from aerobot.utils import save_hdf
import warnings
import subprocess
from typing import Dict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from aerobot.features import kmers
from aerobot.dataset import FeatureDataset, get_feature_order
from aerobot.models import NonlinearClassifier
from aerobot import ncbi
import argparse
import glob
from tqdm import tqdm
import random
import re
# import dask.dataframe as ddf

# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


np.random.seed(42)
random.seed(42)

def get_genome_metadata(genome_ids:List[str], feature_types:List[str]=[]):

    print(f'get_genome_metadata: Retrieving metadata for {len(genome_ids)} genomes.')
    # Want to get predictions for each genome and use them as a reference. 
    genome_metadata = None

    for feature_type in feature_types:
        model = NonlinearClassifier.load(os.path.join(MODELS_PATH, f'nonlinear_{feature_type}_ternary.joblib'))
        dataset = FeatureDataset(os.path.join(DATA_PATH, 'testing_datasets.h5'), feature_type=feature_type).loc(genome_ids)

        if genome_metadata is None:
            genome_metadata = dataset.metadata[['Class', 'physiology']]
        predictions = pd.DataFrame(index=dataset.features.index)

        predictions[f'{feature_type}_prediction'] = model.predict(dataset.to_numpy()[0])
        genome_metadata = genome_metadata.merge(predictions, right_index=True, left_index=True)

    print(f"get_genome_metadata: Writing genome metadata for {len(genome_metadata)} genomes to {os.path.join(contigs_dir, 'genome_metadata.csv')}")
    genome_metadata.to_csv(os.path.join(contigs_dir, 'genome_metadata.csv'))


def split_genome(genome_id:str, contig_size:int=1000, max_n_contigs:int=None) -> pd.DataFrame:
    '''Generate artificial contigs by splitting a genome into contig_size, non-overlapping chunks.'''
    path = os.path.join(GENOMES_PATH, f'{genome_id}.fna')

    contigs = [] # This list will store contigs accumulated across all sequences in a FASTA file (if multiple are present.)
    for record in SeqIO.parse(path, 'fasta'): # Genome can be split into chromosomes. 
        seq = str(record.seq)
        n_contigs = len(seq) // contig_size
        contigs += [seq[i * contig_size: (i + 1) * contig_size] for i in range(n_contigs)]
    
    if max_n_contigs is not None:
        # Grab max_n_contigs of the sampled contigs and convert to SeqRecords. 
        contigs = random.sample(contigs, min(len(contigs), max_n_contigs)) 
    records = [SeqRecord(Seq(c), id=f'{genome_id}_{contig_size}_{i}', description='') for i, c in enumerate(contigs)]

    return records


def parse_contig_id(contig_id:str) -> Dict:
    '''Parse a contig ID, which is of the form {genome_id}_{contig_size}_{contig_number}'''

    genome_id_prefix, genome_id_num, contig_size, contig_num = contig_id.split('_')
    contig_size = int(contig_size)
    genome_id = f'{genome_id_prefix}_{genome_id_num}'
    return {'id':contig_id, 'contig_size':contig_size, 'genome_id':genome_id}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # "You can also have a length floor of 2000, which is the minimum that metabat will consider for binning"
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--contig-sizes', nargs='+', default=[2000, 5000, 10000, 20000, 50000])
    parser.add_argument('--max-n-contigs', type=int, default=1000) 
    parser.add_argument('--n-genomes', type=int, default=50) 
    args = parser.parse_args()

    contigs_dir = os.path.join(args.data_dir, 'contigs')
    genomes_dir = os.path.join(contigs_dir, 'genomes')
    contigs_feature_types = ['nt_3mer', 'nt_4mer', 'nt_5mer']

    dataset = FeatureDataset(os.path.join(args.data_dir, 'testing_datasets.h5'), feature_type='nt_2mer')
    genome_ids = np.random.choice(dataset.metadata.index.values, size=args.n_genomes, replace=False)
    
    get_genome_metadata(genome_ids, feature_types=contigs_feature_types)
    ncbi.download_genomes(genome_ids, path=genomes_dir)

    if not os.path.exists(os.path.join(contigs_dir, 'metagenome.fna')):
        records = [] 
        for genome_id in tqdm(genome_ids, desc='Generating contigs...'):
            for contig_size in args.contig_sizes:
                records += split_genome(genome_id, contig_size=contig_size, max_n_contigs=args.max_n_contigs)
        # Write the simulated metagenome to a FASTA file.
        print(f'Generated {len(records)} contigs from the genome data.')
        SeqIO.write(records, os.path.join(contigs_dir, 'metagenome.fna'), 'fasta')
    else:
        records = list(SeqIO.parse(os.path.join(contigs_dir, 'metagenome.fna'), 'fasta'))

    # Create a metadata DataFrame for each contig by merging the metadata from the Prodigal output with the genome metadata.
    genome_metadata = pd.read_csv(os.path.join(contigs_dir, 'genome_metadata.csv'), index_col=0)
    genome_metadata['genome_id'] = genome_metadata.index 
    # Extract metadata from the contig IDs. 
    metadata = pd.DataFrame([parse_contig_id(record.id) for record in records])
    metadata = metadata.merge(genome_metadata, on=['genome_id'], how='left')
    metadata = metadata.set_index('id')

    # Having memory issues saving more than 50 genomes. Should either run it on the cluster, or just reduce the number of genomes. 
    contig_datasets_path = os.path.join(contigs_dir, 'datasets.h5')
    for i, feature_type in enumerate(CONTIGS_FEATURE_TYPES):

        print(f'Extracting {feature_type} features from the synthetic contigs.')
        k = int(re.search(r'(\d+)', feature_type).group(1))

        features = kmers.from_records(records, k=k, allowed_kmers=get_feature_order(feature_type))
        features = features.fillna(0).astype(int) # Trying to silence a performance warning. 
        save_hdf({feature_type:features}, contig_datasets_path) # , chunksize=1000)

    print(f'Contig data saved to {contig_datasets_path}')

