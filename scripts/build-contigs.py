from typing import List
import pandas as pd 
import numpy as np
import argparse
# from aerobot.contigs import * 
import os
from aerobot.utils import MODELS_PATH, DATA_PATH, save_hdf
import warnings
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from aerobot.features import kmers
from aerobot.dataset import FeatureDataset, load_feature_order
from aerobot.models import NonlinearClassifier
from aerobot import ncbi
import argparse
import glob
from tqdm import tqdm
import random
import re
# import dask.dataframe as ddf

warnings.simplefilter('ignore')

CONTIGS_PATH = os.path.join(DATA_PATH, 'contigs')
GENOMES_PATH = os.path.join(CONTIGS_PATH, 'genomes')
FEATURE_TYPES = ['aa_1mer', 'aa_2mer', 'aa_3mer']

# Prodigal has two "modes" that are worth noting, which are "Anonymous" and "Normal" mode. Normal mode is designed to operate on fairly
# complete, single genomes, while anonymous mode is designed to operate on metagenomes or incomplete genomes. I imagine that it
# is best to run Prodigal on all of the contigs from all of the genomes at once, probaly of all different sizes. This should simulate
# a metagenome, and might be more in line with what the reviewers wanted.

# In any case, it doesn't see suitable to use normal mode on contigs, as this isn't really simulating Prodigal output on contig data; 
# I don't think it allows for incomplete genes, and doesn't mirror a typical metagenomics pipeline (which, according to Dan, would
# involve using anonymous mode on mixed-species contigs).

# NOTE: The metagenome.fna file contained 329,115 total nucleotide sequences, which is apparently way bigger than a normal metagenome. 
# NOTE: The metagenome.faa file contained 3,887,504 total amino acid sequences. 

np.random.seed(42)
random.seed(42)

def get_genome_metadata(genome_ids:List[str], feature_types:List[str]=['aa_1mer', 'aa_2mer', 'aa_3mer']):

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

    print(f"get_genome_metadata: Writing genome metadata for {len(genome_metadata)} genomes to {os.path.join(CONTIGS_PATH, 'genome_metadata.csv')}")
    genome_metadata.to_csv(os.path.join(CONTIGS_PATH, 'genome_metadata.csv'))


def parse_prodigal_output(path:str, use_genomes:List[str]=None) -> List[SeqIO]:

    records, genome_ids, contig_sizes, ids = [], [], [], []
    # Modify the record IDs so that they are correctly grouped by the kmer.from_records function.
    for record in SeqIO.parse(path, 'fasta'):
        # Example Prodigal output file header:
        # >NC_000913_4 # 3734 # 5020 # 1 # ID=1_4;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;
        # I added a contig label to the end, so format would be >NC_000913_{contig_size}_{contig_number}_4 # 3734 ...
        id_ = record.id.split('#')[0]
        genome_id_prefix, genome_id_num, contig_size, contig_num, _ = id_.split('_')
        genome_id = f'{genome_id_prefix}_{genome_id_num}'

        if genome_id in use_genomes:
            record.id = f'{genome_id}_{contig_size}_{contig_num}'
            genome_ids.append(genome_id)
            contig_sizes.append(int(contig_size))
            ids.append(f'{genome_id}_{contig_size}_{contig_num}')
            records.append(record)

    metadata = pd.DataFrame()
    metadata['id'] = ids 
    metadata['genome_id'] = genome_ids 
    metadata['contig_size'] = contig_sizes

    return records, metadata


def prodigal(nt_input_path:str, mode:str='meta') -> List[SeqRecord]:
    '''Run Prodigal on the input nucleotide FASTA file.'''
    # Make a temporary directory to store the temporary files. 

    filename = os.path.basename(nt_input_path).split('.')[0] # Get the name of the file.
    dirname = os.path.dirname(nt_input_path)
    aa_output_path = os.path.join(dirname, f'{filename}.faa')
    gbk_output_path = os.path.join(dirname, f'{filename}.gbk')

    print(f'prodigal: Running Prodigal on file {nt_input_path}.')
    subprocess.run(f'prodigal -i {nt_input_path} -a {aa_output_path} -o {gbk_output_path} -p {mode} -q', shell=True, check=True)


def split_genome(genome_id:str, contig_size:int=1000, max_n_contigs:int=None) -> pd.DataFrame:
    '''Generate artificial contigs by splitting a genome into contig_size, non-overlapping chunks.'''
    # I don't think there is any overlap between contigs, typically, as they are generated by merging overlapping reads.
    # So, it is probably safe to chunk the genome normally. 

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # "You can also have a length floor of 2000, which is the minimum that metabat will consider for binning"
    parser.add_argument('--contig-sizes', nargs='+', default=[2000, 5000, 7000, 10000, 20000, 30000, 40000])
    parser.add_argument('--max-n-contigs', type=int, default=50)
    # parser.add_argument('--n-genomes', type=int, default=75) 
    args = parser.parse_args()

    dataset = FeatureDataset(os.path.join(DATA_PATH, 'testing_datasets.h5'), feature_type='aa_1mer')
    genome_ids = dataset.taxonomy('Class').drop_duplicates(keep='first').index

    get_genome_metadata(genome_ids)
    ncbi.download_genomes(genome_ids, path=GENOMES_PATH)

    if not os.path.exists(os.path.join(CONTIGS_PATH, 'metagenome.fna')):
        records = [] 
        for genome_id in tqdm(genome_ids, desc='Generating contigs...'):
            for contig_size in args.contig_sizes:
                records += split_genome(genome_id, contig_size=contig_size, max_n_contigs=args.max_n_contigs)
        # Write the simulated metagenome to a FASTA file.
        print(f'Generated {len(records)} contigs from the genome data.')
        SeqIO.write(records, os.path.join(CONTIGS_PATH, 'metagenome.fna'), 'fasta')

    if not os.path.exists(os.path.join(CONTIGS_PATH, 'metagenome.faa')):
        prodigal(os.path.join(CONTIGS_PATH, 'metagenome.fna'))

    # use_genomes = random.sample(genome_ids, args.n_genomes)
    use_genomes = genome_ids 
    records, metadata = parse_prodigal_output(os.path.join(CONTIGS_PATH, 'metagenome.faa'), use_genomes=use_genomes)

    # Create a metadata DataFrame for each contig by merging the metadata from the Prodigal output with the genome metadata.
    genome_metadata = pd.read_csv(os.path.join(CONTIGS_PATH, 'genome_metadata.csv'), index_col=0)
    genome_metadata['genome_id'] = genome_metadata.index 
    metadata = metadata.merge(genome_metadata, on=['genome_id'], how='left')
    metadata = metadata.set_index('id')

    # Having memory issues saving more than 50 genomes. Should either run it on the cluster, or just reduce the number of genomes. 
    contig_datasets_path = os.path.join(CONTIGS_PATH, 'contigs_datasets.h5')
    for i, feature_type in enumerate(['aa_1mer', 'aa_2mer', 'aa_3mer']):

        print(f'Extracting {feature_type} features from the synthetic contigs.')
        k = int(re.search(r'(\d+)', feature_type).group(1))

        features = kmers.from_records(records, k=k)
        save_hdf({feature_type:features}, contig_datasets_path) # , chunksize=1000)

    save_hdf({'metadata':metadata}, contig_datasets_path)
    print(f'Contig data saved to {contig_datasets_path}')

    