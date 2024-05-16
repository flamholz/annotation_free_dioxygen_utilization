import re
import pandas as pd 
import numpy as np 
from aerobot.io import RESULTS_PATH, MODELS_PATH, DATA_PATH, load_fasta, save_fasta
from aerobot.dataset import dataset_clean_features, dataset_load, dataset_clean, dataset_load_feature_order
# from aerobot.kmer import kmer_sequence_to_kmers
from Bio import SeqIO
import os 
from aerobot.models import GeneralClassifier
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import subprocess
from typing import List, Tuple
from aerobot.kmer import kmer_count_fasta, kmer_count_dataframe

CONTIG_PATH = os.path.join(DATA_PATH, 'contigs')
GENOMES_PATH = os.path.join(CONTIG_PATH, 'genomes')
SCRIPTS_PATH = os.path.join('..', 'scripts')

KMER_FEATURE_TYPES = ['aa_1mer', 'aa_2mer', 'aa_3mer']
KMER_FEATURE_TYPES += ['nt_1mer', 'nt_2mer', 'nt_3mer', 'nt_4mer', 'nt_5mer']
# KMER_FEATURE_TYPES += ['cds_1mer', 'cds_2mer', 'cds_3mer', 'cds_4mer', 'cds_5mer']


def contigs_get_genome_size(genome_id:str) -> int:
    fasta_df = load_fasta(os.path.join(GENOMES_PATH, f'{genome_id}.fasta'))
    # If the length of the DataFrame is greater than 1, then there are multiple contigs in the file. 
    assert len(fasta_df) == 1, 'generate_contigs: Input FASTA file does not contain a closed genome. '
    genome = fasta_df.iloc[0].seq
    return len(genome)


def contigs_split_genome_v1(genome_id:str, contig_size:int=1000) -> pd.DataFrame:
    '''Generate artificial contigs by splitting a genome into contig_size, non-overlapping chunks.
    
    :param genome_id: The ID of the genome to break into contigs.
    :param contig_size: The size of the contigs. 
    '''
    # I don't think there is any overlap between contigs, typically, as they are generated by merging overlapping reads.
    # So, it is probably safe to chunk the genome normally. 

    # NOTE: I should make sure to save the boundaries of each artificial contig. It might be cool to see if there is any correllation
    # between contig location within the genome and the ability of the model to predict metabolism. 
    
    fasta_df = load_fasta(os.path.join(GENOMES_PATH, f'{genome_id}.fasta'))
    # If the length of the DataFrame is greater than 1, then there are multiple contigs in the file. 
    assert len(fasta_df) == 1, 'generate_contigs: Input FASTA file does not contain a closed genome. '
    genome = fasta_df.iloc[0].seq
    
    if contig_size is not None:
        n_contigs = len(genome) // contig_size
        n_contigs = n_contigs + 1 if (len(genome) % contig_size > 0) else n_contigs # Add one to account for the trailing end. 
        contigs = [genome[i * contig_size: (i + 1) * contig_size] for i in range(n_contigs)]

        df = pd.DataFrame()
        df['seq'] = contigs
        df['header'] = [f'{genome_id}_{i}' for i in range(1, n_contigs + 1)]

    else: # If contig_size is None, use the whole genome. 
        df = pd.DataFrame()
        df['seq'] = [genome]
        df['header'] = [genome_id]
        
    return df


def contigs_split_genome_v2(genome_id:str, contig_size:int=1000) -> pd.DataFrame:
    '''Generate contigs by sampling substrings of size contig_size from the specified genome, with replacement. 

    :param genome_id: The ID of the genome to break into contigs. 
    :param contig_size: The size of the contigs.
    '''
    fasta_df = load_fasta(os.path.join(GENOMES_PATH, f'{genome_id}.fasta'))
    # If the length of the DataFrame is greater than 1, then there are multiple contigs in the file. 
    assert len(fasta_df) == 1, 'generate_contigs: Input FASTA file does not contain a closed genome. '
    genome = fasta_df.iloc[0].seq

    if contig_size is not None:
        # Select a number of sampling events such that the probability that all nucleotides are "covered" is 0.95.
        # NOTE: This does not account for the fact that the last contig_size nucleotides are less likely to be sampled. 
        n_contigs = int(np.log(0.05) / np.log(1 - contig_size / len(genome)))
        print(f'generate_contigs: Sampling {n_contigs} contigs of length {contig_size} from a genome of size {len(genome)}.')
        start_idxs = np.arange(0, len(genome) - contig_size)
        contigs = []
        for _ in range(n_contigs):
            start_idx = np.random.choice(start_idxs)
            contigs.append(genome[start_idx:start_idx + contig_size])

        df = pd.DataFrame()
        df['seq'] = contigs
        df['header'] = [f'{genome_id}_{i}' for i in range(1, n_contigs + 1)]

    else: # If contig_size is None, use the whole genome. 
        df = pd.DataFrame()
        df['seq'] = [genome]
        df['header'] = [genome_id]

    return df

# Now we can use Prodigal to translate all contigs to amino acid sequences. Will have to be careful with this, as the location where the contig
# has been broken up could have an affect on what gets translated. Honestly, best practice would be to split up each genome many times, with different frame shifts,
# or randomly generating splits.
# 
# Prodigal works by building a GC content profile, and other stuff, of sequences in the FASTA file. So, it assumes that all contigs belong to the same
# genome (I think). Because of this, I think I need to pass in the contigs genome-by-genome. 

def contigs_group_prodigal_output(path:str, genome_id:str=None) -> List[Tuple[str, pd.DataFrame]]:
    '''Prodigal can generate multiple predicted genes from each artificial contig. For each gene, it appends a _{index} to the end
    of the contig ID. Because we want to extract features from each contig individually, we want to group the predicted genes according
    to the contig from which they originated.

    :param path
    :param genome_id
    '''
    fasta_df = load_fasta(path)
    try:
        get_contig_id = lambda header : re.search(f'({genome_id}_\d+)_\d+', header).group(1)
        fasta_df.header = fasta_df.header.apply(get_contig_id)
    except AttributeError:
        # If the size of the DataFrame is one, then it contains the complete genome. 
        # The IDs will not match the first pattern, as there is only one underscore.  
        get_contig_id = lambda header : re.search(f'({genome_id})_\d+', header).group(1)
        fasta_df.header = fasta_df.header.apply(get_contig_id)
    return list(fasta_df.groupby('header'))



def contigs_extract_features(contigs_dfs:List[pd.DataFrame], feature_type:str='aa_3mer', genome_id:str=None, label:str=None):
    '''Generate k-mer features from the list of contig DataFrames (which are output from contigs_split_genome_v*). These
    features will be stored in an HDF file, where each key in the file is data for a particular contig size.
    
    :param contig_dfs: A list of DataFrames, each of which is the output of contigs_split_genome_v*.
    :param feature_type: The feature type to extract from each contig.
    :param genome_id
    '''
    tmp = f'tmp_{genome_id}_{feature_type}' # Need to name temporary file uniquely to avoid conflict with other processes.
    hdf_path = os.path.join(CONTIG_PATH, f'{genome_id}_{feature_type}.h5')
    hdf = pd.HDFStore(hdf_path, 'a')
    k = int(re.search('\d', feature_type).group(0)) # Get the size of the k-mers for the feature type. 

    for contigs_df in contigs_dfs:
        contig_size = len(contigs_df.iloc[0].seq) # Get the length of the contigs stored in the DataFrame. 
        print(f'contigs_get_features: Generating {feature_type} data for contigs of size {contig_size}.')
        
        if 'aa' in feature_type: # Only run Prodigal if we need amino acids. 
            save_fasta(contigs_df, f'{tmp}.fna') # Assuming input contigs are nucleotides. 
            # subprocess.run('prodigal -a tmp.faa -o tmp.gbk -i tmp.fna -q', shell=True, check=True)
            subprocess.run(f'~/prodigal -a {tmp}.faa -o {tmp}.gbk -i {tmp}.fna -q', shell=True, check=True)
            # Prodigal output file will probably have multiple entries per nucleotide contig. We will want to group these. 
            contigs_df_grouped_by_contig = contigs_group_prodigal_output(f'{tmp}.faa', genome_id=genome_id)
        elif 'nt' in feature_type:
            contigs_df_grouped_by_contig = list(contigs_df.groupby('header')) # Header column contains the contig ID.

        
        rows = []
        for contig_id, df in contigs_df_grouped_by_contig:
            row = {'contig_id':contig_id}
            row.update(kmer_count_dataframe(df, k=k))
            rows.append(row)
        hdf[str(contig_size)] = pd.DataFrame(rows).fillna(0).set_index('contig_id')

    print(f'contigs_extract_features: Writing results to {hdf_path}')
    hdf.close()
    # Remove all temporary files. 
    subprocess.run(f'rm {tmp}*', shell=True, check=True)


def contigs_features_from_hdf(hdf:pd.HDFStore, feature_type:str=None, contig_size:int=None, normalize:bool=True) -> np.ndarray:
    '''Extract the features for a set of contigs of the specified length from the HDF file. Clean the features, and normalize
    them if specified.

    :param hdf: An open HDF file handle. 
    :param contig_size: The size of contig for which to retrieve feature data. 
    :param normalize: Whether or not to normalize the feature data by constant-sum scaling the rows. This should pretty much
        always be set to True. 
    '''
    feature_df = hdf.get(str(contig_size)) # Extract the DataFrame from the HDFStore object. 
    feature_df = dataset_clean_features(feature_df, feature_type=feature_type) # Make sure the k-mers align with the data the model was trained on.
    
    # Drop any empty rows. These are the cases in which the contig does not have any k-mer overlap with the training set. 
    empty_rows = feature_df.values.sum(axis=1) == 0
    if np.sum(empty_rows) > 0:
        print(f'contigs_predict: Detected {np.sum(empty_rows)} empty rows in the feature data. Removed rows from feature_df.')
        feature_df = feature_df[~empty_rows]

    features = feature_df.values # Extract the feature values from the DataFrame
    index = feature_df.index # Also get the contig IDs. 
    if normalize: # Normalize across rows, if specified. 
        features = features / features.sum(axis=1, keepdims=True)

    return features, index



def contigs_predict(genome_id:str, model:GeneralClassifier, feature_type:str='aa_3mer'):
    
    hdf_path = os.path.join(CONTIG_PATH, f'{genome_id}_{feature_type}.h5')
    hdf = pd.HDFStore(hdf_path, 'r') # Read in the HDF file where the features are stored. 

    contig_sizes = [int(key.replace('/', '')) for key in hdf.keys()] # Get the list of keys (feature types) stored in the file. 
    
    predictions_df = {'prediction':[], 'contig_size':[], 'contig_id':[]}
    for contig_size in contig_sizes:
        print(f'contigs_predict: Generating predictions for contigs of size {contig_size}.')
        features, index = contigs_features_from_hdf(hdf, feature_type=feature_type, contig_size=contig_size)
        predictions_df['prediction'] += list(model.predict(features).ravel())
        predictions_df['contig_size'] += [contig_size] * len(features)
        predictions_df['contig_id'] += list(index)

    hdf.close()
    return predictions_df