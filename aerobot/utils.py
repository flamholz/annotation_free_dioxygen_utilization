'''Functions for reading and writing data from files, as well as some utilities for the command-line interface.'''
import pandas as pd
import numpy as np
import os
from typing import Dict, NoReturn, Tuple
import json
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
import pickle

ROOT_PATH, _ = os.path.split(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT_PATH, '..', 'data')
MODELS_PATH = os.path.join(ROOT_PATH, '..', 'models')
SCRIPTS_PATH = os.path.join(ROOT_PATH, '..', 'scripts')
FIGURES_PATH = os.path.join(ROOT_PATH, '..', 'figures')
RESULTS_PATH = os.path.join(ROOT_PATH, '..', 'results')

# Paths to subdirectories within the data directory. 
CONTIGS_PATH = os.path.join(DATA_PATH, 'contigs')
FEATURES_PATH = os.path.join(DATA_PATH, 'features')
RNA16S_PATH = os.path.join(DATA_PATH, 'rna16s')


FEATURE_TYPES = ['ko', 'ko_terminal_oxidase_genes']
FEATURE_TYPES += ['chemical']
FEATURE_TYPES += ['embedding_genome', 'embedding_oxygen_genes'] #, 'embedding_rna16s'] 
FEATURE_TYPES += ['number_of_genes', 'number_of_oxygen_genes', 'percent_oxygen_genes']
FEATURE_TYPES += [f'nt_{i}mer' for i in range(1, 5)]
FEATURE_TYPES += [f'cds_{i}mer' for i in range(1, 6)]
FEATURE_TYPES += [f'aa_{i}mer' for i in range(1, 4)]

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U']
NUCLEOTIDES = ['A', 'C', 'T', 'G']


class NumpyEncoder(json.JSONEncoder):
    '''Encoder for converting numpy data types into types which are JSON-serializable. Based
    on the tutorial here: https://medium.com/@ayush-thakur02/understanding-custom-encoders-and-decoders-in-pythons-json-module-1490d3d23cf7'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class NumpyDecoder(json.JSONDecoder):
    # TODO: This might be overkill, but for converting JSON stuff back to numpy objects.
    pass


def save_results_dict(results:Dict, path:str) -> NoReturn:
    '''Write a dictionary of results to the output path.

    :param results: A dictionary containing results from a model run, cross-validation, etc.
    :param path: The path to write the results to. 
    '''
    with open(path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)


def load_results_dict(path:str) -> Dict:
    '''Read in a dictionary of results (e.g. from the evaluate function) stored either as a JSON or pickle file.

    :param path: The location of the file containing the results.
    :return: A dictionary containing the results. 
    '''
    fmt = path.split('.')[-1]
    assert fmt in ['pkl', 'json'], 'read_results_dict: File is not in a supported format. Must be either .pkl or .json.'
    if fmt == 'json':
        with open(path, 'r') as f:
            results = json.load(f)
    elif fmt == 'pkl':
        with open(path, 'rb') as f:
            results = pickle.load(f)
    return results


def save_hdf(datasets:Dict[str, pd.DataFrame], path:str)-> NoReturn:
    '''Save a dictionary of pandas DataFrames as an HD5 file at the specified output path.'''

    store = pd.HDFStore(path, 'a')
    for key, value in datasets.items():
        store.put(key, value)
    store.close()


# NOTE: Are the no-rank things getting included?
def training_testing_validation_split(datasets:Dict[str, pd.DataFrame]):
    '''Split datasets into training and validation sets by phylogeny.'''
    np.random.seed(42) # For reproducibility.
    metadata = datasets['metadata']
    
    # Group IDs by phylogenetic class. Convert to a dictionary mapping class to a list of indices.
    # Took out the "include_groups" here because the version of Python which is used in the aerobot-16s environment doesn't support it. 
    ids_by_class = metadata.groupby('Class').apply(lambda x: x.index.tolist()).to_dict()

    testing_ids = []
    for class_, ids in ids_by_class.items():

        n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
        selected_ids = np.random.choice(ids, n, replace=False)
        remaining_ids = [id_ for id_ in ids if id_ not in selected_ids]
        testing_ids.extend(selected_ids)
        ids_by_class[class_] = remaining_ids # Make sure only the remaining IDs are left. 

    # Take 20 percent of the remaining IDs for each class for the validation set. 
    validation_ids = []
    for class_, ids in ids_by_class.items():
        n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
        selected_ids = np.random.choice(ids, n, replace=False)
        remaining_ids = [id_ for id_ in ids if id_ not in selected_ids]
        validation_ids.extend(selected_ids)
        ids_by_class[class_] = remaining_ids # Make sure only the remaining IDs are left.

    # Add all remaining IDs to the training dataset. 
    training_ids =  []
    for ids in ids_by_class.values():
        training_ids.extend(ids)

    # Split the concatenated dataset back into training and validation sets
    training_datasets, testing_datasets, validation_datasets = dict(), dict(), dict()
    for feature_type, dataset in datasets.items():
        training_datasets[feature_type] = dataset[dataset.index.isin(training_ids)]
        testing_datasets[feature_type] = dataset[dataset.index.isin(testing_ids)]
        validation_datasets[feature_type] = dataset[dataset.index.isin(validation_ids)]

    return training_datasets, testing_datasets, validation_datasets

    



