import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple, List
from aerobot.utils import FEATURE_TYPES, AMINO_ACIDS, NUCLEOTIDES
import re
import copy


def is_kmer_feature_type(feature_type:str):
    if feature_type is None:
        return False
    if re.match(r'aa_(\d)mer', feature_type) is not None:
        return True
    # NOTE: nt_ is in "percent_oxygen_genes," so need to be careful here!
    if re.match(r'nt_(\d)mer', feature_type) is not None:
        return True 
    if re.match(r'cds_(\d)mer', feature_type) is not None:
        return True
    else:
        return False 


def get_feature_order(feature_type:str) -> List[str]:
    # Load the training dataset columns, which are used as the reference columns.
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    order = pd.read_hdf(os.path.join(data_path, 'training_datasets.h5'), key=feature_type, stop=1).columns 
    
    # Remove ambiguous bases and amino acids. The removed symbols indicate that the base or amino acid is unknown, and 
    # do not occur very frequently. 
    def is_valid_column(col:str) -> bool:
        ref = AMINO_ACIDS if re.match(r'aa_(\d)mer', feature_type) else NUCLEOTIDES
        return np.all([elem in ref for elem in col])
    
    if is_kmer_feature_type(feature_type): 
        order = [f for f in order if is_valid_column(f)]
    return order

    
def order_features(features:pd.DataFrame, feature_type:str, verbose:bool=False) -> pd.DataFrame:
    '''Ensure that the order of the feature columns in the input DataFrame matches the order of the feature columns in the 
    training dataset. 

    :param features: A DataFrame containing the genome features. 
    :param feature_type: The type of the features contained in the DataFrame. 
    ''' 
    # Don't try to order the 16S embedding features, or if there are no loaded features (which is true when doing 
    # phylogenetic cross-validation). 
    if (feature_type == 'embedding_rna16s') or (feature_type is None):
        return features 

    order = get_feature_order(feature_type)
    missing_cols = [c for c in order if c not in features.columns]
    
    if verbose: # Printing some stuff for debugging purposes. 
        print('order_features:', len(missing_cols), 'columns missing from the', feature_type, 'data.')
        print('order_features:', len([c for c in features.columns if c not in order]), 'extraneous columns in the', feature_type, 'data.')

    # If the data is missing a feature, fill it in with zeros.
    filler = pd.DataFrame(0, index=features.index, columns=missing_cols)
    features = pd.concat([features, filler], axis=1)

    return features[order]


class FeatureDataset():

    def __init__(self, path:str, feature_type:str=None, normalize:bool=True):

        self.feature_type = feature_type

        _, file_type = os.path.splitext(path)

        if file_type == '.h5':
            self.metadata = pd.read_hdf(path, key='metadata')
            self.features =  pd.DataFrame({'genome_id':self.metadata.index}, index=self.metadata.index) if (feature_type is None) else pd.read_hdf(path, key=feature_type) 
            self.labeled = 'physiology' in self.metadata.columns
            # Make sure the row ordering in the features and metadata is consistent. 
            self.features, self.metadata = self.features.align(self.metadata, join='left', axis=0)
            # Get rid of any NaNs in the data.
            self.features = self.features.fillna(0)

        elif file_type == '.csv':
            self.features = pd.read_csv(path, index_col=0) 
            self.metadata = None
            self.labeled = False

        self.ids = self.features.index        
        self.features = order_features(self.features, feature_type) # Make sure the column ordering of the feature columns is consistent. 
        # If the normalize option is specified, and the feature type needs to be normalized, then normalize the rows. 
        if normalize and is_kmer_feature_type(feature_type):
            self.features = self.features.apply(lambda row : row / row.sum(), axis=1)

        self.shape = self.features.shape


    def taxonomy(self, level:str):
        if level in self.metadata:
            return self.metadata[level]

    def __getitem__(self, key):
        dataset = copy.deepcopy(self) 

        if (type(key) == int) or (type(key[0]) in [bool, np.int64]):
            dataset.features = self.features.iloc[key]
            if self.metadata is not None:
                dataset.metadata = self.metadata.iloc[key]

        elif (type(key) == bool) or (type(key[0]) in [bool, np.bool_]):
            dataset.features = self.features[key]
            if self.metadata is not None:
                dataset.metadata = self.metadata[key] 

        elif (type(key) == str) or (type(key[0]) == str):
            dataset.features = self.features.loc[key]
            if self.metadata is not None:
                dataset.metadata = self.metadata.loc[key] 
            
        else:
            raise ValueError(f'FeatureDataset.__getitem__: Key for indexing is not of the appropriate type ({type(key[0])})')

        dataset.shape = dataset.features.shape
        return dataset  

    def __array__(self):
        return self.features.values
    
    def __iter__(self):
        return iter(self.features.values)

    def index(self):
        return self.features.index

    def __len__(self):
        return len(self.features)

    def labels(self, n_classes:int=3):
        # Handle the case of the FeatureDataset being unlabeled.
        if not self.labeled:
            return None 

        labels = self.metadata.physiology
        labels = labels.str.lower()
        if n_classes == 2:
            labels = labels.replace({'aerobe':'tolerant', 'facultative':'tolerant', 'anaerobe':'intolerant'})
        # elif n_classes == 3:
        #     labels = labels.replace({'Aerobe':'aerobe', 'Facultative':'facultative', 'Anaerobe':'anaerobe'})
        return labels.values
  
    def to_numpy(self, n_classes:int=3):
        X = self.features.values # .astype(np.float32)
        y = self.labels(n_classes=n_classes)
        return X, y 

    def align(self, dataset):
        # TODO: Check that this successfully modifies the other FeatureDataset inplace.
        # TODO: Should I do a left join or inner join? Or something else?
        self.features, dataset.features = self.features.align(dataset.features, join='outer', axis=0)
        self.metadata, dataset.metadata = self.metadata.align(dataset.metadata, join='outer', axis=0)

    def concat(self, dataset):
        # TODO: Maybe add a check to make sure both datasets are normalized?
        if dataset.feature_type != self.feature_type:
            raise ValueError('FeatureDatasets must contain data of the same feature type to be concatenated.')
        self.features = pd.concat([self.features, dataset.features], axis=0)
        self.metadata = pd.concat([self.metadata, dataset.metadata], axis=0)
        self.shape = self.features.shape
        return self


def load_datasets(feature_type:str, data_path:str=None):
    '''Load the training, testing, and validation datasets generated by the build.py (or build-rna16s.py) script.'''
    datasets = dict()
    for dataset_type in ['training', 'testing', 'validation']:
        if feature_type == 'embedding_rna16s':
            datasets[dataset_type] = FeatureDataset(os.path.join(data_path, 'rna16s', f'{dataset_type}_dataset.h5'), feature_type=feature_type)
        else:
            datasets[dataset_type] = FeatureDataset(os.path.join(data_path, f'{dataset_type}_datasets.h5'), feature_type=feature_type)
    return datasets





