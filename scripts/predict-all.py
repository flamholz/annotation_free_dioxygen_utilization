import pandas as pd
import numpy as np
from aerobot.utils import FEATURE_TYPES
from aerobot.dataset import FeatureDataset
from aerobot.models import BaseClassifier
from sklearn.linear_model import LogisticRegression
import os
import argparse
from typing import Dict, NoReturn, Tuple
import pickle
import glob
from warnings import simplefilter
import re 

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_model(feature_type, models_dir):
    model_name = f'nonlinear_{feature_type}_ternary.joblib'
    model_path = os.path.join(models_dir, model_name)
    return model_name, BaseClassifier.load(model_path)


def is_kmer_feature_type(feature_type:str):
    if feature_type is None:
        return False
    return re.match(r'(nt|aa|cds)_(\d)mer', feature_type) is not None


def clean_features(feature_type:str, order:list):
    # Remove ambiguous bases and amino acids. The removed symbols indicate that the base or amino acid is unknown, and 
    # do not occur very frequently. 
    def is_valid_column(col:str) -> bool:
        ref = AMINO_ACIDS if re.match(r'aa_(\d)mer', feature_type) else NUCLEOTIDES
        return np.all([elem in ref for elem in col])
    print(feature_type)
    if is_kmer_feature_type(feature_type): 
        print(f'it is a kmer {feature_type}')
        order = [f for f in order if is_valid_column(f)]
    return order

# Load the feature orders for consistency, i.e. ensuring the feature orders are the same as the vectors the models are trained on. 
FEATURE_ORDERS = dict()
for feature_type in FEATURE_TYPES:
    FEATURE_ORDERS[feature_type] = np.loadtxt(io.StringIO(resources.files('aerobot.data').joinpath(f'features/{feature_type}.txt').read_text()), dtype=FEATURE_COLUMN_DTYPES[feature_type]) 
FEATURE_ORDERS = {feature_type:clean_features(feature_type, order) for feature_type, order in FEATURE_ORDERS.items()}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--models-dir', type=str, default='../models')
    parser.add_argument('--data-dir', type=str, default='../data/') # Must be ah=n HDF5 or CSV file. 
    parser.add_argument('--results-dir', type=str, default='../results/12_16_2025/')
    args = parser.parse_args()

    input_file_names = ['training_datasets.h5', 'testing_datasets.h5']
    input_paths = [os.path.join(args.data_dir, input_file_name) for input_file_name in input_file_names] 

    for feature_type in FEATURE_TYPES:
        for input_path in input_paths:
                try:
                    model_name, model = get_model(feature_type, args.models_dir)
                    dataset = FeatureDataset(input_path, feature_type=feature_type)
                    predictions_df = model.predict(dataset)

                    dataset_type = os.path.basename(input_path).replace('_datasets.h5', '')
                    output_file_name = f'{dataset_type}_{model_name}' # Model name already contains the feature type.
                    output_path = os.path.join(args.results_dir, output_file_name)
                    predictions_df.to_csv(output_path)
                    print(f'Output written to {output_path}')
                except Exception as err:
                    print(f'Failed on feature type {feature_type} and input {input_path}.')
                    print(err)




