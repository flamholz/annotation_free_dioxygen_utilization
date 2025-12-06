import pandas as pd
import numpy as np
from aerobot.utils import FEATURE_TYPES, AMINO_ACIDS, NUCLEOTIDES
from aerobot.dataset import FeatureDataset
from aerobot.models import BaseClassifier
from sklearn.linear_model import LogisticRegression
import os
import argparse
from typing import Dict, NoReturn, Tuple
import pickle
import glob
from warnings import simplefilter
import io
import re 

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_model(feature_type, models_dir):
    model_name = f'nonlinear_{feature_type}_ternary.joblib'
    model_path = os.path.join(models_dir, model_name)
    return model_name.replace('.joblib', ''), BaseClassifier.load(model_path)


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
                    output_file_name = f'{dataset_type}_{model_name}.csv' # Model name already contains the feature type.
                    output_path = os.path.join(args.results_dir, output_file_name)
                    predictions_df.to_csv(output_path)
                    print(f'Output written to {output_path}')
                except Exception as err:
                    print(f'Failed on feature type {feature_type} and input {input_path}.')
                    print(err)




