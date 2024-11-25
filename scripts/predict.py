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
from warnings import simplefilter

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', '-m', type=str)
    parser.add_argument('--input-path', '-i', type=str) # Must be ah=n HDF5 or CSV file. 
    parser.add_argument('--feature-type', '-f', type=str, default='aa_3mer', choices=FEATURE_TYPES + ['embedding_rna16s'])
    parser.add_argument('--output-path', '-o', type=str, default=None)

    args = parser.parse_args()
    
    model = BaseClassifier.load(args.model_path)

    dataset = FeatureDataset(args.input_path, feature_type=args.feature_type)
    predictions_df = model.predict(dataset)

    if dataset.labeled:
        print('Balanced accuracy:', model.accuracy(dataset))

    if args.output_path is None:
        input_file_name = os.path.basename(args.input_path)
        input_file_name, _ = os.path.splitext(input_file_name)
        output_path = os.path.join('.', 'predict_' + input_file_name + '.csv')
    else:
        output_path = args.output_path

    print(f'\nWriting results to {output_path}.')
    predictions_df.to_csv(output_path)




