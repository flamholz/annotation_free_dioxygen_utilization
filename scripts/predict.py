'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import numpy as np
import tqdm
from aerobot.io import MODELS_PATH, RESULTS_PATH, FEATURE_TYPES
from aerobot.dataset import dataset_clean_features
from aerobot.models import GeneralClassifier
from sklearn.linear_model import LogisticRegression
import os
import argparse
from typing import Dict, NoReturn
import time
import pickle
from warnings import simplefilter

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Name of the pre-trained model to use. Expected to be stored in the models directory.')
    parser.add_argument('--input-path', '-i', type=str, help='Path to the data on which to run the trained model. This should be in CSV format.')
    parser.add_argument('--feature-type', '-f', type=str, default='aa_3mer', choices=FEATURE_TYPES, help='The feature type of the data.')
    parser.add_argument('--output-path', '-o', type=str, default=None, help='The location to which the predictions will be written.')

    args = parser.parse_args()
    t1 = time.perf_counter()

    data = pd.read_csv(args.input_path, index_col=0) # Need to preserve the index, which is the genome ID.
    data = dataset_clean_features(data, feature_type=args.feature_type) # Make sure the feature ordering is correct. 

    model_path = os.path.join(MODELS_PATH, args.model)
    model = GeneralClassifier.load(model_path) # Load the trained model. 
    X = data.values # Extract the raw data from the input DataFrame.
    y_pred = model.predict(X)

    results = pd.DataFrame(index=data.index) # Make sure to add the index back in!
    results['prediction'] = y_pred.ravel() # Ravel because Nonlinear output is a column vector. 

    print(f'\nWriting results to {args.output_path}.')
    results.to_csv(args.output_path)
    
    t2 = time.perf_counter()
    print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

