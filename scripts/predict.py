import pandas as pd
import numpy as np
from aerobot.utils import MODELS_PATH, RESULTS_PATH, FEATURE_TYPES
from aerobot.dataset import FeatureDataset
from aerobot.models import BaseClassifier
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
    parser.add_argument('--model-path', '-m', type=str, help='Name of the pre-trained model to use. Expected to be stored in the models directory.')
    parser.add_argument('--input-path', '-i', type=str, help='Path to the dataset. This should be stored in an H5 file.')
    parser.add_argument('--feature-type', '-f', type=str, default='aa_3mer', choices=FEATURE_TYPES, help='The feature type of the data.')
    parser.add_argument('--output-path', '-o', type=str, default=None, help='The location to which the predictions will be written.')

    args = parser.parse_args()
    t1 = time.perf_counter()

    dataset = FeatureDataset(args.input_path, feature_type=args.feature_type) # Make sure the feature ordering is correct. 
    X, y = dataset.to_numpy() # Extract the raw data from the input DataFrame.
    model = BaseClassifier.load(args.model_path)
    y_pred = model.predict(X)

    results = pd.DataFrame(index=dataset.index()) # Make sure to add the index back in!
    results['prediction'] = y_pred.ravel() # Ravel because Nonlinear output is a column vector. 

    print(f'\nWriting results to {args.output_path}.')
    results.to_csv(args.output_path)
    
    t2 = time.perf_counter()
    print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

