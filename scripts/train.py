'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import numpy as np
import tqdm
import os
from aerobot.io import DATA_PATH, MODELS_PATH, RESULTS_PATH, FEATURE_TYPES, save_results_dict, read_params
from aerobot.dataset import dataset_load_training_validation
from aerobot.models import GeneralClassifier, evaluate, Nonlinear
from sklearn.linear_model import LogisticRegression
# from joblib import Parallel, delayed
from os import path
import argparse
from typing import Dict, NoReturn
import time
import pickle

def print_summary(results:Dict) -> NoReturn:
    '''Print a summary of the evaluation results to the terminal.'''
    task = 'binary' if results['binary'] else 'ternary' # Get whether the classification task was ternary or binary.
    print('\nResults of training a', results['model_class'], f'classifier for {task} classification.')
    print('\tFeature type:', results['feature_type'])
    print('\tBalanced accuracy on training dataset:', results['training_acc'])
    print('\tBalanced accuracy on validation dataset:', results['validation_acc'])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic'], help='The type of model to train.')
    parser.add_argument('--feature-type', '-f', type=str, default=None, nargs='+', choices=FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--binary', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    # Optional parameters for Nonlinear classifiers. 
    parser.add_argument('--n-epochs', default=100, type=int, help='The maximum number of epochs to train the Nonlinear classifier.') 
    parser.add_argument('--lr', default=0.0001, type=float, help='The learning rate for training the Nonlinear classifier.') 
    parser.add_argument('--weight-decay', default=0.01, type=float, help='The L2 regularization penalty to be passed into the Adam optimizer of the Nonlinear classifier.') 
    parser.add_argument('--batch-size', default=16, type=int, help='The size of the batches for Nonlinear classifier training')  
    parser.add_argument('--hidden-dim', default=512, type=int, help='The number of nodes in the second linear layer of the Nonlinear classifier.')
    # Optional parameters for LogisticRegression classifiers.
    parser.add_argument('--C', default=100, type=float, help='Inverse of regularization strength for the LogisticRegression classifier' ) 
    parser.add_argument('--penalty', default='l2', type=str, help='The norm of the penalty term for the LogisticRegression classifier.') 
    parser.add_argument('--max-iter', default=100000, type=int, help='Maximum number of iterations for the LogisticRegression classifier.')

    args = parser.parse_args()


    feature_types = FEATURE_TYPES if args.feature_type is None else args.feature_type

    for feature_type in feature_types:
        t1 = time.perf_counter()
        # Load the cleaned-up training and validation datasets.
        training_dataset, validation_dataset = dataset_load_training_validation(feature_type, binary=args.binary, to_numpy=True)
        # Unpack the numpy arrays stored in the dictionary. 
        X, y, X_val, y_val = training_dataset['features'], training_dataset['labels'], validation_dataset['features'], validation_dataset['labels']

        model_class = getattr(args, 'model-class') # Extract the specified model class.
        params = read_params(args, model_class=model_class) # Read in model parameters from the command-line arguments.
        if model_class == 'nonlinear':
            params.update({'input_dim':X.shape[-1], 'n_classes':3 if not args.binary else 2}) # Make sure input dimensions are included. 
            model = GeneralClassifier(model_class=Nonlinear, params=params)
            model.fit(X, y, X_val=X_val, y_val=y_val)
        elif model_class == 'logistic':
            model = GeneralClassifier(model_class=LogisticRegression, params=params)
            model.fit(X, y)
        
        results = evaluate(model, X, y, X_val, y_val)
        # Store the model parameters in the results dictionary. 
        results.update(params)
        results['feature_type'] = feature_type # Add feature type to the results.
        results['model_class'] = model_class
        results['binary'] = args.binary

        print_summary(results) # Print a summary of the training run to the terminal. 

        task = 'binary' if args.binary else 'ternary'

        result_path = os.path.join(RESULTS_PATH, f'train_{model_class}_{feature_type}_{task}.json')
        print(f'\nWriting results to {result_path}.')
        save_results_dict(results, result_path)

        model_path = os.path.join(MODELS_PATH, f'{model_class}_{feature_type}_{task}.joblib')
        print(f'Saving trained model to {model_path}.')
        model.save(model_path)
        
        t2 = time.perf_counter()
        print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

