from aerobot.dataset import FeatureDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from aerobot.utils import FEATURE_TYPES, save_results_dict, RESULTS_PATH, DATA_PATH, load_datasets
import argparse
from aerobot.models import evaluate, Nonlinear, GeneralClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, NoReturn
import numpy as np 
import pandas as pd
import os

# TODO: How can I write tests for this? 
LEVELS = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.


def phylogenetic_cross_validation(dataset:FeatureDataset, n_splits:int=25, level:str='Class', model_class:str='nonlinear', binary:bool=False) -> Dict:
    '''Perform cross-validation using holdout sets partitioned according to the specified taxonomic level. For example, if the 
    specified level is 'Class', then the closest relative to any member of the holdout set will be an organism in the same phylum. If 
    the level is 'Family', then the closest relative to any member of the holdout set will be an organism in the same order... etc.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :param n_splits: The number of folds for K-fold cross-validation. This must be no fewer than the number of groups.
    :param n_repeats: The number of times to repeat grouped K-fold cross validation. 
    '''
    groups = dataset.taxonomy(level).values # Extract the taxonomy labels from the labels DataFrame.

    if model_class in ['nonlinear', 'logistic']:
        dataset = dataset_to_numpy(dataset) # Convert the dataset to numpy arrays after extracting the taxonomy labels.
        X, y = dataset['features'], dataset['labels'] # Extract the array and targets from the numpy dataset.
    elif model_class == 'randrel':
        X = dataset['labels'].index.values 
        dataset = dataset_to_numpy(dataset) # Convert the dataset to numpy arrays after extracting the taxonomy labels.
        y = dataset['labels']

    # Filter out anything which does not have a taxonomic classification at the specified level.
    X, y = X[groups != 'no rank'], y[groups != 'no rank']
    groups = groups[groups != 'no rank']

    # GroupShuffleSplit generates a sequence of randomized partitions in which a subset of groups are held out for each split.
    group_shuffle_split = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)
    scores = []
    for train_idxs, test_idxs in group_shuffle_split.split(X, y, groups=groups):
        if model_class == 'nonlinear':
            model = GeneralClassifier(model_class=Nonlinear, params={'input_dim':X.shape[-1], 'n_classes':3 if not binary else 2})
            model.fit(X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs])
        elif model_class == 'logistic':
            model = GeneralClassifier(model_class=LogisticRegression) # , params={'max_iter':100000000})
            model.fit(X[train_idxs], y[train_idxs])
        elif model_class == 'randrel':
            model = GeneralClassifier(model_class=RandomRelative, params={'level':level, 'n_classes':3 if not binary else 2}, normalize=False) # Don't need to fit this model.
            model.fit(X[train_idxs], y[train_idxs]) # Really just to populate the n_classes attribute.
        # Evaluate the trained model on the holdout set.
        results = evaluate(model, X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs])
        scores.append(results['validation_acc']) # Store the balanced accuracy.

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic', 'randrel'], help='The type of model to train.')
    parser.add_argument('feature-type', type=str, default=None, choices=FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--n-splits', default=25, type=int, help='The number of folds for K-fold cross validation.')
    parser.add_argument('--n-classes', default=3, type=int)

    args = parser.parse_args()
    model_class = getattr(args, 'model-class') # Get the model class to run.
    feature_type = getattr(args, 'feature-type') # Get the model class to run.

    datasets = load_datasets(feature_type)
    dataset = datasets['training'].concat(datasets['validation'])
        # Should probably report the standard deviation, mean, and standard error for each run.
        scores = dict()
        for level in LEVELS:
            print(f'Performing phylogeny-based cross-validation with {level.lower()}-level holdout set.')
            # Retrieve the balanced accuracy scores for the level. 
            scores[level] = phylogenetic_cross_validation(dataset, level=level, model_class=model_class, n_splits=args.n_splits, binary=args.binary)

        results = {'scores':scores}
        # Add other relevant information to the results dictionary. Make sure to not include tax_data if it's there.
        results['feature_type'] = feature_type # Add feature type to the results.
        results['model_class'] = model_class
        results['binary'] = args.binary

        task = 'binary' if args.binary else 'ternary'

        result_path = os.path.join(RESULTS_PATH, f'phylo_cv_{model_class}_{feature_type}_{task}.json')
        print(f'\nWriting results to {result_path}.')
        save_results_dict(results, result_path)

