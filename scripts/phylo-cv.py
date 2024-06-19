from aerobot.dataset import load_datasets, FeatureDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from aerobot.utils import FEATURE_TYPES, save_results_dict, RESULTS_PATH, DATA_PATH
import argparse
from aerobot.models import NonlinearClassifier, LogisticClassifier, RandomRelativeClassifier
from typing import Dict, List, NoReturn
import numpy as np 
import pandas as pd
import os

# TODO: How can I write tests for this? 
LEVELS = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.


def phylogenetic_cross_validation(dataset:FeatureDataset, n_splits:int=25, level:str='Class', model_class:str='nonlinear', n_classes:int=3) -> Dict:
    '''Perform cross-validation using holdout sets partitioned according to the specified taxonomic level. For example, if the 
    specified level is 'Class', then the closest relative to any member of the holdout set will be an organism in the same phylum. If 
    the level is 'Family', then the closest relative to any member of the holdout set will be an organism in the same order... etc.'''
    X, y = dataset.to_numpy()

    groups = dataset.taxonomy(level).values # Extract the taxonomy labels from the labels DataFrame.
    # Filter out anything which does not have a taxonomic classification at the specified level.
    X, y = X[groups != 'no rank'], y[groups != 'no rank']
    # The RandomRelativeClassifier uses genome IDs as input, so need to adjust the X array accordingly. 
    if model_class == 'randrel':
        X = dataset.index()
    groups = groups[groups != 'no rank']

    # GroupShuffleSplit generates a sequence of randomized partitions in which a subset of groups are held out for each split.
    test_accs = []
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)
    for train_idxs, test_idxs in gss.split(X, y, groups=groups):

        if model_class == 'nonlinear':
            model = NonlinearClassifier(input_dim=dataset.dims(), output_dim=n_classes)
            # For the Nonlinear classifier, need to further subdivide the training data into a training and validation set.
            val_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * 0.2), replace=False)
            train_idxs = [i for i in train_idxs if i not in val_idxs]
            model.fit(X[train_idxs], y[train_idxs], X_val=X[val_idxs], y_val=y[val_idxs])

        elif model_class == 'logistic':
            model = LogisticClassifier(n_classes=n_classes)
            model.fit(X[train_idxs], y[train_idxs])

        elif model_class == 'randrel':
            model = RandomRelativeClassifier(level=level)
            model.fit(dataset.metadata)

        test_accs.append(model.balanced_accuracy(X, y)) 

    return test_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic', 'randrel'], help='The type of model to train.')
    parser.add_argument('feature-type', type=str, default=None, choices=FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--n-splits', default=5, type=int, help='The number of folds for K-fold cross validation.')
    parser.add_argument('--n-classes', default=3, type=int)

    args = parser.parse_args()
    model_class = getattr(args, 'model-class') # Get the model class to run.
    feature_type = getattr(args, 'feature-type') # Get the model class to run.

    datasets = load_datasets(feature_type)
    dataset = datasets['training'].concat(datasets['validation'])

    results = dict()
    for level in LEVELS:
        print(f'Performing phylogeny-based cross-validation with {level.lower()}-level holdout set.')
        # Retrieve the balanced accuracy scores for the level. 
        test_accs = phylogenetic_cross_validation(dataset, level=level, model_class=model_class, n_splits=args.n_splits, n_classes=args.n_classes)
        results[f'{level.lower()}_test_accs'] = test_accs
    
    # Add other relevant information to the results dictionary. Make sure to not include tax_data if it's there.
    results['feature_type'] = feature_type # Add feature type to the results.
    results['model_class'] = model_class
    results['n_classes'] = args.n_classes

    task = 'binary' if args.n_classes == 2 else 'ternary'
    if model_class == 'randrel':
        result_path = os.path.join(RESULTS_PATH, f'phylo_cv_{model_class}_{task}.json')
    else:
        result_path = os.path.join(RESULTS_PATH, f'phylo_cv_{model_class}_{feature_type}_{task}.json')
    print(f'\nWriting results to {result_path}.')
    save_results_dict(results, result_path)

