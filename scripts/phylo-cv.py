from aerobot.dataset import load_datasets, FeatureDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from aerobot.utils import FEATURE_TYPES, save_results_dict
import argparse
from aerobot.models import NonlinearClassifier, LogisticClassifier, RandomRelativeClassifier, LinearClassifier
from typing import Dict, List, NoReturn
import numpy as np 
import pandas as pd
import os

def load_datasets(feature_type:str, data_path:str=None):
    '''Load the training, testing, and validation datasets generated by the build.py (or build-rna16s.py) script.'''
    datasets = dict()
    for dataset_type in ['training', 'testing', 'validation']:
            datasets[dataset_type] = FeatureDataset(os.path.join(data_path, f'{dataset_type}_datasets.h5'), feature_type=feature_type)
    return datasets


def phylogenetic_cross_validation(dataset:FeatureDataset, n_splits:int=25, level:str='Class', model_class:str='nonlinear', n_classes:int=3) -> Dict:
    '''Perform cross-validation using holdout sets partitioned according to the specified taxonomic level. For example, if the 
    specified level is 'Class', then the closest relative to any member of the holdout set will be an organism in the same phylum. If 
    the level is 'Family', then the closest relative to any member of the holdout set will be an organism in the same order... etc.'''
    # Filter out anything which does not have a taxonomic classification at the specified level.
    groups = dataset.taxonomy(level).fillna('no rank').values 
    dataset = dataset[groups != 'no rank']
    groups = groups[groups != 'no rank']
    
    # GroupShuffleSplit generates a sequence of randomized partitions in which a subset of groups are held out for each split.
    test_accs = []
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)
    for train_idxs, test_idxs in gss.split(dataset, groups=groups):

        if model_class == 'nonlinear':
            model = NonlinearClassifier(input_dim=dataset.shape()[-1], output_dim=n_classes)
            # For the Nonlinear classifier, need to further subdivide the training data into a training and validation set.
            val_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * 0.2), replace=False)
            train_idxs = [i for i in train_idxs if i not in val_idxs]
            model.fit(dataset[train_idxs], dataset[val_idxs])
        
        elif model_class == 'linear':
            model = LinearClassifier(input_dim=dataset.shape()[-1], output_dim=n_classes)
            # For the Linear classifier, need to further subdivide the training data into a training and validation set.
            val_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * 0.2), replace=False)
            train_idxs = [i for i in train_idxs if i not in val_idxs]
            model.fit(dataset[train_idxs], dataset[val_idxs])

        elif model_class == 'logistic':
            model = LogisticClassifier(n_classes=n_classes)
            model.fit(dataset[train_idxs])

        elif model_class == 'randrel':
            model = RandomRelativeClassifier(level=level, n_classes=n_classes)
            model.fit(dataset)

        test_accs.append(model.accuracy(dataset)) 

    return test_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic', 'randrel', 'linear'])
    parser.add_argument('--feature-type', '-f', default=None, type=str)
    parser.add_argument('--n-splits', default=5, type=int)
    parser.add_argument('--n-classes', default=3, type=int)
    parser.add_argument('--data-path', default='../data/')
    parser.add_argument('--output-path', '-o', default='.')

    args = parser.parse_args()
    model_class = getattr(args, 'model-class') # Get the model class to run.

    datasets = load_datasets(args.feature_type, data_path=args.data_path)
    dataset = datasets['training'].concat(datasets['validation'])

    results = dict()

    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.
    for level in levels:
        print(f'\nPerforming phylogeny-based cross-validation with {level.lower()}-level holdout set.')
        # Retrieve the balanced accuracy scores for the level. 
        test_accs = phylogenetic_cross_validation(dataset, level=level, model_class=model_class, n_splits=args.n_splits, n_classes=args.n_classes)
        results[f'{level.lower()}_test_accs'] = test_accs
    
    # Add other relevant information to the results dictionary. Make sure to not include tax_data if it's there.
    results['feature_type'] = feature_type if model_class != 'randrel' else None 
    results['model_class'] = model_class
    results['n_classes'] = args.n_classes

    output_path = '.' if (args.output_path is None) else args.output_path
    task = 'binary' if args.n_classes == 2 else 'ternary'
    output_file_name = f'phylo_cv_{model_class}_{task}.json' if (model_class == 'randrel') else f'phylo_cv_{model_class}_{args.feature_type}_{task}.json'
    output_path = os.path.join(output_path, output_file_name)

    print(f'\nWriting results to {output_path}.')
    save_results_dict(results, output_path)

