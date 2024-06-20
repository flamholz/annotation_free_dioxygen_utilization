from aerobot.models import LinearClassifier, NonlinearClassifier, LogisticClassifier
import argparse
from aerobot.utils import FEATURE_TYPES, DATA_PATH, RESULTS_PATH, MODELS_PATH, save_results_dict
import time
from aerobot.dataset import FeatureDataset, load_datasets
import os
import numpy as np
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'linear', 'logistic'], help='The type of model to train.')
    parser.add_argument('feature-type', type=str, default=None, choices=FEATURE_TYPES + ['embedding_rna16s'], help='The feature type on which to train.')
    parser.add_argument('--n-classes', default=3, type=int, help='Number of classes.')

    t1 = time.perf_counter()
    # torch.manual_seed(42)
    args = parser.parse_args()

    feature_type = getattr(args, 'feature-type')
    model_class = getattr(args, 'model-class')

    datasets = load_datasets(feature_type) 

    X_train, y_train = datasets['training'].to_numpy(n_classes=args.n_classes)
    X_val, y_val = datasets['validation'].to_numpy(n_classes=args.n_classes)
    X_test, y_test = datasets['testing'].to_numpy(n_classes=args.n_classes)

    if model_class == 'nonlinear':
        model = NonlinearClassifier(output_dim=args.n_classes, input_dim=X_train.shape[-1])
        model.fit(X_train, y_train, X_val, y_val)
    elif model_class == 'linear':
        model = LinearClassifier(output_dim=args.n_classes, input_dim=X_train.shape[-1])
        model.fit(X_train, y_train, X_val, y_val)
    elif model_class == 'logistic':
        model = LogisticClassifier(n_classes=args.n_classes)
        model.fit(X_train, y_train)

    # Using default parameters, so no need to put them in the results dictionary. 
    results = dict()
    results['n_classes'] = args.n_classes
    results['train_acc'] = model.balanced_accuracy(X_train, y_train)
    results['val_acc'] = model.balanced_accuracy(X_val, y_val)
    results['test_acc'] = model.balanced_accuracy(X_test, y_test)
    results['confusion_matrix'] = model.confusion_matrix(X_test, y_test).ravel()
    results['val_accs'] = None if not hasattr(model, 'val_accs') else model.val_accs
    results['train_accs'] = None if not hasattr(model, 'train_accs') else model.train_accs
    results['val_losses'] = None if not hasattr(model, 'val_losses') else model.val_losses
    results['train_losses'] = None if not hasattr(model, 'train_losses') else model.val_accs
   
    print('Balanced accuracy on training dataset:', results['train_acc'])
    print('Balanced accuracy on testing dataset:', results['test_acc'])
    print('Balanced accuracy on validation dataset:', results['val_acc'])
    
    task = 'binary' if args.n_classes == 2 else 'ternary'
    result_path = os.path.join(RESULTS_PATH, f'train_{model_class}_{feature_type}_{task}.json')
    print(f'\nWriting results to {result_path}.')
    save_results_dict(results, result_path)
    model_path = os.path.join(MODELS_PATH, f'{model_class}_{feature_type}_{task}.joblib')
    print(f'Saving trained model to {model_path}.')
    model.save(model_path)

    t2 = time.perf_counter()
    print(f'\nModel training complete in {np.round(t2 - t1, 2)} seconds.')

