'''Script for training the 16S embedding-based classifier.'''

from aerobot.rna16s import * 
import argparse
from aerobot.io import DATA_PATH, MODELS_PATH, RESULTS_PATH, save_results_dict
import json 
import os 
import joblib

# Josh used the "best model weights" for evaluating the classifier.'''
RNA16S_WEIGHTS_PATH = os.path.join(MODELS_PATH, 'rna16s_weights.pth')
RNA16S_ENCODER_PATH = os.path.join(MODELS_PATH, 'rna16s_encoder.joblib')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', type=str, help='Name of the pre-trained model to use. Expected to be stored in the models directory.')
    parser.add_argument('--input-path', '-i', type=str, help='Path to the data on which to run the trained model. This should be in CSV format.')
    parser.add_argument('--feature-type', '-f', type=str, default='aa_3mer', choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--output-path', '-o', type=str, default=None, help='The location to which the predictions will be written.')

    encoder = joblib.load(RNA16S_ENCODER_PATH)
    model = Rna16SClassifier.load(MODEL_WEIGHTS_PATH)
    # Load the training, testing, and validation datasets. 
    datasets, encoder = rna16s_load_datasets()

    # Instantiate a classifier. 
    model = Rna16SClassifier()
    train_accs, val_accs, best_epoch = model.fit(datasets['training'], val_dataset=datasets['validation'], batch_size=args.batch_size, n_epochs=args.n_epochs)

    # Add some information about the model training to a dictionary. 
    results = dict()
    results['train_accs'] = train_accs
    results['val_accs'] = val_accs
    results['best_epoch'] = best_epoch
    results['batch_size'] = args.batch_size
    results['n_epochs'] = args.n_epochs

    # Save a summary of the training to a JSON file. )
    save_results_dict(results, os.path.join(RESULTS_PATH, 'train_rna16s_results.json'))

    # Save the encoder and the trained model weights to the MODELS_PATH directory. 
    print(f'Saving the model encoder to {RNA16S_ENCODER_PATH}')
    joblib.dump(encoder, RNA16S_ENCODER_PATH)
    print(f'Saving the best model weights to {RNA16S_WEIGHTS_PATH}')
    torch.save(model.best_model_weights, RNA16S_WEIGHTS_PATH)

