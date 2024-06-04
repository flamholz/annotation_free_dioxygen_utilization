'''Script for training the 16S embedding-based classifier.'''

from aerobot.rna16s import * 
import argparse
from aerobot.io import DATA_PATH, MODELS_PATH, RESULTS_PATH, save_results_dict
import json 
import os 
import joblib

# TODO: Update this to support binary classification (?)
RNA16S_WEIGHTS_PATH = os.path.join(MODELS_PATH, 'rna16s_weights.pth')
RNA16S_ENCODER_PATH = os.path.join(MODELS_PATH, 'rna16s_encoder.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=16, type=int, help='The size of the batches to use for model training.')
    parser.add_argument('--n-epochs', default=200, type=int, help='The number of epochs to train the model for.')
    args = parser.parse_args()

    encoder = rna16s_get_encoder()

    # Check to see if the embedding path exists. Only generate the embeddings if not. 
    emb_val_path = os.path.join(RNA16S_PATH, 'rna16s_emb_val.csv')
    emb_train_path = os.path.join(RNA16S_PATH, 'rna16s_emb_train.csv')

    if (not os.path.exists(emb_val_path)) or (not os.path.exists(emb_train_path)):
        # Load the sequences in to the sequence dataset. 
       rna16s_embed(os.path.join(RNA16S_PATH, 'rna16s_val.csv'), emb_val_path, encoder=encoder) 
       rna16s_embed(os.path.join(RNA16S_PATH, 'rna16s_train.csv'), emb_train_path, encoder=encoder)

    # Load the training and validation datasets. 
    train_dataset = Rna16SEmbeddingDataset(emb_train_path, encoder)
    val_dataset = Rna16SEmbeddingDataset(emb_val_path, encoder)

    # Instantiate a classifier. 
    model = Rna16SClassifier()
    train_accs, val_accs, best_epoch = model.fit(train_dataset, val_dataset=val_dataset, batch_size=args.batch_size, n_epochs=args.n_epochs)

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

