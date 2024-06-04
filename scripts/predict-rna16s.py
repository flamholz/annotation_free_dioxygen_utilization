'''Script for predicting microbial physiology using the 16S embedding-based classifier.'''

from aerobot.rna16s import * 
import argparse
from aerobot.io import DATA_PATH, MODELS_PATH, RESULTS_PATH, save_results_dict
import json 
import numpy as no
import os
import time 
from sklearn.metrics import balanced_accuracy_score
import joblib

# TODO: Update this to support binary classification (?)
RNA16S_WEIGHTS_PATH = os.path.join(MODELS_PATH, 'rna16s_weights.pth')
RNA16S_ENCODER_PATH = os.path.join(MODELS_PATH, 'rna16s_encoder.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default=os.path.join(RESULTS_PATH, 'predict_rna16s.csv'), type=str)
    args = parser.parse_args()

    # Load the encoder used when training the model. 
    encoder = joblib.load(RNA16S_ENCODER_PATH)

    # Check to see if the embedding path exists. Only generate the embeddings if not. 
    emb_test_path = os.path.join(RNA16S_PATH, 'rna16s_emb_test.csv')

    if not os.path.exists(emb_test_path):
        # Load the sequences in to the sequence dataset. 
       rna16s_embed(os.path.join(RNA16S_PATH, 'rna16s_test.csv'), emb_test_path, encoder=encoder) 

    # Load the embedded testing data.  
    test_dataset = Rna16SEmbeddingDataset(emb_test_path, encoder)
    # Load the model weights. 
    model = Rna16SClassifier.load(RNA16S_WEIGHTS_PATH)

    labels, predictions, labels_decoded = model.predict(test_dataset, return_decoded_labels=True)

    results_df = pd.DataFrame(index=test_dataset.genome_ids) # Make sure to add the index back in!
    results_df['prediction'] = predictions # Ravel because Nonlinear output is a column vector. 
    if labels is not None:
        results_df['label'] = labels
        results_df['label_decoded'] = labels_decoded

        test_acc = balanced_accuracy_score(labels, predictions)
        print('Balanced accuracy on testing data:', np.round(test_acc, 2))
    
    print(f'Writing results to {args.output_path}.')
    results_df.to_csv(args.output_path)




