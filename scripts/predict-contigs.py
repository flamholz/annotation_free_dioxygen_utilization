'''Script for applying trained models to k-mer-based feature types generated from contigs.'''
from typing import List
import argparse
from aerobot.contigs import * 
from aerobot.ncbi import ncbi_download_genomes
from aerobot.models import GeneralClassifier
import os
from aerobot.io import MODELS_PATH, DATA_PATH
import warnings

warnings.simplefilter('ignore')

GENOMES_PATH = os.path.join(DATA_PATH, 'contigs', 'genomes')
DOWNLOADED_GENOME_IDS = [filename.replace('.fasta', '') for filename in os.listdir(GENOMES_PATH)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome-ids', nargs='+', default=DOWNLOADED_GENOME_IDS, help='A list of complete genome IDs for which to generate contig-based predictions')
    parser.add_argument('--feature-type', '-f', default='aa_1mer', choices=KMER_FEATURE_TYPES, help='The feature types to extract from the generated contigs.')
    parser.add_argument('--model-class', default='nonlinear', type=str, choices=['nonlinear', 'logistic'], help='The model class on which to generate the predictions.')
    parser.add_argument('--binary', type=bool, default=0)

    args = parser.parse_args()

    dir_path = contigs_make_feature_type_directory(args.feature_type)

    # Make sure all genome IDs are saved to the GENOMES_PATH directory. 
    genome_ids = ncbi_download_genomes(args.genome_ids)

    task = 'binary' if args.binary else 'ternary'
    # Load the model to use for predicting the data. 
    model_path = os.path.join(MODELS_PATH, f'{args.model_class}_{args.feature_type}_{task}.joblib')
    assert os.path.exists(model_path), f'{args.model_class.capitalize()} model for {task} classification of {args.feature_type} features is not present in {MODELS_PATH}'
    model = GeneralClassifier.load(model_path)
    
    predictions_df = [] # Will store the predictions_df for every genome ID. 
    for genome_id in genome_ids:
        # If the feature type data have already been created, don't re-generate. 
        if not os.path.exists(os.path.join(dir_path, f'{genome_id}_{args.feature_type}.h5')):
             # Generate a list of reasonable contig sizes. None will result in a prediction for the entire genome being generated. 
            contig_sizes = list(range(1000, 11000, 1000)) + [20000, 30000, 40000] + [None]
            contigs_dfs = [contigs_split_genome(genome_id, contig_size=contig_size) for contig_size in contig_sizes]
            contigs_extract_features(contigs_dfs, genome_id=genome_id, feature_type=args.feature_type)

        predictions_df.append(contigs_predict(genome_id, model, feature_type=args.feature_type))
    
    predictions_df = pd.concat(predictions_df, axis=0) # Combine all predictions DataFrames into a single DataFrame.         
    # Write the predictions to a CSV file. Overwrite any existing file. 
    predictions_df = predictions_df.set_index('contig_size')
    predictions_df.to_csv(os.path.join(RESULTS_PATH, f'predict_contigs_{args.feature_type}.csv'))
    