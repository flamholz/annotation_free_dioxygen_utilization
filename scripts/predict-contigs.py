'''Script for applying trained models to k-mer-based feature types generated from contigs.'''
from typing import List
import argparse
from aerobot.contigs import * 
from aerobot.ncbi import ncbi_download_genomes
from aerobot.models import GeneralClassifier
import os
from aerobot.io import MODELS_PATH
import warnings

warnings.simplefilter('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('genome-ids', nargs='+', help='A list of complete genome IDs for which to generate contig-based predictions')
    parser.add_argument('--feature-type', '-f', default='aa_1mer', choices=KMER_FEATURE_TYPES, help='The feature types to extract from the generated contigs.')
    parser.add_argument('--model-class', default='nonlinear', type=str, choices=['nonlinear', 'logistic'], help='The model class on which to generate the predictions.')
    parser.add_argument('--binary', type=bool, default=0)

    args = parser.parse_args()

    genome_ids = getattr(args, 'genome-ids')
    # Make sure all genome IDs are saved to the GENOMES_PATH directory. 
    genome_ids = ncbi_download_genomes(genome_ids, complete_genomes_only=True)

    task = 'binary' if args.binary else 'ternary'
    # Load the model to use for predicting the data. 
    model_path = os.path.join(MODELS_PATH, f'{args.model_class}_{args.feature_type}_{task}.joblib')
    assert os.path.exists(model_path), f'{args.model_class.capitalize()} model for {task} classification of {args.feature_type} features is not present in {MODELS_PATH}'
    model = GeneralClassifier.load(model_path)
    
    for genome_id in genome_ids:

        genome_size = contigs_get_genome_size(genome_id)

        contig_sizes = list(range(1000, 100000, 5000))
        contig_sizes += list(range(100000, genome_size - 10000, 50000)) # Generate a list of reasonable contig sizes. 
        contig_sizes = list(contig_sizes) + [None] # None will result in a prediction for the entire genome being generated. 
        
        contigs_dfs = [contigs_split_genome_v2(genome_id, contig_size=contig_size) for contig_size in contig_sizes]
        contigs_extract_features(contigs_dfs, genome_id=genome_id, feature_type=args.feature_type)

        predictions_df = contigs_predict(genome_id, model, feature_type=args.feature_type)
        
        # Write the predictions to a CSV file. Overwrite any existing file. 
        predictions_df = pd.DataFrame(predictions_df).set_index('contig_size')
        predictions_df.to_csv(os.path.join(RESULTS_PATH, f'predict_contigs_{genome_id}_{args.feature_type}.csv'))
    