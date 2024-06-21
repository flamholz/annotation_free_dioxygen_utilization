import subprocess 
from aerobot.utils import SCRIPTS_PATH, FEATURE_TYPES, DATA_PATH, RESULTS_PATH, MODELS_PATH, CONTIGS_PATH
import os
from aerobot.models import NonlinearClassifier
from aerobot.dataset import order_features, is_kmer_feature_type
import pandas as pd

PREDICT = os.path.join(SCRIPTS_PATH, 'predict.py')
TRAIN = os.path.join(SCRIPTS_PATH, 'train.py')

# Training the models --------------------------------------------------------------------------------------------------------------------------
# print('\nTraining the models...\n')
for feature_type in ['aa_1mer', 'aa_2mer', 'aa_3mer']:
    subprocess.run(f'python {TRAIN} logistic {feature_type} --n-classes 2', shell=True, check=True)
    subprocess.run(f'python {TRAIN} logistic {feature_type}', shell=True, check=True)
    subprocess.run(f'python {TRAIN} nonlinear {feature_type}', shell=True, check=True)

# Running trained models on Earth Microbiome Project and Black Sea data -------------------------------------------------------------------------
# NOTE: This needs to be run on HPC, as the Earth Microbiome Project dataset is too large. 
# print('\nRunning trained models on Earth Microbiome Project and Black Sea data...\n')
# for source in ['earth_microbiome', 'black_sea']:
#     print(f'Predicting oxygen utilization from {source} MAGs.')
#     aa_3mers = order_features(pd.read_csv(os.path.join(DATA_PATH, source, 'aa_3mer.csv'), index_col=0), 'aa_3mer')
#     print('Number of MAGs:', len(aa_3mers))
#     for model_class in ['nonlinear', 'logistic']:
#         model = NonlinearClassifier.load(os.path.join(MODELS_PATH, f'{model_class}_aa_3mer_ternary.joblib'))

#         predictions = pd.DataFrame(index=aa_3mers.index)
#         predictions['prediction'] = model.predict(aa_3mers.values)
#         prediction.to_csv(os.path.join(RESULTS_PATH, f'predict_{source}_{model_class}_aa_3mer_ternary.csv'))

# Running trained models on synthetic contigs----------------------------------------------------------------------------------------------------
print('\nRunning trained models on synthetic contigs...\n')

# for feature_type in ['nt_3mer', 'nt_4mer', 'nt_5mer']:
#     model_path = os.path.join(MODELS_PATH, f'nonlinear_{feature_type}_ternary.joblib')
#     output_path = os.path.join(RESULTS_PATH, f'predict_contigs_nonlinear_{feature_type}_ternary.csv')
#     input_path = os.path.join(CONTIGS_PATH, 'datasets.h5')
    
#     subprocess.run(f'python {PREDICT} --m {model_path} -f {feature_type} -i {input_path} -o {output_path}', shell=True, check=True)

# Phylogenetic cross-validation ----------------------------------------------------------------------------------------------------------------
# script = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')
# for feature_type in FEATURE_TYPES:
#     cmd = f'python {script} nonlinear --feature-type {feature_type}'
#     subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
