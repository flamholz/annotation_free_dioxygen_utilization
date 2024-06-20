import subprocess 
from aerobot.utils import SCRIPTS_PATH, FEATURE_TYPES, DATA_PATH, RESULTS_PATH, MODELS_PATH, CONTIGS_PATH
import os
from aerobot.models import NonlinearClassifier
from aerobot.dataset import order_features
import pandas as pd

PREDICT = os.path.join(SCRIPTS_PATH, 'predict.py')
TRAIN = os.path.join(SCRIPTS_PATH, 'train.py')

# Training the models --------------------------------------------------------------------------------------------------------------------------
# for feature_type in FEATURE_TYPES + ['embedding_rna16s']:
#     subprocess.run(f'python {script} logistic {feature_type} --n-classes 2', shell=True, check=True)
#     subprocess.run(f'python {script} logistic {feature_type}', shell=True, check=True)
#     subprocess.run(f'python {script} nonlinear {feature_type}', shell=True, check=True)

# Running trained models on Earth Microbiome Project and Black Sea data -------------------------------------------------------------------------
print('\nRunning trained models on Earth Microbiome Project and Black Sea data.\n')
for source in ['earth_microbiome', 'black_sea']:
    print(f'Predicting oxygen utilization from {source} MAGs.')
    aa_3mers = order_features(pd.read_csv(os.path.join(DATA_PATH, source, 'aa_3mer.csv'), index_col=0), 'aa_3mer')
    print('Number of MAGs:', len(aa_3mers))
    for model_class in ['nonlinear', 'logistic']:
        model = NonlinearClassifier.load(os.path.join(MODELS_PATH, f'{model_class}_aa_3mer_ternary.joblib'))

        predictions = pd.DataFrame(index=aa_3mers.index)
        predictions['prediction'] = model.predict(aa_3mers.values)
        prediction.to_csv(os.path.join(RESULTS_PATH, f'predict_{source}_{model_class}_aa_3mer_ternary.csv'))

# Running trained models on synthetic contigs----------------------------------------------------------------------------------------------------
# NOTE: This needs to be run on HPC, as contig dataset is too large. 

# script = os.path.join(SCRIPTS_PATH, 'predict.py')
# for feature_type in ['aa_1mer', 'aa_2mer', 'aa_3mer']:
#     model_path = os.path.join(MODELS_PATH, f'nonlinear_{feature_type}_ternary.joblib')
#     output_path = os.path.join(RESULTS_PATH, f'predict_contigs_nonlinear_{feature_type}_ternary.csv')
#     input_path = os.path.join(CONTIGS_PATH, 'datasets.h5')
    
#     subprocess.run(f'python {script} --m {model_path} -f {feature_type} -i {input_path} -o {output_path}', shell=True, check=True)

# Phylogenetic cross-validation ----------------------------------------------------------------------------------------------------------------
# script = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')
# for feature_type in FEATURE_TYPES:
#     cmd = f'python {script} nonlinear --feature-type {feature_type}'
#     subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
