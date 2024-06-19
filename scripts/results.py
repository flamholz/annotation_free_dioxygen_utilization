import subprocess 
from aerobot.utils import SCRIPTS_PATH, FEATURE_TYPES, DATA_PATH, RESULTS_PATH, MODELS_PATH, CONTIGS_PATH
import os

# Training the models --------------------------------------------------------------------------------------------------------------------------
# script = os.path.join(SCRIPTS_PATH, 'train.py')
# for feature_type in [f for f in FEATURE_TYPES if f != 'embedding_rna16s']:
#     subprocess.run(f'python {script} logistic {feature_type} --n-classes 2', shell=True, check=True)
#     subprocess.run(f'python {script} logistic {feature_type}', shell=True, check=True)
#     subprocess.run(f'python {script} nonlinear {feature_type}', shell=True, check=True)

# Running models on Earth Microbiome Project and Black Sea data --------------------------------------------------------------------------------
# script = os.path.join(SCRIPTS_PATH, 'predict.py')
# for source in ['earth_microbiome', 'black_sea']:
#     for model_class in ['nonlinear', 'logistic']:
#         input_path =  os.path.join(DATA_PATH, source, 'aa_3mer.csv')
#         output_path = os.path.join(RESULTS_PATH, f'{source}_{model_class}_aa_3mer_ternary.csv')
#         subprocess.run(f'python {script} -m {model_class}_aa_3mer_ternary.joblib -f aa_3mer -i {input_path} -o {output_path}', shell=True, check=True)

# Predicting metabolism from contigs -----------------------------------------------------------------------------------------------------------
script = os.path.join(SCRIPTS_PATH, 'predict.py')
for feature_type in ['aa_1mer', 'aa_2mer', 'aa_3mer']:
    model_path = os.path.join(MODELS_PATH, f'nonlinear_{feature_type}_ternary.joblib')
    output_path = os.path.join(RESULTS_PATH, f'predict_contigs_nonlinear_{feature_type}_ternary.csv')
    input_path = os.path.join(CONTIGS_PATH, 'datasets.h5')
    
    subprocess.run(f'python {script} --m {model_path} -f {feature_type} -i {input_path} -o {output_path}', shell=True, check=True)

# Phylogenetic cross-validation ----------------------------------------------------------------------------------------------------------------
# script = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')
# for feature_type in FEATURE_TYPES:
#     cmd = f'python {script} nonlinear --feature-type {feature_type}'
#     subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
