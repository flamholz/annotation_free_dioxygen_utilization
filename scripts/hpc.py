import subprocess 
from aerobot.io import SCRIPTS_PATH, FEATURE_TYPES
import os

SCRIPT = os.path.join(SCRIPTS_PATH, 'train.py')
# subprocess.run(f'python {SCRIPT} logistic --binary 1', shell=True, check=True)
# subprocess.run(f'python {SCRIPT} logistic', shell=True, check=True)
# subprocess.run(f'python {SCRIPT} nonlinear', shell=True, check=True)

SCRIPT = os.path.join(SCRIPTS_PATH, 'predict-contigs.py')
for feature_type in ['aa_1mer', 'aa_2mer', 'aa_3mer']:
    cmd = f'python {SCRIPT} nonlinear --feature-type {feature_type}'
    subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)

SCRIPT = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')
# for feature_type in FEATURE_TYPES:
for feature_type in [f for f in FEATURE_TYPES if ('aa_' in f) or ('cds_' in f) or ('nt_' in f)]:
    cmd = f'python {SCRIPT} nonlinear --feature-type {feature_type}'
    subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
