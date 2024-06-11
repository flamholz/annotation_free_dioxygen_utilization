import subprocess 
from aerobot.io import SCRIPTS_PATH, FEATURE_TYPES
import os

SCRIPT = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')

for feature_type in FEATURE_TYPES:
    
    cmd = f'python {SCRIPT} nonlinear --feature-type {feature_type}'
    subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
