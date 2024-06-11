# Shell script for running multiple contig-based predictions on HPC.

import subprocess 
from aerobot.contigs import KMER_FEATURE_TYPES
from aerobot.io import SCRIPTS_PATH
import os

SCRIPT = os.path.join(SCRIPTS_PATH, 'phylo-cv.py')

for feature_type in KMER_FEATURE_TYPES:
    
    cmd = f'python {SCRIPT} --feature-type {feature_type}'
    subprocess.run(f'sbatch -J {feature_type} --time 24:00:00 --output={feature_type}.out --mem 64GB --wrap "{cmd}"', shell=True, check=True)
