# Shell script for running multiple contig-based predictions on HPC.

import subprocess 
from aerobot.contigs import KMER_FEATURE_TYPES
from aerobot.io import SCRIPTS_PATH
import os

SCRIPT = os.path.join(SCRIPTS_PATH, 'predict-contigs.py')
# Select a few different genome IDs... perhaps five from each category. 
genome_ids = ['GCA_000093085', 'GCF_000875755', 'GCF_001295365'] # Facultative
genome_ids += ['GCF_000237085', 'GCF_016028255', 'GCF_000016385'] # Anaerobe
genome_ids += ['GCF_003431975', 'GCF_000022525', 'GCF_000973625'] # Aerobe 

genome_ids = ' '.join(genome_ids) # Format as a string to be passed into the script. 

for feature_type in KMER_FEATURE_TYPES:
    
    cmd = f'python {SCRIPT} {genome_ids} --feature-type {feature_type}'
    subprocess.rum(f'sbatch --time 24:00:00 --mem 64GB --wrap {cmd}', shell=True, check=True)