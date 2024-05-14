#!/bin/bash


#SBATCH --time=99:00:00   # walltime
#SBATCH --mem=100GB

# For some reason the logistic models didn't seem to be converging on HPC?
# python ~/aerobot/scripts/train.py 'logistic' 
# python ~/aerobot/scripts/train.py 'nonlinear'

python ~/aerobot/scripts/phylo-cv.py 'logistic' 
python ~/aerobot/scripts/phylo-cv.py 'nonlinear' 
python ~/aerobot/scripts/phylo-cv.py 'meanrel' 

