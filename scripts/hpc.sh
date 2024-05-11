#!/bin/bash


#SBATCH --time=99:00:00   # walltime
#SBATCH --mem=100GB
python ~/aerobot/scripts/train.py 'logistic' 


python ~/aerobot/scripts/train.py 'nonlinear' 


# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_3mer_from_contigs.csv -k 3 
# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_2mer_from_contigs.csv -k 2 
# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_1mer_from_contigs.csv -k 1

# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_3mer_from_contigs.csv -k 3 
# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_2mer_from_contigs.csv -k 2 
# python ~/aerobot/scripts/count-aa-kmers.py -i ~/aerobot/results/black_sea/bs_contigs.faa -o ~/aerobot/results/black_sea/bs_aa_1mer_from_contigs.csv -k 1

# # Run the nonlinear models on the data. 
# python ~/aerobot/scripts/run-pretrained.py -m ~/aerobot/models/aa_3mer_nonlinear_model.joblib -i ~/aerobot/results/black_sea/bs_aa_3mer_from_contigs.csv -o ~/aerobot/results/black_sea/bs_predictions_nonlinear_aa_3mer_from_contigs.csv -f 'aa_3mer'
# python ~/aerobot/scripts/run-pretrained.py -m ~/aerobot/models/aa_2mer_nonlinear_model.joblib -i ~/aerobot/results/black_sea/bs_aa_2mer_from_contigs.csv -o ~/aerobot/results/black_sea/bs_predictions_nonlinear_aa_2mer_from_contigs.csv -f 'aa_2mer'
# python ~/aerobot/scripts/run-pretrained.py -m ~/aerobot/models/aa_1mer_nonlinear_model.joblib -i ~/aerobot/results/black_sea/bs_aa_1mer_from_contigs.csv -o ~/aerobot/results/black_sea/bs_predictions_nonlinear_aa_1mer_from_contigs.csv -f 'aa_1mer'
