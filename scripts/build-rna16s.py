import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from aerobot.io import load_fasta, DATA_PATH
from aerobot.rna16s import RNA16S_PATH
from aerobot.ncbi import ncbi_rna16s_get_seqs
from Bio import pairwise2
from Bio.SeqIO import FastaIO
import pandas as pd
# In paper http://onlinelibrary.wiley.com/doi/10.1111/1462-2920.13023/abstract
# Primers are in http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0116106#sec004
from copy import copy, deepcopy
import argparse
import os
from tqdm import tqdm 

# Values for pairwise local alignment: 2 points for matches, no points deducted for non-identical characters, 
# -10 are deducted for opening a gap, and -1 points are deducted for extending a gap.
# TODO: Figure out why these values were selected.  
MATCH_SCORE = 1
MISMATCH_PENALTY = 0
GAP_START_PENALTY = -10
GAP_EXTENSION_PENALTY = -1

FORWARD_PRIMER = 'TATGGTAATTGTCTCCTACGGRRSGCAGCAG'
REVERSE_PRIMER = 'AGTCAGTCAGCCGGACTACNVGGGTWTCTAAT'

RNA16S_FULL_LENGTH_PATH = os.path.join(RNA16S_PATH, 'rna16s_full_length.fasta') # Path where the full-length RNA sequences will be written.  
RNA16S_V3_V4_REGIONS_PATH = os.path.join(RNA16S_PATH, 'rna16s_v3_v4_regions.fasta')  

# TODO: What is the V3 V4 region? Why is it important, and why are we focusing on it?

# What on Earth is this doing??
def get_amplicon(seq, forward_primer:str=None, reverse_primer:str=None):
    '''*I think* this function extracts the portion of the sequence which is from the PCR-amplified genome (i.e. the part of the
    sequence which is sandwiched between the two primers).'''
    # Align the sequence with the forward primer. 
    alignments = pairwise2.align.localms(forward_primer, seq, MATCH_SCORE, MISMATCH_PENALTY, GAP_START_PENALTY, GAP_EXTENSION_PENALTY)
    start = 0 if (len(alignments) == 0) else alignments[0][0].find(forward_primer) + len(forward_primer) + 1

    alignments = pairwise2.align.localms(reverse_primer, seq, MATCH_SCORE, MISMATCH_PENALTY, GAP_START_PENALTY, GAP_EXTENSION_PENALTY)
    stop = 0 if (len(alignments) == 0) else alignments[0][0].find(reverse_primer) - len(reverse_primer)

    return start, stop # NOTE: Not sure if start and stop are the appropriate names here. 


def get_primers(forward_primer:str=FORWARD_PRIMER, reverse_primer:str=REVERSE_PRIMER):
    '''Convert forward and reverse primers into a format which works with the rest of the code. Specifically, convert all 
    upper-case nucleotides to lower-case, and take the reverse complement of the recerse primer.'''
    # Put the primers in the correct form. 
    forward_primer = forward_primer.lower()
    # Get the reverse complement of the reverse primer. 
    # reverse_primer = str(Seq(reverse_primer).complement()).lower()[::-1]
    reverse_primer = str(Seq(reverse_primer).reverse_complement())
    return forward_primer, reverse_primer


def train_test_split(metadata_df:pd.DataFrame):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Load in metadata, which contains the accessions for the 16S sequences.
    metadata_df = pd.read_csv(os.path.join(RNA16S_PATH, 'Mark_Westoby_Organism_Metadata_Export_02152018.tsv'), sep='\t')

    # Get the 16S sequence accessions from the metadata file.  
    rna16s_ids = metadata_df.GENBANK_16S_ID[~metadata_df.GENBANK_16S_ID.str.contains('(null)')]
    rna16s_ids = rna16s_ids.unique().tolist()

    if not os.path.exists(RNA16S_FULL_LENGTH_PATH):
        rna16s_seqs = ncbi_rna16s_get_seqs(rna16s_ids)
        
        print(f'Writing full length sequences to file {RNA16S_FULL_LENGTH_PATH}')
        with open(os.path.join(RNA16S_PATH, 'rna16s_full_length.fasta'), 'w') as f:
            SeqIO.write(rna16s_seqs, f, 'fasta')

    if not os.path.exists(RNA16S_V3_V4_REGIONS_PATH):
        forward_primer, reverse_primer = get_primers()
        # Load in the dictionary of sequences, and extract the region between the primers. 
        seq_dict = SeqIO.to_dict(SeqIO.parse(RNA16S_FULL_LENGTH_PATH, 'fasta')) # How does this get loaded?

        for idx, record in tqdm(seq_dict.items(), desc='Aligning primers to full-length sequences...'):
            seq = str(record.seq).replace('-','').lower()
            # Get the start and stop indices for the sub-sequence between the primers. 
            start, stop = get_amplicon(seq, forward_primer=forward_primer, reverse_primer=reverse_primer)
            record.seq = Seq(seq[start:stop]) # Slice the full-length sequence. 
            seq_dict[idx] = record
        # Make sure sequences are long enough (over 350 BP). 
        seq_dict = {i:s for i, s in seq_dict.items() if len(s.seq) > 350}
        
        # Save the V3-V4 region sequences to a file in FASTA format
        print(f'Writing V3-V4 regions to file {RNA16S_V3_V4_REGIONS_PATH}.')
        with open(RNA16S_V3_V4_REGIONS_PATH, 'w') as f:
            SeqIO.write(seq_dict.values(), f, 'fasta')


    
    metadata_df = metadata_df[['GENBANK_16S_ID','OXYGEN_REQUIREMENT']] # Grab the relevant columns from the metadata DataFrame.
    metadata_df = metadata_df[~metadata_df['GENBANK_16S_ID'].str.contains('(null)')]
    metadata_df = metadata_df[~metadata_df['GENBANK_16S_ID'].str.contains('(null)')]
    metadata_df.columns = ['genome_id', 'label']

    label_map = {'Anaerobe':'anaerobe', 'Aerobe':'aerobe', 'Facultative':'facultative', 'Obligate anaerobe':'anaerobe', 'Obligate aerobe':'aerobe','Facultative anaerobe':'facultative'}
    metadata_df = metadata_df[metadata_df.label.isin(label_map)] # Filter for organisms with an oxygen label is in the map. 
    metadata_df.label = metadata.label.rename(label_map)

    fasta_df = load_fasta(RNA16S_V3_V4_REGIONS_PATH)
    fasta_df['id'] = [s.split('.')[0] for s in fasta_df.index.tolist()]
    print(fasta_df)

# # de-duplicate
# reps = seqs.set_index("accession").join(md.set_index("GeneBank")).dropna()
# reps = reps[["sequence","Oxygen_label"]].drop_duplicates()

# # shuffle before split
# reps_shuffled = reps.sample(len(reps))

# reps_train = reps_shuffled.iloc[0:800]
# reps_test = reps_shuffled.iloc[801:900]
# reps_valid = reps_shuffled.iloc[901:]


# reps_test.to_csv(testFile)
# reps_train.to_csv(trainingFile)
# reps_valid.to_csv(validationFile)

# print("Done!")
