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
from aerobot.rna16s import RNA16S_TEST_PATH, RNA16S_TRAIN_PATH, RNA16S_VAL_PATH

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
    start = alignments[0][0].find(forward_primer) + len(forward_primer) + 1 if (len(alignments) > 0) else 0 

    alignments = pairwise2.align.localms(reverse_primer, seq, MATCH_SCORE, MISMATCH_PENALTY, GAP_START_PENALTY, GAP_EXTENSION_PENALTY)
    stop = alignments[0][0].find(reverse_primer) - len(reverse_primer) if (len(alignments) > 0) else 0

    return start, stop # NOTE: Not sure if start and stop are the appropriate names here. 


def get_primers(forward_primer:str=FORWARD_PRIMER, reverse_primer:str=REVERSE_PRIMER):
    '''Convert forward and reverse primers into a format which works with the rest of the code. Specifically, convert all 
    upper-case nucleotides to lower-case, and take the reverse complement of the recerse primer.'''
    # Put the primers in the correct form. 
    forward_primer = forward_primer.lower()
    # Get the reverse complement of the reverse primer. 
    # reverse_primer = str(Seq(reverse_primer).complement()).lower()[::-1]
    reverse_primer = str(Seq(reverse_primer).reverse_complement()).lower()
    return forward_primer, reverse_primer


def train_test_validation_split(fasta_df:pd.DataFrame):
    '''Divide RNA 16S sequences into training, testing, and validation datasets.'''
    # Shuffle before splitting.
    fasta_df = fasta_df.sample(len(fasta_df))

    fasta_df_train = fasta_df.iloc[0:800]
    fasta_df_test = fasta_df.iloc[801:900]
    fasta_df_val = fasta_df.iloc[901:]

    fasta_df_test.to_csv(RNA16S_TEST_PATH)
    fasta_df_train.to_csv(RNA16S_TRAIN_PATH)
    fasta_df_val.to_csv(RNA16S_VAL_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Load in metadata, which contains the accessions for the 16S sequences.
    metadata_df = pd.read_csv(os.path.join(RNA16S_PATH, 'Mark_Westoby_Organism_Metadata_Export_02152018.tsv'), sep='\t')

    # Get the 16S sequence accessions from the metadata file.  
    rna16s_ids = metadata_df.GENBANK_16S_ID[~metadata_df.GENBANK_16S_ID.str.contains('(null)', regex=False)]
    rna16s_ids = rna16s_ids.unique().tolist()

    if not os.path.exists(RNA16S_FULL_LENGTH_PATH):
        rna16s_seqs = ncbi_rna16s_get_seqs(rna16s_ids)
        
        print(f'Writing full length sequences to file {RNA16S_FULL_LENGTH_PATH}')
        with open(RNA16S_FULL_LENGTH_PATH, 'w') as f:
            SeqIO.write(rna16s_seqs, f, 'fasta')

    if not os.path.exists(RNA16S_V3_V4_REGIONS_PATH):
        forward_primer, reverse_primer = get_primers()

        # Load in the dictionary of sequences, and extract the region between the primers. 
        seq_dict = SeqIO.to_dict(SeqIO.parse(RNA16S_FULL_LENGTH_PATH, 'fasta')) # How does this get loaded?

        # i = 0
        v3_v4_seq_dict = dict()
        for idx, record in tqdm(seq_dict.items(), desc='Aligning primers to full-length sequences...'):
            seq = str(record.seq).replace('-','').lower()
            # Get the start and stop indices for the sub-sequence between the primers. 
            start, stop = get_amplicon(seq, forward_primer=forward_primer, reverse_primer=reverse_primer)
            record.seq = Seq(seq[start:stop]) # Slice the full-length sequence. 
            v3_v4_seq_dict[idx] = record
        # Remove all sequences shorter than 350 nucleotides.  
        v3_v4_seq_dict = {i:s for i, s in v3_v4_seq_dict.items() if len(s.seq) > 350}
        
        # Save the V3-V4 region sequences to a file in FASTA format
        print(f'Writing V3-V4 regions to file {RNA16S_V3_V4_REGIONS_PATH}.')
        with open(RNA16S_V3_V4_REGIONS_PATH, 'w') as f:
            SeqIO.write(v3_v4_seq_dict.values(), f, 'fasta')


    metadata_df = metadata_df[['GENBANK_16S_ID','OXYGEN_REQUIREMENT']] # Grab the relevant columns from the metadata DataFrame.
    metadata_df = metadata_df[~metadata_df['GENBANK_16S_ID'].str.contains('(null)', regex=False)]
    metadata_df = metadata_df[~metadata_df['GENBANK_16S_ID'].str.contains('(null)', regex=False)]
    metadata_df.columns = ['id', 'label']
    label_map = {'Anaerobe':'anaerobe', 'Aerobe':'aerobe', 'Facultative':'facultative', 'Obligate anaerobe':'anaerobe', 'Obligate aerobe':'aerobe','Facultative anaerobe':'facultative'}
    metadata_df = metadata_df[metadata_df.label.isin(label_map)] # Filter for organisms with an oxygen label is in the map. 
    metadata_df.label = metadata_df.label.rename(label_map)

    fasta_df = load_fasta(RNA16S_V3_V4_REGIONS_PATH)
    fasta_df['id'] = [s.split('.')[0] for s in fasta_df.header.tolist()]
    # Merge the sequences and metadata, and drop all duplicates. 
    fasta_df = fasta_df.set_index('id').join(metadata_df.set_index('id')).dropna()
    fasta_df = fasta_df[['seq', 'label']].drop_duplicates()

    train_test_validation_split(fasta_df)
