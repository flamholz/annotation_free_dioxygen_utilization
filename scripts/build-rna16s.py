import pandas as pd
import numpy as np 
from Bio import pairwise2
from Bio import SeqIO
from Bio.Seq import Seq
from aerobot.utils import DATA_PATH, save_hdf, training_testing_validation_split
from aerobot.features import rna16s
import aerobot.entrez
import pandas as pd
import os
import tables
from typing import Dict 
from tqdm import tqdm
import warnings

# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


# NOTE: Primers are in http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0116106#sec004

# Values for pairwise local alignment: 2 points for matches, no points deducted for non-identical characters, 
# -10 are deducted for opening a gap, and -1 points are deducted for extending a gap.
# TODO: Figure out why these values were selected.  

MATCH_SCORE = 1
MISMATCH_PENALTY = 0
GAP_START_PENALTY = -10
GAP_EXTENSION_PENALTY = -1

FORWARD_PRIMER = 'TATGGTAATTGTCTCCTACGGRRSGCAGCAG'
REVERSE_PRIMER = 'AGTCAGTCAGCCGGACTACNVGGGTWTCTAAT'

RNA16S_FULL_LENGTH_PATH = os.path.join(DATA_PATH, 'rna16s_full_length.fasta') # Path where the full-length RNA sequences will be written.  
RNA16S_V3_V4_REGIONS_PATH = os.path.join(DATA_PATH, 'rna16s_v3_v4_regions.fasta') 


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


def load_fasta(path) -> pd.DataFrame:
    ids, seqs = [], []
    for record in SeqIO.parse(path, 'fasta'):
        ids.append(str(record.id))
        seqs.append(str(record.seq))
    df = pd.DataFrame()
    df['seqs'] = seqs 
    df['id'] = [id_.split('.')[0] for id_ in ids] 
    return df.set_index('id')


def remove_duplicates(seq_dict:Dict[str, str]):

    new_seq_dict = dict()
    for id_, record in seq_dict.items():
        new_id = id_.split('.')[0]
        # Make sure to modify the underlying record so the changes get written to the FASTA file.
        record.id = new_id 
        record.name = new_id
        new_seq_dict[new_id] = record 
    n_removed = len(seq_dict) - len(new_seq_dict)
    print(f'Removed {n_removed} duplicate entries from 16S dataset.')
    return new_seq_dict


if __name__ == '__main__':


    # Load in metadata, which contains the accessions for the 16S sequences.
    # Metadata is from paper http://onlinelibrary.wiley.com/doi/10.1111/1462-2920.13023/abstract
    metadata = pd.read_csv(os.path.join(DATA_PATH, 'rna16s_metadata.csv'), index_col=0)

    # Get the 16S sequence accessions from the metadata file.  
    ids = metadata.index.unique().tolist()

    if not os.path.exists(RNA16S_FULL_LENGTH_PATH):
        seqs = aerobot.entrez.download_rna16s_seqs(ids)
        
        print(f'Writing full-length 16S sequences to {RNA16S_FULL_LENGTH_PATH}')
        with open(RNA16S_FULL_LENGTH_PATH, 'w') as f:
            SeqIO.write(seqs, f, 'fasta')

    forward_primer, reverse_primer = get_primers()

    # Load in the dictionary of sequences, and extract the region between the primers. 
    seq_dict = SeqIO.to_dict(SeqIO.parse(RNA16S_FULL_LENGTH_PATH, 'fasta')) # I imagine the keys are IDs. 
    # Remove duplicates to prevent leakage by removing the '.1' or '.2' suffixes. 
    seq_dict = remove_duplicates(seq_dict)
    seq_dict.pop('AE017261') # This sequence is like a billion base pairs long for some reason. 
    seq_dict.pop('BX897699') # Ditto. 

    if not os.path.exists(RNA16S_V3_V4_REGIONS_PATH):
        v3_v4_seq_dict = dict()
        for id_, record in tqdm(seq_dict.items(), desc='Aligning primers to full-length sequences...'):
            seq = str(record.seq).replace('-','').lower()
            # Get the start and stop indices for the sub-sequence between the primers.
            start, stop = get_amplicon(seq, forward_primer=forward_primer, reverse_primer=reverse_primer)
            record.seq = Seq(seq[start:stop]) # Slice the full-length sequence. 
            v3_v4_seq_dict[id_] = record
        # Remove all sequences shorter than 350 nucleotides.  
        v3_v4_seq_dict = {i:s for i, s in v3_v4_seq_dict.items() if len(s.seq) > 350}
        
        # Save the V3-V4 region sequences to a file in FASTA format
        print(f'Writing V3-V4 regions to file {RNA16S_V3_V4_REGIONS_PATH}.')
        with open(RNA16S_V3_V4_REGIONS_PATH, 'w') as f:
            SeqIO.write(v3_v4_seq_dict.values(), f, 'fasta')

    rna16s_features = rna16s.from_fasta(RNA16S_V3_V4_REGIONS_PATH)

    dataset = {'embedding.rna16s':rna16s_features, 'metadata':metadata}
    training_dataset, testing_dataset, validation_dataset = training_testing_validation_split(dataset) 

    # Save each dataset to an HDF file.
    print('Saving 16S RNA training data to', os.path.join(DATA_PATH, 'rna16s_training_dataset.h5'))
    save_hdf(training_dataset, os.path.join(DATA_PATH, 'rna16s_training_dataset.h5'))
    print('Saving 16S RNA validation data to', os.path.join(DATA_PATH, 'rna16s_validation_dataset.h5'))
    save_hdf(validation_dataset, os.path.join(DATA_PATH, 'rna16s_validation_dataset.h5'))
    print('Saving 16S RNA testing data to', os.path.join(DATA_PATH, 'rna16s_testing_dataset.h5'))
    save_hdf(testing_dataset, os.path.join(DATA_PATH, 'rna16s_testing_dataset.h5'))



