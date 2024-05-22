import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from aerobot.io import load_fasta
from Bio import pairwise2
from Bio.SeqIO import FastaIO
import pandas as pd
# In paper http://onlinelibrary.wiley.com/doi/10.1111/1462-2920.13023/abstract
# Primers are in http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0116106#sec004
from copy import copy, deepcopy
import argparase

# Values for pairwise local alignment: 2 points for matches, no points deducted for non-identical characters, 
# -10 are deducted for opening a gap, and -1 points are deducted for extending a gap.
# TODO: Figure out why these values were selected.  
MATCH_SCORE = 1
MISMATCH_PENALTY = 0
GAP_START_PENALTY = -10
GAP_EXTENSION_PENALTY = -1

# What on Earth is this doing??
def get_amplicon(seq, forward_primer:str=None, reverse_primer:str=None):
    '''*I think* this function extracts the portion of the sequence which is from the PCR-amplified genome (i.e. the part of the
    sequence which is sandwiched between the two primers).'''
    # Align the sequence with the forward primer. 
    alignments = pairwise2.align.localms(forward_primer, seq, MATCH_SCORE, MISMATCH_PENALTY, GAP_START_PENALTY, GAP_EXTENSION_PENALTY)
    print(alignments)
    start = 0 if (len(alignments) == 0) else alignments[0][0].find(forward_primer) + len(forward_primer) + 1

    alignments = pairwise2.align.localms(reverse_primer, seq, MATCH_SCORE, MISMATCH_PENALTY, GAP_START_PENALTY, GAP_EXTENSION_PENALTY)
    stop = 0 if (len(alignments) == 0) else alignments[0][0].find(reverse_primer) - len(reverse_primer)

    return start, stop # NOTE: Not sure if start and stop are the appropriate names here. 


def fix_primers(forward_primer:str, reverse_primer:str):
    '''Convert forward and reverse primers into a format which works with the rest of the code. Specifically, convert all 
    upper-case nucleotides to lower-case, and take the reverse complement of the recerse primer.'''
    forward_primer = forward_primer.lower()
    # Get the reverse complement of the reverse primer. 
    # reverse_primer = str(Seq(reverse_primer).complement()).lower()[::-1]
    reverse_primer = str(Seq(reverse_primer).reverse_complement())
    return forward_primer, reverse_primer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--email', help='Email address for accessing NCBI utilities through Entrez.')

    Entrez.email = 'prichter@caltech.edu'  # Always provide your email address when using NCBI E-utilities

    metadata_df = pd.read_csv('Mark_Westoby_Organism_Metadata_Export_02152018.txt', sep='\t')

    # Get the 16S sequence accessions from the metadata file.  
    ncbi_gene_ids = metadata_df.GENBANK_16S_ID[~metadata_df.GENBANK_16S_ID.str.match('(null)')]
    ncbi_gene_ids = ncbi_gene_ids.unique().tolist()

    ncbi_get_seqs



print('downloading sequences')

sequence_records = get_sequences_by_accession_list(accessions)

print('writing full length seqs to file')
with open(seqFile_fullLength, "w") as output_handle:
    SeqIO.write(sequence_records, output_handle, "fasta")


# Extract V3-V4 region
print("Extracting V3-V4 region")
FORWARD_PRIMER = 'TATGGTAATTGTCTCCTACGGRRSGCAGCAG'
REVERSE_PRIMER = 'AGTCAGTCAGCCGGACTACNVGGGTWTCTAAT'
forward,reverse = convert_primers(forward,reverse)


seq_dict = SeqIO.to_dict(SeqIO.parse(seqFile_fullLength,'fasta'))
seq_dict_vr = {}
for idx,record in seq_dict.items():
    seq = str(record.seq).replace('-','').lower()
    start,finish = get_amplicon(seq,forward,reverse)
    record.seq = Seq(seq[start:finish])
    seq_dict_vr[idx] = record

print("make sure sequences are long enough (above 350 bp for V3-V4)")
seq_dict_vr = {x:y for x,y in seq_dict_vr.items() if len(y.seq)>350}

print("write to file")
# Save the V3-V4 region sequences to a file in FASTA format
with open(seqFile_V34, "w") as output_handle:
    SeqIO.write(seq_dict_vr.values(), output_handle, "fasta")



# split data for train test
    
md = metadata[["GENBANK_16S_ID","OXYGEN_REQUIREMENT"]]
md = md[~md["GENBANK_16S_ID"].apply(lambda x: x == "(null)")]
md = md[~md["OXYGEN_REQUIREMENT"].apply(lambda x: x == "(null)")]
md.columns = ["GeneBank","Oxygen"]

cmap = {"Anaerobe":"Anaerobe","Aerobe":"Aerobe","Facultative":"Facultative","Obligate anaerobe":"Anaerobe","Obligate aerobe":"Aerobe",'Facultative anaerobe':"Facultative"}
md = md[md.Oxygen.isin(list(cmap))]
md["Oxygen_label"] = md.Oxygen.apply(lambda x: cmap[x])

seqs = fasta_to_dataframe(seqFile_V34)
seqs["accession"] = [x.split(".")[0] for x in seqs.index.tolist()]

# de-duplicate
reps = seqs.set_index("accession").join(md.set_index("GeneBank")).dropna()
reps = reps[["sequence","Oxygen_label"]].drop_duplicates()

# shuffle before split
reps_shuffled = reps.sample(len(reps))

reps_train = reps_shuffled.iloc[0:800]
reps_test = reps_shuffled.iloc[801:900]
reps_valid = reps_shuffled.iloc[901:]


reps_test.to_csv(testFile)
reps_train.to_csv(trainingFile)
reps_valid.to_csv(validationFile)

print("Done!")
