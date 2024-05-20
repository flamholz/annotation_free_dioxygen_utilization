import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.SeqIO import FastaIO
import pandas as pd
# In paper http://onlinelibrary.wiley.com/doi/10.1111/1462-2920.13023/abstract
# Primers are in http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0116106#sec004
from copy import copy, deepcopy

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


if __name__ == '__main__':


    Entrez.email = "goldford.joshua@gmail.com"  # Always provide your email address when using NCBI E-utilities

    seqFile_fullLength = "data_16s/16S_sequences.fasta"
    seqFile_V34 = "data_16s/16S_sequences_V34.fasta"
    trainingFile = "data_16s/seqs.train.csv"
    testFile = "data_16s/seqs.test.csv"
    validationFile = "data_16s/seqs.valid.csv"


def convert_primers(forward,reverse):
    forward = forward.lower()
    reverse = str(Seq(reverse).complement()).lower()
    reverse = reverse[::-1].lower() 
    return forward,reverse

def get_sequences_by_accession_list(accession_list):
    records = []
    for accession in accession_list:
        try:
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()
            records.append(record)
        except Exception as e:
            print(f"Error fetching accession {accession}: {e}")
    return records

def fasta_to_dataframe(fasta_file):
    ids = []
    sequences = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))

    return pd.DataFrame({"sequence": sequences}, index=ids)

print('pulling accessions')

metadata = pd.read_csv("../metals_metabolism/data/taxa/Mark_Westoby_Organism_Metadata_Export_02152018.txt",sep='\t')

# Download 16S sequences for GTDB genomes
ncbi_gene_ids = metadata[~metadata.GENBANK_16S_ID.apply(lambda x: x == "(null)")]
accessions = ncbi_gene_ids.GENBANK_16S_ID.unique().tolist()

print('downloading sequences')

sequence_records = get_sequences_by_accession_list(accessions)

print('writing full length seqs to file')
with open(seqFile_fullLength, "w") as output_handle:
    SeqIO.write(sequence_records, output_handle, "fasta")


# Extract V3-V4 region
print("Extracting V3-V4 region")
forward = 'TATGGTAATTGTCTCCTACGGRRSGCAGCAG'
reverse = 'AGTCAGTCAGCCGGACTACNVGGGTWTCTAAT'
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
