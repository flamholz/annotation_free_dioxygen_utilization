'''Code for the classifier based on GenSLM embeddings of 16S ribosome sequences.'''
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
# I might need to install gslm, here: https://github.com/ramanathanlab/genslm/blob/main/setup.cfg 
from genslm import GenSLM, SequenceDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax
import pandas as pd
import pickle

def rna16s_fine_tune_model():
    pass 

def 