import torch 
import numpy as np
from genslm import GenSLM, SequenceDataset
import pandas as pd
from typing import List, Tuple
from Bio import SeqIO
from aerobot.utils import ROOT_PATH
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_genslm():
    # First argument is model ID corresponding to a pre-trained model. (e.g., genslm_25M_patric)
    # model_cache_dir is a directory where model weights have been downloaded to
    model = GenSLM('genslm_25M_patric', model_cache_dir=os.path.join(ROOT_PATH, 'features')) 
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # print(f'rna16s_load_genslm: Loaded genslm model with {total_params} parameters.')
    
    # This freezes the weights of the base GenSLM model.  
    for param in model.parameters():
        param.requires_grad = False 
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 0, 'rna16s_load_genslm: There should be no trainable parameters in the loaded pre-trained model.'

    return model


def load_fasta(path) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    for record in SeqIO.parse(path, 'fasta'):
        ids.append(str(record.id))
        seqs.append(str(record.seq))
    return ids, seqs


class SequenceDataset(SequenceDataset):
    '''A Dataset object for working with 16S sequences and their corresponding metabolic labels.'''
    def __init__(self, ids:List[str], seqs:List[str]):

        # Need to grab some attributes from the GenSLM model. 
        gslm = load_genslm()
        seqs = [seq.upper() for seq in seqs]
        self.ids = ids
        super().__init__(seqs, gslm.seq_length, gslm.tokenizer) 


    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['id'] = self.ids[idx]
        return item


def from_fasta(path:str):

    genslm = load_genslm() # Load the pre-trained model. 
    dataset = SequenceDataset(*load_fasta(path))

    # Process the sequences in batches in order to reduce computational cost. 
    dataloader = DataLoader(dataset, batch_size=1)  # Initialize a DataLoader with the training Dataset. 

    ids, embeddings = [], []
    for batch in tqdm(dataloader, desc='rna_16s.from_fasta: Embedding sequences...'):
        ids.append(batch['id'])
        # Pass the inputs into the underlying GenSLM model to produce embeddings.  
        inputs = {'input_ids':batch['input_ids'], 'attention_mask':batch['attention_mask'], 'output_hidden_states':True}
        outputs = genslm(**inputs)
        # Extract the last set of hidden states and mean-pool over sequence length. 
        embeddings.append(outputs.hidden_states[-1].mean(dim=1).detach().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    df = pd.DataFrame(embeddings)
    df.index = np.array(ids).ravel() # Set the index to be the genome ID. 

    return df
