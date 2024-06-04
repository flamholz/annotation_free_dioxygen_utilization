'''Code for the classifier based on GenSLM embeddings of 16S ribosome sequences.'''
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
from Bio import SeqIO
from typing import Tuple, Dict, List
# I might need to install gslm, here: https://github.com/ramanathanlab/genslm/blob/main/setup.cfg 
# Needed to install a Rust compiler before installing this. with this command: curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
# After installing Rust, I also needed to run apt install pkg-config
# Still didn't work, so I tried installing libssl-dev using apt. 
# Still didn't work, so I tried downgrading Python to 3.7. That finally worked. Not sure why, something with bugs when trying to install h5py and tokenizers. 
# Unfortunately, when I did this, I need to reinstall a ton of stuff. 
from genslm import GenSLM, SequenceDataset
import pandas as pd
from aerobot.io import DATA_PATH
import sklearn.model_selection
import pandas as pd
import pickle
import sklearn
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from time import perf_counter 

RNA16S_PATH = os.path.join(DATA_PATH, '16s')
RNA16S_TRAIN_PATH = os.path.join(RNA16S_PATH, 'rna16s_train.csv')
RNA16S_TEST_PATH = os.path.join(RNA16S_PATH, 'rna16s_test.csv')
RNA16S_VAL_PATH = os.path.join(RNA16S_PATH, 'rna16s_val.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Probably want to make sure everything works for unlabeled data. 

def rna16s_load_genslm():
    # First argument is model ID corresponding to a pre-trained model. (e.g., genslm_25M_patric)
    # model_cache_dir is a directory where model weights have been downloaded to
    model = GenSLM('genslm_25M_patric', model_cache_dir=RNA16S_PATH)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # print(f'rna16s_load_genslm: Loaded genslm model with {total_params} parameters.')
    
    # This freezes the weights of the base GenSLM model.  
    for param in model.parameters():
        param.requires_grad = False 
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 0, 'rna16s_load_genslm: There should be no trainable parameters in the loaded pre-trained model.'

    return model 


def rna16s_get_encoder() -> LabelEncoder:
    '''Get a fitted LabelEncoder object, which has been fit on the training labels.'''

    train_labels = pd.read_csv(RNA16S_TRAIN_PATH, usecols=['label'])
    train_labels = train_labels.label.values.tolist()
    
    encoder = LabelEncoder() # Instantiate a LabelEncoder. 
    encoder.fit(train_labels)
    return encoder


class Rna16SSequenceDataset(SequenceDataset):
    '''A Dataset object for working with 16S sequences and their corresponding metabolic labels.'''
    def __init__(self, path:str, encoder:LabelEncoder):

        # Need to grab some attributes from the GenSLM model. 
        gslm = rna16s_load_genslm()
        # Expect this DataFrame to have genome ID as index, a label column, and the raw sequences. 
        df = pd.read_csv(path, index_col=0) 
        seqs = [seq.upper() for seq in df['seq']]
        super().__init__(seqs, gslm.seq_length, gslm.tokenizer)

        self.labels = encoder.transform(df['label'].values)
        self.labels_decoded = df['label'].values # Also store the original string labels. 
        self.genome_ids = df.index.values

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['label'] = self.labels[idx]
        item['label_decoded'] = self.labels_decoded[idx]
        item['genome_id'] = self.genome_ids[idx]
        return item


class Rna16SEmbeddingDataset(Dataset):
    '''A Dataset object for working with 16S sequences and their corresponding metabolic labels.'''
    def __init__(self, path:str, encoder:LabelEncoder):

        super(Rna16SEmbeddingDataset, self).__init__()
        # Expect this DataFrame to have genome ID as index, a label column, and the stored embeddings. 
        df = pd.read_csv(path, index_col=0) 
        
        # If labels are present in the file, load them into the Dataset. 
        self.labels, self.labels_decoded = None, None
        if 'label' in df.columns:
            self.labels = encoder.transform(df['label'].values)
            self.labels_decoded = df['label'].values # Also store the original string labels. 

        self.embeddings = df.drop(columns=['label']).values
        self.genome_ids = df.index.values

        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = dict()
        if self.labels is not None:
            item['label'] = self.labels[idx]
            item['label_decoded'] = self.labels_decoded[idx]
        item['embedding'] = torch.FloatTensor(self.embeddings[idx])
        item['genome_id'] = self.genome_ids[idx]
        return item


def rna16s_embed(seq_path:str, emb_path:str, batch_size:int=1, encoder:LabelEncoder=None):

    genslm = rna16s_load_genslm() # Load the pre-trained model. 

    dataset = Rna16SSequenceDataset(seq_path, encoder)
    # Process the sequences in batches in order to reduce computational cost. 
    dataloader = DataLoader(dataset, batch_size=batch_size)  # Initialize a DataLoader with the training Dataset. 

    labels, genome_ids, embeddings = [], [], []
    for batch in tqdm(dataloader, desc='rna16s_embed: Embedding sequences...'):
        genome_ids.append(batch['genome_id'])
        labels.append(batch['label_decoded']) # Grab the plain labels (not one-hot encoded).

        # Pass the inputs into the underlying GenSLM model to produce embeddings.  
        inputs = {'input_ids':batch['input_ids'], 'attention_mask':batch['attention_mask'], 'output_hidden_states':True}
        outputs = genslm(**inputs)
        # Extract the last set of hidden states and mean-pool over sequence length. 
        embeddings.append(outputs.hidden_states[-1].mean(dim=1).detach().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    assert embeddings.shape[1] == 512, 'rna16s_embed: Embedding dimension should be 512.'

    df = pd.DataFrame(embeddings)
    df.index = np.array(genome_ids).ravel() # Set the index to be the genome ID. 
    if len(labels) > 0:
        df['label'] = np.array(labels).ravel()

    print(f'rna16s_embed: Writing embeddings to {emb_path}')
    df.to_csv(emb_path)


class Rna16SClassifier(torch.nn.Module):
    def __init__(self, n_classes:int=3, hidden_dim:int=512):
        '''Initialize a model for classifying genslm-generated 16S embeddings.'''
        super(Rna16SClassifier, self).__init__()

        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, n_classes))
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr  = 0.01
        # Change optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, embedding:torch.FloatTensor=None, **kwargs):
        '''A forward pass of the Rna16SClassifier.'''
        return self.classifier(embedding)

    def predict(self, dataset:Rna16SEmbeddingDataset, return_decoded_labels:bool=False) -> Tuple[List[int], List[float]]:

        self.eval() # Put the model in evaluation mode so that nothing weird happens with the weights. 
        
        labels, labels_decoded = None, None
        predictions = []
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 
        assert len(dataloader) == 1, 'Rna16SClassifier: The DataLoader should only have one batch when batch_size=len(Dataset).'
        
        for batch in dataloader: # Should only be one batch. 
            if 'label' in batch:
                labels = batch.pop('label').tolist()
                labels_decoded = batch.pop('label_decoded')
            outputs = self(**batch)
            predictions = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy().tolist()
            # predictions.append(torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy())

        # Order tuple output to match order of arguments for balanced_accuracy_score function. 
        # TRUE LABELS GO FIRST. I really need to stop switching them. 
        if (not return_decoded_labels):
            return labels, predictions
        else:
            return labels, predictions, labels_decoded

    def fit(self, train_dataset:Rna16SEmbeddingDataset, val_dataset:Rna16SEmbeddingDataset=None, batch_size:int=16, n_epochs:int=200):
        self.train() # Put the model in training mode. 

        dataloader = DataLoader(train_dataset, batch_size=batch_size)  # Initialize a DataLoader with the training Dataset. 
        val_accs, train_accs = [], []
        best_val_acc = 0 # For storing the best validation accuracy encountered. 
        best_model_weights = None
        best_epoch = 0
        # for epoch in tqdm(range(n_epochs), desc='Rns16SClassifier.fit'):
        for epoch in tqdm(range(n_epochs), desc=f'Rna16SClassifier.fit'):
            for batch in dataloader:
                batch = {k:batch[k].to(device) for k in ['embedding', 'label']} 
                self.optimizer.zero_grad()
                outputs = self(**batch)
                loss = self.loss_func(outputs, batch['label'])
                loss.backward()
                self.optimizer.step()

            train_accs.append(sklearn.metrics.balanced_accuracy_score(*self.predict(train_dataset)))

            val_acc = sklearn.metrics.balanced_accuracy_score(*self.predict(val_dataset))
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_model_weights = self.state_dict().copy() # .detach().clone()
        
        print(f'Rna16SClassifier: Best validation accuracy {best_val_acc} achieved at epoch {best_epoch + 1}.')
        self.best_model_weights = best_model_weights # Store the best model weights.
        return train_accs, val_accs, best_epoch # Return the computed accuracies and the best epoch. 

    @classmethod
    def load(cls, path:str):
        instance = cls()
        instance.to(device)
        # Load the trained model weights located at the path. 
        instance.load_state_dict(torch.load(path, device))

        return instance



    

