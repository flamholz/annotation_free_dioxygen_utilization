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
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection
import pandas as pd
import pickle
import sklearn
from tqdm import tqdm 

RNA16S_PATH = os.path.join(DATA_PATH, '16s')
RNA16S_TRAIN_PATH = os.path.join(RNA16S_PATH, 'rna16s_train.csv')
RNA16S_TEST_PATH = os.path.join(RNA16S_PATH, 'rna16s_test.csv')
RNA16S_VAL_PATH = os.path.join(RNA16S_PATH, 'rna16s_val.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Should I implement the thing where I store the best model weights in the other models? Seems like it might be a good idea. 

# NOTE: The file I am basing this on is called "fine-tune," but it seems as though all of the weights for the LLM are frozen?

def rna16s_load_genslm():
    # First argument is model ID corresponding to a pre-trained model. (e.g., genslm_25M_patric)
    # model_cache_dir is a directory where model weights have been downloaded to
    model = GenSLM('genslm_25M_patric', model_cache_dir=RNA16S_PATH)
    model.to(device)
    return model 


class Rna16SDataset(SequenceDataset):
    '''A Dataset object for working with 16S sequences and their corresponding metabolic labels.'''
    def __init__(self, seqs, labels, *args, **kwargs):
        # Need to grab some attributes from the GenSLM model. 
        # TODO: There is definitely a better way to do this... 
        gslm = rna16s_load_genslm()
        super().__init__(seqs, gslm.seq_length, gslm.tokenizer, *args, **kwargs)
        self.labels = labels

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['label'] = self.labels[idx]
        return item


class Rna16SClassifier(torch.nn.Module):
    def __init__(self, n_classes:int=3, hidden_dim:int=512):
        '''Initialize a 16S-based classifier.'''
        super(Rna16SClassifier, self).__init__()

        self.genslm = rna16s_load_genslm()
        # This freezes the weights of the base GenSLM model.  
        for param in self.genslm.parameters():
            param.requires_grad = False
        
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, n_classes))
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr  = 0.01
        # Change optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input_ids:torch.FloatTensor=None, attention_mask:torch.FloatTensor=None):

        # Pass the inputs into the underlying GenSLM model to produce embeddings.  
        kwargs = {'input_ids':input_ids, 'attention_mask':attention_mask, 'output_hidden_states':True}
        outputs = self.genslm(**kwargs)
        # Extract the last set of hidden states and mean-pool over sequence length. 
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        return self.classifier(embeddings)

    def predict(self, dataset:Rna16SDataset):

        self.eval() # Put the model in evaluation mode so that nothing weird happens with the weights. 
        
        labels = None
        dataloader = DataLoader(dataset, batch_size=len(dataset)) 
        assert len(dataloader) == 1, 'Rns16SClassifier: The DataLoader should only have one batch when batch_size=len(Dataset).'
        for batch in dataloader:
            if 'label' in batch:
                labels = batch.pop('label')
            outputs = self(**batch)
            predictions = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()

        # Order tuple output to match order of arguments for balanced_accuracy_score function. 
        # TRUE LABELS GO FIRST. I really need to stop switching them. 
        return labels.tolist(), predictions.tolist()


    def fit(self, train_dataset:Rna16SDataset, val_dataset:Rna16SDataset=None, batch_size:int=16, n_epochs:int=200):
        self.train() # Put the model in training mode. 

        dataloader = DataLoader(train_dataset, batch_size=batch_size)  # Initialize a DataLoader with the training Dataset. 
        val_accs, train_accs = [], []
        best_val_acc = 0 # For storing the best validation accuracy encountered. 
        best_model_weights = None
        best_epoch = 0
        # for epoch in tqdm(range(n_epochs), desc='Rns16SClassifier.fit'):
        for epoch in range(n_epochs):
            for batch in tqdm(dataloader, total=len(dataloader), desc=f'Rna16SClassifier.fit: Training classifier, epoch {epoch} of {n_epochs}.'):
                labels = batch.pop('label').to(device)
                batch = {k:v.to(device) for k, v in batch.items()} 

                self.optimizer.zero_grad()

                outputs = self(**batch)
                loss = self.loss_func(outputs, labels)
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
        
        genslm = rna16s_load_genslm() # Load the "base model."
        instance = cls(genslm) # Use the default hidden_dim and n_classes. 
        instance.to(device)
        # Load the trained model weights located at the path. 
        instance.load_state_dict(torch.load(path, device))

        return instance


def rna16s_load_datasets(n:int=None) -> Tuple[Rna16SDataset, Rna16SDataset]:
    # Load the training data and split it into training and validation datasets. 
    train_df = pd.read_csv(RNA16S_TRAIN_PATH)
    if n is not None:
        train_df = train_df.iloc[:n]
    # train_df, val_df = sklearn.model_selection.train_test_split(train_df, test_size=0.1, random_state=42)
    # Load the testing data from a seperate file. 
    # test_df = pd.read_csv(os.path.join(RNA16S_PATH, 'testing_data.csv'))
    val_df = pd.read_csv(RNA16S_VAL_PATH)

    # Map labels to integers. Fit the encoder using the training labels. 
    encoder = LabelEncoder()
    encoder.fit(train_df['label'].values)

    datasets = {}
    # for dataset_label, df in zip(['training', 'validation', 'testing'], [train_df, val_df, test_df]):
    for dataset_label, df in zip(['training', 'validation'], [train_df, val_df]):
        seqs = [seq.upper() for seq in df['seq']]
        labels = encoder.transform(df['label'].tolist()) # Convert the labels to integers using the fitted encoder.
        dataset = Rna16SDataset(seqs, labels)
        datasets[dataset_label] = dataset 

    return datasets, encoder


    

