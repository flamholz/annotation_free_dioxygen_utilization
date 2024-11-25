from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import joblib
import torch
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from tqdm import tqdm
from typing import Tuple, NoReturn, List, Dict
from sklearn.linear_model import LogisticRegression
import pandas as pd
import copy 
from sklearn.preprocessing import OneHotEncoder

# Use a GPU if one is available. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Figure out why model training doesn't seem to be reproducible.

def shuffle(X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Shuffle the input arrays without modifying them inplace.'''
    shuffle_idxs = np.arange(len(X)) # Get indices to shuffle the inputs and labels. 
    np.random.shuffle(shuffle_idxs)
    return X[shuffle_idxs, :], y[shuffle_idxs, :]


def get_batches(X:np.ndarray, y:np.ndarray, batch_size:int=16) -> Tuple[np.ndarray, np.ndarray]:
    '''Create batches of size batch_size from training data and labels.'''
    # Don't bother with balanced batches. Doesn't help much with accuracy anyway.
    n_batches = len(X) // batch_size + 1
    return np.array_split(X, n_batches, axis=0), np.array_split(y, n_batches, axis=0)


def process_outputs(output:np.ndarray):
    '''Convert model output, which is a (n_samples, n_classes) vector of Softmax-processed logits, to a Numpy array 
    of zeros and ones. The output array can then be used with the inverse_transform method of the fitted one-hot encoder.'''
    y_pred = np.zeros(output.shape)
    # Find the maximum output in the three-value output, and set it to 1. Basically converting to a one-hot encoding.
    for i in range(len(output)):
        j = np.argmax(output[i])
        y_pred[i, j] = 1
    return y_pred 


class BaseClassifier():
    binary_classes = {0:'tolerant', 1:'intolerant'}
    ternary_classes = {0:'aerobe', 1:'anaerobe', 2:'facultative'}

    def __init__(self, n_classes:int=3):
        '''Initialization function for a classifier.'''
        self.scaler = StandardScaler()
        self.n_classes = n_classes # Should be either 2 or 3 for binary or ternary classification. 
        self.classes = BaseClassifier.binary_classes if (n_classes == 2) else BaseClassifier.ternary_classes

    def _predict(self, X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise(NotImplementedError)

    def _decode(self, output:np.ndarray) -> np.ndarray:
        '''Converts model outputs, which is an array of probabilities with size (n_samples, n_classes), into predicted
        labels.'''
        labels = np.argmax(output, axis=1).ravel()
        labels = [self.classes[l] for l in labels] # Convert indices back to strings. 
        return np.array(labels)

    def save(self, path:str) -> NoReturn:
        '''Save the GeneralClassifier instance to a file.'''
        joblib.dump((self), path)

    @classmethod
    def load(cls, path:str):
        '''Load a saved GeneralClassifier object from a file.'''
        return joblib.load(path)

    def predict(self, dataset) -> pd.DataFrame:
        '''Prediction function to use externally. Returns a DataFrame containing both the raw model outputs
        and the predicted labels. 

        :param dataset: A FeatureDataset object containing encoded genomes. 
        :return: A pandas DataFrame with a columns containing the certainty values for each class, as 
            well as the predicted labels.'''
        # Intialize a DataFrame to store the prediction. 
        predictions_df = pd.DataFrame(index=dataset.ids)
        predictions_df.index.name = 'genome_id'

        X, _ = dataset.to_numpy() # Extract the features as a numpy array of shape (n_samples, n_features)
        output, labels = self._predict(X)

        if output is not None: # Output is None for the RandomRelativeClassifier
            for i, c in self.classes.items():
                predictions_df[c] = output[:, i].ravel()
        predictions_df['label'] = labels

        return predictions_df

    def accuracy(self, dataset) -> float:
        '''Compute the balanced accuracy on the input FeatureDataset.''' 
        X, y = dataset.to_numpy(n_classes=self.n_classes)
        _, labels = self._predict(X)
        return balanced_accuracy_score(y, labels)

    def confusion_matrix(self, dataset) -> np.ndarray:
        '''Compute the confusion matrix on the input FeatureDataset.'''
        X, y = dataset.to_numpy(n_classes=self.n_classes)
        _, labels = self._predict(X)
        return confusion_matrix(y, labels)


class RandomRelativeClassifier(BaseClassifier):

    def __init__(self, level:str='Phylum', n_classes:int=3):
        '''Initialize a RandomRelative classifier.'''
        BaseClassifier.__init__(self, n_classes=n_classes)
        self.level = level
        self.n_classes = n_classes

    def fit(self, dataset):
        self.taxonomy = dataset.metadata[[self.level, 'physiology']]
        if self.n_classes == 2:
            self.taxonomy = self.taxonomy.replace({'Aerobe':'tolerant', 'Facultative':'tolerant', 'Anaerobe':'intolerant'})
        elif self.n_classes == 3:
            self.taxonomy = self.taxonomy.replace({'Aerobe':'aerobe', 'Facultative':'facultative', 'Anaerobe':'anaerobe'})

    def _predict(self, X:np.ndarray):
        # Get the taxonomy label at self.level for each genome ID in X.
        X_taxonomy = self.taxonomy.loc[X.ravel(), self.level].values
        labels = []
        for t in X_taxonomy:
            relatives = self.taxonomy[self.taxonomy[self.level] == t] # Get all relatives at the specified level.
            labels.append(np.random.choice(relatives.physiology.values)) # Choose a random physiology label from among the relatives. 
        labels = np.array(labels).ravel()
        return None, labels


class BasePytorchClassifier(torch.nn.Module, BaseClassifier):

    def __init__(self, n_epochs:int=None, batch_size:int=None, n_classes:int=None):
        '''Initialize a classifier using PyTorch.

        :param n_epochs: The maximum number of epochs to train the classifier. 
        :param batch_size: The size of the batches for model training. 
        :param n_classes: The number of classes. This is the output dimension of the second linear layer. 
        '''        
        BaseClassifier.__init__(self, n_classes=n_classes)
        torch.nn.Module.__init__(self)

        self.val_accs, self.train_accs, self.train_losses, self.val_losses = [], [], [], []

        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def _encode(self, labels:np.ndarray) -> torch.FloatTensor:
        '''Converts an array of strings labels into a one-hot encoded PyTorch tensor.'''
        classes = {v:k for k, v in self.classes.items()}
        labels = [classes[l] for l in labels] # Convert the string labels to integers. 
        labels = torch.Tensor(labels).to(torch.int64) # Convert to a tensor of integers for compatibility with one-hot encoder. 
        # If number of classes is not set, it will be inferred as one greater than the largest class value in the input tensor.
        return torch.nn.functional.one_hot(labels, num_classes=self.n_classes).to(torch.float32) # Need to convert back to float. 

    def forward(self, X:np.ndarray) -> torch.FloatTensor:
        '''A forward pass of the model.'''
        X = torch.FloatTensor(X).to(device) # Convert numpy array to Tensor. Make sure it is on the GPU, if available.
        return self.classifier(X)
        
    def _predict(self, X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Predict function for using internally. Returns the model outputs converted to string labels, 
        for use with the balanced accuracy function.
        
        :param X: A numpy array of shape (n_samples, n_features) which contains the genome embeddings. 
        :return: A numpy array of shape (n_samples) containing predicted class labels for each sample. 
        '''
        self.eval() # Model in evaluation mode.
        X = self.scaler.transform(X) # Don't forget to Z-score scale the data!
        output = self(X).detach().numpy() # Convert output tensor to numpy array. 
        labels = self._decode(output) # Convert back to string labels. 
        self.train()
        return output, labels

    def _acc(self, X:np.ndarray, y:np.ndarray) -> float:
        _, labels = self._predict(X) # Get the predicted labels. 
        y = self._decode(y.detach().numpy()) # y has been one-hot encoded, so need to convert back to labels. 
        return balanced_accuracy_score(y, labels)

    def _loss(self, output, y, weight=None):
        '''Mean-squared error loss.'''
        weight = torch.FloatTensor([1] * self.n_classes) if (weight is None) else weight
        # Get a 16-dimensional column vector of weights for each row. 
        weight = torch.matmul(y, weight.reshape(self.n_classes, 1))
        return torch.mean((y - output)**2 * weight)

    def fit(self, dataset, val_dataset):
        '''Train the classifier on input data.'''
        X, y = dataset.to_numpy(n_classes=self.n_classes)
        X_val, y_val = val_dataset.to_numpy(n_classes=self.n_classes)
        
        # Compute loss weights as the inverse frequency.
        weight = torch.FloatTensor([1 / ((y == self.classes[i]).sum() / len(y)) for i in range(self.n_classes)])

        X = self.scaler.fit_transform(X) 
        X_val = self.scaler.transform(X_val)
        y, y_val = self._encode(y), self._encode(y_val) # One-hot encode the labels. 
 
        best_epoch, best_model_weights = 0, copy.deepcopy(self.state_dict()) 

        self.train() # Model in train mode. 

        for epoch in tqdm(range(self.n_epochs), desc='Training classifier...'):

            X, y = shuffle(X, y) # Shuffle the transformed data. 
            for X_batch, y_batch in zip(*get_batches(X, y)):
                y_pred = self(X_batch) # Output will have two or three dimensions, depending on n_classes.
                train_loss = self._loss(y_pred, torch.FloatTensor(y_batch).to(device), weight)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
              
            self.train_losses.append(self._loss(self(X), y, weight=weight).item()) # Store the average weighted train losses over the epoch. 
            self.train_accs.append(self._acc(X, y)) # Store model accuracy on the training dataset. 
            self.val_losses.append(self._loss(self(X_val), y_val).item()) # Store the unweighted loss on the validation data.
            self.val_accs.append(self._acc(X_val, y_val)) # Store model accuracy on the validation dataset. 

            if self.val_accs[-1] >= max(self.val_accs):
                best_epoch = epoch
                best_model_weights = copy.deepcopy(self.state_dict())

        self.load_state_dict(best_model_weights) # Load the best enountered model weights.
        self.best_epoch = best_epoch 
        print(f'PyTorchClassifier.fit: Best validation accuracy of {np.round(max(self.val_accs), 2)} achieved at epoch {self.best_epoch}.')


class LogisticClassifier(BaseClassifier):

    def __init__(self, n_classes:int=3):

        BaseClassifier.__init__(self, n_classes=n_classes)
        # Use the legacy parameters from before my time on this project.
        self.classifier = LogisticRegression(penalty='l2', C=100, max_iter=100000)

    def _encode(self, labels:np.ndarray) -> np.ndarray:
        '''Converts an array of strings labels into a numpy array of integers. Although the LogisticRegression object
        works with strings, this ensures consistent ordering of classes.'''
        classes = {v:k for k, v in self.classes.items()}
        labels = [classes[l] for l in labels] # Convert the string labels to integers. 
        # If number of classes is not set, it will be inferred as one greater than the largest class value in the input tensor.
        return np.array(labels)


    def fit(self, dataset) -> NoReturn:
        X, y = dataset.to_numpy(self.n_classes)
        y = self._encode(y)
        X = self.scaler.fit_transform(X) # Don't forget to Z-scale!
        self.classifier.fit(X, y)

    def _predict(self, X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        X = self.scaler.transform(X)
        output = self.classifier.predict_proba(X) # Returns an array of shape n_samples, n_classes. 
        labels = self._decode(output) # Convert back to string labels. 
        return output, labels


class LinearClassifier(BasePytorchClassifier):

    def __init__(self, input_dim:int=None, output_dim:int=None, lr:float=0.01, n_epochs:int=100, batch_size:int=16):
        '''Initialize a Nonlinear classifier.'''        
        BasePytorchClassifier.__init__(self, n_epochs=n_epochs, n_classes=output_dim, batch_size=batch_size)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), torch.nn.Softmax(dim=1))
        
        # Set weight_decay to correspond to the regularization strength of the LogisticClassifier.
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=0.01)


class NonlinearClassifier(BasePytorchClassifier):
    '''Two-layer neural network for classification.'''
    def __init__(self, 
        input_dim:int=None, 
        hidden_dim:int=512, 
        output_dim:int=None, 
        lr:float=0.0001, 
        n_epochs:int=100, 
        batch_size:int=16):
        '''Initialize a Nonlinear classifier.'''        
        # BasePytorchClassifier.__init__(self, n_epochs=n_epochs, n_classes=output_dim, batch_size=batch_size)
        super().__init__(n_epochs=n_epochs, n_classes=output_dim, batch_size=batch_size)
        
        # NOTE: SoftMax is applied automatically when using the torch CrossEntropy loss function. 
        # Because we are using a custom loss function, we need to apply SoftMax here.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.n_classes),
            torch.nn.Softmax(dim=1)).to(device)
        
        # Set weight_decay to correspond to the regularization strength of the LogisticClassifier.
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=0.01)
    


