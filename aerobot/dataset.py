
'''Code for loading and processing the training and validation datasets created by the build_datasets script.'''
import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
from aerobot.chemical import chemical_get_features
from aerobot.io import load_hdf, FEATURE_SUBTYPES, FEATURE_TYPES, DATA_PATH
import json

# TODO: Should constant-sum scaling happen before or after removal of the X amino acids? And before or after we
# filter the k-mers in the validation datasets (EMP and Black Sea)?

# Issue I see with normalizing after filtering by k-mers is that it might make it seem like some k-mers are a more
# important part of the input sequence than they actually are, i.e. it will bias the normalized vector in favor of the
# k-mers we are selecting. I think it might be best to normalize prior to filtering features. 

# TODO: Another question is when to drop the feature columns with NaNs. There is a possibility that columns with NaNs in the training
# and validation sets may not align. I think it might be best to drop NaN columns in the training set and then fill with zeros for the
# validation set to avoid data leakage. If we use the validation set to select which features we train on, that is data leakage.



def dataset_align(dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    '''Align the features and labels in a dataset, so that the indices match.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: The input datset with the indices in the features and labels DataFrames matched and aligned.
    '''
    features, labels = dataset['features'], dataset['labels'] # Unpack the stored DataFrames.
    n = len(features) # Get the original number of elements in the feature DataFrame for checking later. 
    features, labels  = features.align(labels, join='inner', axis=0) # Align the indices.

    # Make sure everything worked as expected.
    assert np.all(np.array(features.index) == np.array(labels.index)), 'dataset_align: Indices in training labels and data do not align.'
    assert len(features) == n, f'dataset_align: {n - len(features)} rows of data were lost during alignment.'

    return {'features':features, 'labels':labels} # Return the aligned dataset.


def dataset_normalize(dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    '''Apply constant-sum scaling to the rows in the dataset.

    :param dataset
    :return: A dataset with constant-sum normalized rows. 
    '''
    dataset['features'] = dataset['features'].apply(lambda row : row / row.sum(), axis=1)
    #assert np.all(dataset['features'].values.sum(axis=1) == 1), 'dataset_normalize: Normalization failed.'
    return dataset


def dataset_to_numpy(dataset:Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    '''Convert the input dataset, which is a dictionary of pandas DataFrames, to a dictionary mapping
    'features' and 'labels' to numpy arrays. The 'labels' array contains only the values in the 
    physiology column.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: The input dataset with the DataFrames converted to numpy arrays. The the feature array is of 
        type is np.float32 and size (n, d) where n is the number of entries and d is the feature dimension.
        The labels array is one dimensional and of length n, and is of type np.object_.
    '''
    numpy_dataset = dict() # Create a new dataset.
    numpy_dataset['features'] = None if dataset['features'] is None else dataset['features'].values # .astype(np.float32)
    numpy_dataset['labels'] = dataset['labels'].physiology.values
    return numpy_dataset


def dataset_clean(dataset:Dict[str, pd.DataFrame], feature_type:str=None, binary:bool=False) -> Tuple[Dict]:
    '''Clean up the input dataset by doing the following.
        (1) Formatting the physiology labels for ternary or binary classification. 
        (2) Standardizing the feature set and order. The reference set of features against which all datasets are standardized
            are the non-NaN columns in the training set; see dataset_load_feature_order for more information.
        (3) Filling all remaining NaNs in the dataset with 0.
        (4) Aligning the feature and label indices.

    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :param feature_type: The feature type of the dataset. 
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :return: The cleaned-up dataset.
    '''
    # Select a label map for the binary or ternary classification task.
    if binary:
        label_map = {"Aerobe": "tolerant", "Facultative": "tolerant", "Anaerobe": "intolerant"}
    else:
        label_map = {"Aerobe": "aerobe", "Facultative": "facultative", "Anaerobe": "anaerobe"}
    dataset['labels'].physiology = dataset['labels'].physiology.replace(label_map) # Format the labels.

    if dataset['features'] is not None:
        # Ensure that the column ordering in the dataset matches the reference. This should remove
        # all columns which were NaN in the training set. 
        dataset['features'] = dataset['features'][dataset_load_feature_order(feature_type)]
        dataset['features'] = dataset['features'].fillna(0) # Fill in remaining NaNs with zeros. 
        dataset = dataset_align(dataset) # Align the features and labels indices.

    return dataset


def dataset_load(path:str, feature_type:str=None, normalize:bool=True) -> Dict:
    '''Load a dataset for a particular feature type from the specified path. No filtering is applied to any of the
    features in the dataset at this point. 
    
    :param feature_type: Feature type to load from the HDF file. If None, genome IDs are used 
        as the feature type (for working with MeanRelative classifiers).
    :param path: Path to the HDF dataset to load. 
    :return: A dictionary with keys 'features' and 'labels' containing the feature data and metadata.
    '''
    subtype = None
    assert feature_type in FEATURE_TYPES + FEATURE_SUBTYPES + [None], f'dataset_load: Input feature type {feature_type} is invalid.'
    # Special case if the feature_type is a "subtype", which is stored as a column in the metadata.
    if (feature_type in FEATURE_SUBTYPES) and (feature_type is not None):
        feature_type, subtype = feature_type.split('.')
        
    dataset = load_hdf(path, feature_type=feature_type) # Read from the HDF file.
    if subtype is not None: # If a feature subtype is given, extract the information from the metadata.
        dataset['features'] = dataset['features'][[subtype]]

    # If the normalize option is specified, and the feature type is "normalizable," then normalize the rows.
    is_normalizable_feature_type = (feature_type is not None) and (('aa_' in feature_type) or ('nt_' in feature_type) or ('cds_' in feature_type))
    if normalize and is_normalizable_feature_type:
        dataset = dataset_normalize(dataset)

    return dataset


def dataset_get_features(dataset:Dict[str, pd.DataFrame]) -> np.ndarray:
    '''Extract the names of the columns in the features DataFrame for a given dataset.

    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: A numpy array of features in the same order as the columns in the features DataFrame. 
    '''
    assert isinstance(dataset['features'], pd.DataFrame), 'dataset_get_features: Input dataset must contain DataFrames.'
    features = dataset['features'].columns
    return features.to_numpy()


def dataset_load_feature_order(feature_type:str, drop_x:bool=True, drop_na:bool=True) -> np.ndarray:
    '''Load the columns ordering for a particular feature type. This function returns columns ordered
    in the same way as the training dataset, which is used as a reference throughout the project.

    :param feature_type: The feature type for which to load data.
    :param drop_na: Whether or not to drop the features which contain NaN values. 
    :param drop_x: Whether or not to drop the X amino acids when amino acid feature sets are being loaded.
    :return: A numpy array of features, which are the columns of the features DataFrame for the input feature type. 
    '''
    dataset = dataset_load(os.path.join(DATA_PATH, 'updated_training_datasets.h5'), feature_type=feature_type) # Load the training dataset. 
    if drop_na: # Drop any feature columns which contain NaNs. 
        dataset['features'] = dataset['features'].dropna(axis=1)
    if ('aa_' in feature_type) and (drop_x): # Remove all unknown amino acids from the feature set.
        dataset['features'] = dataset['features'][[f for f in dataset['features'].columns if 'X' not in f]]
    return dataset_get_features(dataset)


def dataset_load_training_validation(feature_type:str, binary:bool=False, to_numpy:bool=True, normalize:bool=True) -> Tuple[Dict]:
    '''Load training and validation datasets for the specified feature type.

    :param feature_type: The feature type for which to load data.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :param to_numpy: Whether or not to convert the feature sets to numpy ndarrays for model compatibility.
    :param normalize: Whether or not to constant-sum scale the k-mer counts. This will only be applied if the feature type is "normalizable."
    :return: A 2-tuple of dictionaries with the cleaned-up training and validation datasets as numpy arrays.
    '''
    training_dataset = dataset_load(os.path.join(DATA_PATH, 'updated_training_datasets.h5'), feature_type=feature_type, normalize=normalize)
    validation_dataset = dataset_load(os.path.join(DATA_PATH, 'updated_validation_datasets.h5'), feature_type=feature_type, normalize=normalize)

    # Clean up both datasets.
    validation_dataset = dataset_clean(validation_dataset, binary=binary, feature_type=feature_type)
    training_dataset = dataset_clean(training_dataset, binary=binary, feature_type=feature_type)

    if to_numpy:
        validation_dataset = dataset_to_numpy(validation_dataset)
        training_dataset = dataset_to_numpy(training_dataset)

    return training_dataset, validation_dataset


def dataset_clean_features(df:pd.DataFrame, feature_type:str='aa_3mer'):
    '''Make sure the features in the input data match the features (including order) of the data on 
    which the model was trained.

    :param df: The data on which to run the model.
    :param feature_type: The feature type of the data in the DataFrame.
    '''
    feature_order = dataset_load_feature_order(feature_type) # Load in the correct features.
    
    missing = 0
    for f in feature_order:
        # If the data is missing a feature, fill it in with zeros.
        if f not in df.columns:
            missing += 1
            df[f] = np.zeros(len(df))

    print('dataset_clean_features:', missing, feature_type, 'features are missing from the input data. Filled missing data with 0.')
    df = df[feature_order] # Ensure the feature ordering is consistent. 
    return df