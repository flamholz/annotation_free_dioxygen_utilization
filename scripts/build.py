import numpy as np
import pandas as pd
from aerobot.utils import save_hdf, DATA_PATH, FEATURE_TYPES, training_testing_validation_split, FEATURES_PATH
from aerobot.features import chemical
import aerobot
import os
import subprocess
import wget
from typing import NoReturn, Tuple, Dict
import tables
import warnings
import argparse

# NOTE: What is the difference between the RefSeq and GenBank assemblies?

# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

SUPPRESSED_GENOME_IDS = pd.read_csv(os.path.join(DATA_PATH, 'suppressed_genomes.csv'), usecols=['id'])['id']
SUPPRESSED_GENOME_IDS = [id_.split('.')[0] for id_ in SUPPRESSED_GENOME_IDS] # Remove the version ID. 

TERMINAL_OXIDASE_KOS = pd.read_csv(os.path.join(FEATURES_PATH, 'terminal_oxidase_genes.csv')).ko.unique()
CHEMICAL_INPUTS = ['aa_1mer', 'cds_1mer', 'number_of_genes', 'nt_1mer']


# NOTE: Called these 'data' and not 'features' because the same function also works for the labels. 
def load_data(feature_type:str=None, source:str='madin') -> pd.DataFrame:
    '''Load the training data from Madin et. al. This data is stored in an H5 file, as it is too large to store in 
    separate CSVs.'''

    output = dict()
    path = os.path.join(DATA_PATH, f'{source}_datasets.h5')

    if feature_type == 'chemical': # There was a bug here where I was repeatedly loading nt_1mer... would this have thrown an error?
        # kwargs = {f + '_df':pd.read_hdf(path, key=f) for f in CHEMICAL_INPUTS} 
        kwargs = {f + '_df':pd.read_hdf(path, key=f) for f in CHEMICAL_INPUTS} 
        data = chemical.from_features(**kwargs)
    elif feature_type == 'ko_terminal_oxidase_genes':
        ko_df = pd.read_hdf(path, key='ko') 
        data = ko_df[[ko for ko in TERMINAL_OXIDASE_KOS if ko in ko_df.columns]]
    else:
        data = pd.read_hdf(path, key=feature_type)

    return data 


# NOTE: Called these 'data' and not 'features' because the same function also works for the labels. 
def merge_data(jablonska_data:pd.DataFrame, madin_data:pd.DataFrame, fill_nans:bool=True) -> pd.DataFrame:
    '''Merge the training and validation datasets, ensuring that the duplicate entries are taken care of by standardizing
    the index labels.'''

    # Standardize the indices so that the datasets can be compared.
    madin_data.index = [i.split('.')[0] for i in madin_data.index]
    jablonska_data.index = [i.split('.')[0] for i in jablonska_data.index]

    # Combine the datasets, ensuring that we don't drop any feature columns (specifying join='outer').
    data = pd.concat([madin_data, jablonska_data], axis=0, join='outer')
    if fill_nans: # Only fill NaNs if specified, to avoid messing up the label data.
        data = data.fillna(0) # Maybe should add a check to make sure there are no missing columns in the case of labels.

    return data


def fill_missing_taxonomy(metadata:pd.DataFrame) -> pd.DataFrame:
    '''Fill in missing taxonomy information from the GTDB taxonomy strings. This is necessary because
    different data sources have different taxonomy information populated. Note that every entry should have
    either a GTDB taxonomy string or filled-in taxonomy data.
    '''
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. 

    tax = metadata.gtdb_taxonomy.str.split(';', expand=True) # Split the GTDB taxonomy strings.
    tax = tax.apply(lambda x: x.str.split('__').str[1]) # Remove the g__, s__, etc. prefixes.
    tax.columns = levels # Label the taxonomy columns. 
    # Use the tax DataFrame to fill in missing taxonomy values in the labels DataFrame
    metadata = metadata.replace('no rank', np.nan).combine_first(tax)
    
    # I noticed that no entries have no assigned species, but some do not have an assigned genus. 
    # Decided to autofill genus with the species string. I checked to make sure that every non-NaN genus is 
    # consistent with the genus in the species string, so this should be OK.
    assert np.all(~metadata.Species.isnull()), 'fill_missing_taxonomy: Some entries have no assigned Species'
    metadata['Genus'] = metadata['Species'].apply(lambda s : s.split(' ')[0])


    unclassified = metadata[metadata.isnull().any(axis=1)] # Get all rows with any null entry. 
    for row in [row for row in unclassified.itertuples() if not (row.Family is None)]:
        family = row.Family
        relatives = metadata[metadata.Family == row]
        if len(relatives) > 0:
            relative = relatives.iloc[0]
            row.Domain = relative.Domain 
            row.Phylum = relative.Phylum 
            row.Class = relative.Class
            row.Order = relative.Order
            labels.loc[row.Index] = row # Replace the row with the filled-in row.

    metadata[levels] = metadata[levels].fillna('no rank') # Fill in all remaining blank taxonomies with 'no rank'
    return metadata


# TODO: This function should be like three lines of code, not sure why it was longer. 
def remove_duplicates(data:pd.DataFrame) -> pd.DataFrame:
    '''Checks for duplicate IDs. If the data are identical, the first entry is kept. If not, 
    then both duplicates are dropped.'''
    n = len(data)
    data = data[~data.index.duplicated(keep='first')]
    print(f'remove_duplicates: Removed {n - len(data)} duplicate entries.')
    return data


def remove_suppressed(data:pd.DataFrame) -> pd.DataFrame:
    '''Removes genomes which have been suppressed by NCBI.'''
    n = len(data)
    # Convert the IDs to RefSeq, which is the format of the suppressed IDs obtained from NCBI FTP site.
    refseq_ids = np.array([id_.replace('GCA', 'GCF') for id_ in data.index])
    data = data[~np.isin(refseq_ids, SUPPRESSED_GENOME_IDS)]
    print(f'remove_suppressed: Removed {n - len(data)} suppressed entries.')
    return data    



if __name__ == '__main__':

    # download_data() # Download training data from Google Cloud if it has not been already.
    datasets = dict()

    # for feature_type in ['metadata'] + FEATURE_TYPES:
    for feature_type in FEATURE_TYPES:
        print(f'\nBuilding {feature_type} data...')

        # Load in the datasets.
        jablonska_data = load_data(feature_type, source='jablonska')
        madin_data = load_data(feature_type, source='madin')

        data = merge_data(jablonska_data, madin_data, fill_nans=feature_type!='metadata')
        data = remove_duplicates(data)
        data = remove_suppressed(data)

        if feature_type == 'metadata':
            data = fill_missing_taxonomy(data)

        datasets[feature_type] = data
 

    training_datasets, testing_datasets, validation_datasets = training_testing_validation_split(datasets)

    print('\nNumber of genomes in training datasets:', len(training_datasets['aa_1mer']))
    print('Number of genomes in validation datasets:', len(validation_datasets['aa_1mer']))
    print('Number of genomes in testing datasets:', len(testing_datasets['aa_1mer']))

    # Save each dataset to an HDF file.
    print('\nSaving training data to', os.path.join(DATA_PATH, 'training_datasets.h5'))
    save_hdf(training_datasets, os.path.join(DATA_PATH, 'training_datasets.h5'))
    print('Saving validation data to', os.path.join(DATA_PATH, 'validation_datasets.h5'))
    save_hdf(validation_datasets, os.path.join(DATA_PATH, 'validation_datasets.h5'))
    print('Saving testing data to', os.path.join(DATA_PATH, 'testing_datasets.h5'))
    save_hdf(testing_datasets, os.path.join(DATA_PATH, 'testing_datasets.h5'))



