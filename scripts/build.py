'''Merge the Madin and Jablonska datasets, removing redundant genomes. since they overlap somewhat. Then split them into training 
and validation sets with no repeated species across the two sets. '''
import numpy as np
import pandas as pd
from aerobot.io import save_hdf, DATA_PATH, FEATURE_TYPES
import os
import subprocess
import wget
from typing import NoReturn, Tuple, Dict
import tables
import warnings
# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

CHEMICAL_INPUTS = ['aa_1mer', 'cds_1mer', 'aa_1mer', 'metadata', 'nt_1mer']
CANONICAL_NTS = ['A', 'C', 'G', 'T']
GC_NTS = ['G', 'C']

RNA_NOSC_DF = pd.read_csv(os.path.join(DATA_PATH, 'nt_nosc.csv'))
RNA_NOSC_DF = RNA_NOSC_DF[RNA_NOSC_DF.type == 'RNA'].set_index('letter_code')
RNA_NC = RNA_NOSC_DF.NC
RNA_ZC = RNA_NOSC_DF.NOSC
CANONICAL_NTS_ALL = RNA_NOSC_DF.index.unique().tolist() # Includes DNA and RNA names

AA_NOSC_DF = pd.read_csv(os.path.join(DATA_PATH, 'aa_nosc.csv'), index_col=0)
AA_NC = AA_NOSC_DF.NC
AA_ZC = AA_NOSC_DF.NOSC
CANONICAL_AAS = AA_NOSC_DF.index.tolist()

TERMINAL_OXIDASE_KOS = pd.read_csv(os.path.join(DATA_PATH, 'terminal_oxidase_kos.csv')).ko.unique()
# KOS = pd.read_csv(os.path.join(DATA_PATH, 'kos.csv')) # The union of KO groups in the Madin and Jablonska data.


def get_chemical_rna_features(nt_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate chemical features of the RNA coding sequences. Calculates the formal 
    C oxidation state of mRNAs, as well as the C, N, O and S content. 

    :param nt_1mer_df: A DataFrame containing the counts of single nucleotides, either RNA or DNA. Assumed to single stranded.
    :return: A pd.Series containing the formal C oxidation state of the RNA coding sequences.
    '''
    my_nts = sorted(set(nt_1mer_df.columns).intersection(CANONICAL_NTS_ALL))
    canonical_data = nt_1mer_df[my_nts]
    NC_total = (canonical_data @ RNA_NC[my_nts])
    Ne_total = (canonical_data @ (RNA_ZC[my_nts] * RNA_NC[my_nts]))
    mean_cds_zc = Ne_total / NC_total
    mean_cds_zc.name = 'cds_nt_zc'
    cols = [mean_cds_zc]
    for elt in 'CNO':
        col = 'N{0}'.format(elt)
        ns = RNA_NOSC_DF[col]
        totals = (canonical_data @ ns[my_nts])
        means = totals / canonical_data.sum(axis=1)
        means.name = 'cds_nt_{0}'.format(col)
        cols.append(means)
    return pd.concat(cols, axis=1)


def get_chemical_gc_content(nt_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate the GC content of the canonical nucleotides.

    :param nt_1mer_df: A DataFrame containing the counts of single nucleotides, either RNA or DNA. Assumed to single stranded.
    :return: A pd.Series containing the GC content of each genome.
    '''
    canonical_data = nt_1mer_df[CANONICAL_NTS]
    gc_content = canonical_data[GC_NTS].sum(axis=1) / canonical_data.sum(axis=1)
    gc_content.name = 'gc_content'
    return gc_content


def get_chemical_aa_features(aa_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate chemical features of the coding sequences.Calculates the formal C oxidation state 
    of mRNAs, as well as the C, N, O and S content. 

    :param aa_1mer_df: A DataFrame containing the counts of single amino acids.
    :return: A pd.Series containing the formal C oxidation state.
    '''
    aas = sorted(set(aa_1mer_df.columns).intersection(CANONICAL_AAS))
    canonical_data = aa_1mer_df[aas]
    NC_total = (canonical_data @ AA_NC[aas])
    Ne_total = (canonical_data @ (AA_ZC[aas] * AA_NC[aas]))
    mean_cds_zc = Ne_total / NC_total
    mean_cds_zc.name = 'cds_aa_zc'
    cols = [mean_cds_zc]
    for elt in 'CNOS':
        col = 'N{0}'.format(elt)
        ns = AA_NOSC_DF[col]
        totals = (canonical_data @ ns[aas])
        means = totals / canonical_data.sum(axis=1)
        means.name = 'cds_aa_{0}'.format(col)
        cols.append(means)
    return pd.concat(cols, axis=1)


def get_chemical_features(metadata_df:pd.DataFrame=None, cds_1mer_df:pd.DataFrame=None, aa_1mer_df:pd.DataFrame=None, nt_1mer_df=None) -> pd.DataFrame:
    '''Compute chemical features using other feature DataFrames and the metadata.

    :param metadata_df: DataFrame containing the gene metadata.
    :param nt_1mer_df: DataFrame containing the nt_1mer feature data.
    :param aa_1mer_df: DataFrame containing the aa_1mer feature data.
    :param cds_1mer_df: DataFrame containing the cds_1mer feature data.
    :return: A DataFrame containing the chemical feature data.
    '''
    n_genes = metadata_df.drop_duplicates()['number_of_genes']
    gc_content = get_chemical_gc_content(nt_1mer_df)
    aa_features = get_chemical_aa_features(aa_1mer_df)
    rna_features = get_chemical_rna_features(cds_1mer_df)

    return pd.concat([gc_content, n_genes, aa_features, rna_features], axis=1).dropna(axis=0)


def load_data_jablonska(feature_type:str, path:str=os.path.join(DATA_PATH, 'jablonska/'), ):
    df = pd.read_csv(os.path.join(path, f'jablonska_{feature_type}.csv'), index_col=0)
    if feature_type == 'metadata': # For some reason, the genome doesn't get set as the index here.
        df = df.set_index('genome')
    return df


def load_data_madin(feature_type:str, path:str=os.path.join(DATA_PATH, 'madin/madin.h5')):
    # Create a dictionary mapping each feature type to a key in the HD5 file.
    key_map = {f:f for f in FEATURE_TYPES} # Most keys are the same as the feature type names.
    key_map.update({'embedding.genome':'WGE', 'embedding.geneset.oxygen':'OGSE', 'metadata':'AF'})
    key_map.update({'labels':'labels'})
    return pd.read_hdf(path, key=key_map[feature_type])


def load_data(feature_type:str=None, source:str='madin') -> Dict[str, pd.DataFrame]:
    '''Load the training data from Madin et. al. This data is stored in an H5 file, as it is too large to store in 
    separate CSVs. 

    :param path: The path to the HD5 file containing the training data.
    :param feature_type: The feature type to load.
    :return: A dictionary containing the feature data and corresponding labels.'''
    output = dict()
    load_data_func = load_data_jablonska if source == 'jablonska' else load_data_madin

    output['labels'] = load_data_func('labels')

    if feature_type == 'chemical': # There was a bug here where I was repeatedly loading nt_1mer... would this have thrown an error?
        # kwargs = {f + '_df':pd.read_hdf(path, key=f) for f in CHEMICAL_INPUTS} 
        kwargs = {f + '_df':load_data_func(f) for f in CHEMICAL_INPUTS} 
        features = get_chemical_features(**kwargs)
    elif feature_type == 'KO.geneset.terminal_oxidase':
        ko_df = load_data_func('KO')
        features = ko_df[[ko for ko in TERMINAL_OXIDASE_KOS if ko in ko_df.columns]]
    elif 'metadata' in feature_type:
        metadata_df = load_data_func('metadata')
        # Calculate the percentage of oxygen genes for the combined dataset and and add it to the dataset
        metadata_df['pct_oxygen_genes'] = metadata_df['oxygen_genes'] / metadata_df['number_of_genes']
        features = metadata_df[[feature_type.split('.')[-1]]]
    else:
        features = load_data_func(feature_type)
    output['features'] = features # Add the features to the output dictionary. 

    return output


def merge_datasets(training_dataset:Dict[str, pd.DataFrame], validation_dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    '''Merge the training and validation datasets, ensuring that the duplicate entries are taken care of by standardizing
    the index labels. Note that this function DOES NOT remove duplicate entries.

    :param training_dataset: A dictionary containing the training features and labels. 
    :param validation_dataset: A dictionary containing the validation features and labels. 
    :return: A dictionary containing the combined validation and training datasets.  
    '''
    def standardize_index(dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        '''Standardizes the indices the labels and features DataFrames in a dataset
        by removing the '.d' suffix, where d is a digit. This allows for comparison between datasets.

        :param dataset: The dataset to operate on.
        :return: The dataset with standardized indices.
        '''
        for key, data in dataset.items():
            data.index = [i.split('.')[0] for i in data.index]
            dataset[key] = data
        return dataset

    # Standardize the indices so that the datasets can be compared.
    training_dataset, validation_dataset = standardize_index(training_dataset), standardize_index(validation_dataset)
    # Unpack the features and labels dataframes from the datasets.
    training_features, training_labels = training_dataset['features'], training_dataset['labels'].drop(columns=['annotation_file', 'embedding_file'])
    validation_features, validation_labels = validation_dataset['features'], validation_dataset['labels'].drop(columns=['annotation_file', 'embedding_file'])

    # Combine the datasets, ensuring that the features columns which do not overlap are removed (with the join='inner')
    # features = pd.concat([training_features, validation_features], axis=0, join='inner')
    features = pd.concat([training_features, validation_features], axis=0, join='outer')
    features = features.fillna(0)
    labels = pd.concat([training_labels, validation_labels], axis=0, join='outer')

    return {'features':features, 'labels':labels}


def fill_missing_taxonomy(labels:pd.DataFrame) -> pd.DataFrame:
    '''Fill in missing taxonomy information from the GTDB taxonomy strings. This is necessary because
    different data sources have different taxonomy information populated. Note that every entry should have
    either a GTDB taxonomy string or filled-in taxonomy data.

    :param labels: The combined training and validation labels DataFrame.
    :return: The labels DataFrame with corrected taxonomy.
    '''
    levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. 

    tax = labels.gtdb_taxonomy.str.split(';', expand=True) # Split the GTDB taxonomy strings.
    tax = tax.apply(lambda x: x.str.split('__').str[1]) # Remove the g__, s__, etc. prefixes.
    tax.columns = levels # Label the taxonomy columns. 
    # Use the tax DataFrame to fill in missing taxonomy values in the labels DataFrame
    labels = labels.replace('no rank', np.nan).combine_first(tax)
    
    # I noticed that no entries have no assigned species, but some do not have an assigned genus. 
    # Decided to autofill genus with the species string. I checked to make sure that every non-NaN genus is 
    # consistent with the genus in the species string, so this should be OK.
    assert np.all(~labels.Species.isnull()), 'autofill_taxonomy: Some entries have no assigned Species'
    labels['Genus'] = labels['Species'].apply(lambda s : s.split(' ')[0])

    for level in levels[::-1]: # Make sure all taxonomy has been populated.
        n_unclassified = np.sum(labels[level].isnull())
        if n_unclassified > 0:
            print(f'\tfill_missing_taxonomy: {n_unclassified} entries have no assigned {level.lower()}.')
        # assert n_unclassified == 0, f'fill_missing_taxonomy: {n_unclassified} entries have no assigned {level.lower()}.'
    # labels[levels] = labels[levels].fillna('no rank') # Fill any NaNs with a "no rank" string for consistency.
    labels[levels] = labels[levels].fillna('no rank') # Fill in all remaining blank taxonomies with 'no rank'
    return labels


def remove_duplicates(data:pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    '''Checks for duplicate IDs. If the data are identical, the first entry is kept. If not, 
    then both duplicates are dropped.

    :param data: A DataFrame containing all data.
    :return: A 3-tuple (new_df, duplicate_ids, removed_ids), where new_df is the DataFrame
        with duplicates removed, duplicate_ids are the IDs that were identical duplicates (one is retained),
        and removed_ids are the IDs that were removed entirely due to inconsistent data.
    '''
    duplicate_ids = data.index[data.index.duplicated()]
    ids_to_remove = []

    for id_ in duplicate_ids:
        duplicate_entries = data.loc[id_] # Get all entries in the DataFrame which match the ID.
        # NOTE: Keeping the first entry prefers keeping the entries from the training dataset, due to how they are concatenated.
        # This should ensure that GTDB taxonomy is preferred in the labels DataFrames.
        first_entry = duplicate_entries.iloc[0] # Get the first duplicate entry.
        # Check if the duplicate entries are consistent. If not, remove. 
        if not all(duplicate_entries == first_entry):
            ids_to_remove.append(id_)

    data = data.drop(ids_to_remove, axis=0) # Remove the inconsistent entries.
    duplicated = data.index.duplicated() # Recompute duplicated entries.
    duplicate_ids = data.index[duplicated].tolist() # Get the IDs of the duplicate entries. 

    return data[~duplicated].copy(), duplicate_ids, ids_to_remove


def training_testing_validation_split(all_datasets:Dict[str, pd.DataFrame], random_seed:int=42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    '''Split concatenated feature dataset into training and validation sets using phylogeny.
    
    :param all_datasets: A dictionary mapping each feature type to the corresponding dataset.
    :param random_seed: A random seed for reproducibility.
    :return: A 3-tuple of dictionaries. Each dictionary maps each feature type to a training (first tuple element) testing
        (second tuple element), or validation (third tuple element) dataset.
    '''
    labels = all_datasets['labels'] # Get the labels out of the dictionary.
    # Group IDs by phylogenetic class. Convert to a dictionary mapping class to a list of indices.
    ids_by_class = labels.groupby('Class').apply(lambda x: x.index.tolist(), include_groups=False).to_dict()
    np.random.seed(42) # For reproducibility.

    testing_ids = []
    for class_, ids in ids_by_class.items():
        n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
        selected_ids = np.random.choice(ids, n, replace=False)
        remaining_ids = [id_ for id_ in ids if id_ not in selected_ids]
        testing_ids.extend(selected_ids)
        ids_by_class[class_] = remaining_ids # Make sure only the remaining IDs are left. 

    # Take 20 percent of the remaining IDs for each class for the validation set. 
    validation_ids = []
    for class_, ids in ids_by_class.items():
        n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
        selected_ids = np.random.choice(ids, n, replace=False)
        remaining_ids = [id_ for id_ in ids if id_ not in selected_ids]
        validation_ids.extend(selected_ids)
        ids_by_class[class_] = remaining_ids # Make sure only the remaining IDs are left.

    # Add all remaining IDs to the training dataset. 
    training_ids =  []
    for ids in ids_by_class.values():
        training_ids.extend(ids)

    # Split the concatenated dataset back into training and validation sets
    training_datasets, testing_datasets, validation_datasets = dict(), dict(), dict()
    for feature_type, dataset in all_datasets.items():
        training_datasets[feature_type] = dataset[dataset.index.isin(training_ids)]
        testing_datasets[feature_type] = dataset[dataset.index.isin(testing_ids)]
        validation_datasets[feature_type] = dataset[dataset.index.isin(validation_ids)]

    return training_datasets, testing_datasets, validation_datasets

# def training_validation_split(all_datasets:Dict[str, pd.DataFrame], random_seed:int=42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
#     '''Split concatenated feature dataset into training and validation sets using phylogeny.
    
#     :param all_datasets: A dictionary mapping each feature type to the corresponding dataset.
#     :param random_seed: A random seed for reproducibility.
#     :return: A 2-tuple of dictionaries. Each dictionary maps each feature type to a training (first tuple element) or validation
#         (second tuple element) dataset.
#     '''
#     labels = all_datasets['labels'] # Get the labels out of the dictionary.
#     # Group IDs by phylogenetic class. Convert to a dictionary mapping class to a list of indices.
#     ids_by_class = labels.groupby('Class').apply(lambda x: x.index.tolist(), include_groups=False).to_dict()

#     counts_by_class = {k: len(v) for k, v in ids_by_class.items()}

#     np.random.seed(random_seed) # For reproducibility. 
#     validation_ids = []
#     for class_, ids in ids_by_class.items():
#         n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
#         validation_ids.extend(np.random.choice(ids, n, replace=False))

#     # Split the concatenated dataset back into training and validation sets
#     training_datasets, validation_datasets = dict(), dict()
#     for feature_type, dataset in all_datasets.items():
#         training_datasets[feature_type] = dataset[~dataset.index.isin(validation_ids)]
#         validation_datasets[feature_type] = dataset[dataset.index.isin(validation_ids)]

#     return training_datasets, validation_datasets



if __name__ == '__main__':

    # download_data() # Download training data from Google Cloud if it has not been already.

    all_datasets = dict()
    for feature_type in FEATURE_TYPES:
        print(f'Building datasets for {feature_type}...')
        # Load in the datasets.
        jablonska_dataset = load_data(feature_type, source='jablonska')
        madin_dataset = load_data(feature_type, source='madin')

        print(f'\tMerging datasets...')
        dataset = merge_datasets(jablonska_dataset, madin_dataset)
        features, labels = dataset['features'], dataset['labels']

        # Fill in gaps in the taxonomy data using the GTDB taxonomy strings.
        dataset['labels'] = fill_missing_taxonomy(dataset['labels'])

        for key, data in dataset.items(): # key is "features" or "labels"
            data, duplicate_ids, removed_ids = remove_duplicates(data)
            if len(removed_ids) > 0:
                print(f'\tRemoved {len(removed_ids)} inconsistent entries in {key}.')
            dataset[key] = data

        # If there are already labels in the dictionary, check to make sure the new labels are equal.
        if 'labels' in all_datasets: # NOTE: There should be 3587 labels, 3480 with no duplicates
            l1, l2 = len(dataset['labels']), len(all_datasets['labels'])
            assert l1 == l2, f'Labels are expected to be the same length, found lengths {l1} and {l2}. Failed on feature type {feature_type}.'
            assert np.all(dataset['labels'].physiology.values == all_datasets['labels'].physiology.values), f'Labels are expected to be the same across datasets. Failed on feature type {feature_type}.'
        else: # Add labels to the dictionary if they aren't there already.
            all_datasets['labels'] = dataset['labels']
        
        all_datasets[feature_type] = dataset['features']

    training_datasets, validation_datasets = training_validation_split(all_datasets)

    print('Saving the datasets...')
    save_hdf(all_datasets, os.path.join(DATA_PATH, 'updated_all_datasets.h5'))
    save_hdf(training_datasets, os.path.join(DATA_PATH, 'updated_training_datasets.h5'))
    save_hdf(validation_datasets, os.path.join(DATA_PATH, 'updated_validation_datasets.h5'))



