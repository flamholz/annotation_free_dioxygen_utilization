from aerobot.io import DATA_PATH, RESULTS_PATH
import numpy as np 
import os
import pandas as pd 
import subprocess
from typing import Dict

DATA_PATH = os.path.join(DATA_PATH, 'earth_microbiome')
LEVELS = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']


def load_habitat_map(merge_human_habitats:bool=True) -> Dict[str, str]:
    '''Loads a dictionary which maps habitat names in the Earth Microbiome Project data to cleaned-up names. 

    :param merge_human_habitats: This breaks up the human habitats into more granular sub-categories. 
    :return: The habitat map.     
    '''
    # Replace habitat names with more general categories. 
    habitat_map = pd.read_csv(os.path.join(DATA_PATH, 'habitats.csv'), index_col=0).to_dict()['habitat']

    if not merge_human_habitats:
        # Change the human entries in the habitat map to distinguish between different environments. 
        habitat_map['Human feces'] = 'Human feces'
        habitat_map['Human fecal'] = 'Human feces'
        habitat_map.pop('Human oral')
        habitat_map.pop('Human colon tissue')
        habitat_map.pop('Human bile duct')
        habitat_map.pop('Premature human infant gut')
        habitat_map.pop('Human skin')
        habitat_map.pop('Human gut')

    return habitat_map


def parse_taxonomy_metadata(metadata_df:pd.DataFrame) -> pd.DataFrame:
    '''Parse the taxonomy strings in the metadata, so that there are separate columns for each
    taxonomic level.'''

    def parse_taxonomy_string(taxonomy_string:str) -> Dict[str, str]:
        '''Parse a taxonomy string into a dictionary mapping the taxonomic level to the organism's taxonomy.'''
        prefix_map = {'d':'Domain', 'p':'Phylum', 'c':'Class', 'o':'Order', 'f':'Family', 'g':'Genus', 's':'Species'}
        # Sometimes the entry is NaN, which I am taking to mean no taxonomic label. 
        if type(taxonomy_string) != str:
            return dict()

        taxonomy = taxonomy_string.split(';')
        taxonomy = [tuple(t.split('__')) for t in taxonomy]
        taxonomy = {prefix_map[p]:t for p, t in taxonomy}
        return taxonomy

    parsed_metadata_df = []
    for metadata_row in metadata_df.itertuples():
        row = dict()
        row['genome_id'] = metadata_row.Index
        row['habitat'] = metadata_row.habitat
        row.update(parse_taxonomy_string(metadata_row.ecosystem)) # For whatever reason, the taxonomy string is under the "ecosystem" column.
        parsed_metadata_df.append(row)

    parsed_metadata_df = pd.DataFrame(parsed_metadata_df)
    parsed_metadata_df = parsed_metadata_df.fillna('') # Fill any remaining NaNs with empty strings.

    return parsed_metadata_df 


def get_taxonomy_coverage_df(metadata_df:pd.DataFrame):
    '''Get a DataFrame indicating the proportion of organisms in each habitat which have a taxonomy label at 
    each taxonomic level.'''

    def get_coverage(df:pd.DataFrame, level:str):
        '''Get the fraction of organisms with a label at the specified level.'''
        coverage = df[level].apply(len).values
        coverage = np.sum(coverage > 0) / len(coverage)
        return coverage

    metadata_df = parse_taxonomy_metadata(metadata_df)
    # Clean up the habitat labels. 
    metadata_df.habitat = metadata_df.habitat.replace(load_habitat_map(merge_human_habitats=False))
    # Some habitats don't seem to be matching up, so try to remove any potential reasons for this. 
    metadata_df.habitat = metadata_df.habitat.str.lower()
    metadata_df.habitat = metadata_df.habitat.str.strip()

    taxonomy_coverage_df = []
    for habitat, habitat_df in metadata_df.groupby('habitat'):
        row = dict()
        row['habitat'] = habitat
        for level in LEVELS:
            row[level] = get_coverage(habitat_df, level)
        taxonomy_coverage_df.append(row)
    taxonomy_coverage_df = pd.DataFrame(taxonomy_coverage_df) 

    return taxonomy_coverage_df


def filter_data(df:pd.DataFrame) -> pd.DataFrame:
    '''Apply some quality controls to the Earth Microbiome Project data:
        (1) Filter out genomes with less than 50 percent completeness. 
        (2) Filter out entries from samples with fewer than 10 MAGs.
        (3) Filter out entries from habitats with fewer than 10 samples. 
    
    :param df: A pandas DataFrame containing both the classifier predictions and Earth Microbiome Project metadata. 
    :return: A filtered DataFrame. 
    '''
    # Filter out low-completeness MAGs.
    df = df[df.completeness > 50]
    print('filter_data:', len(df), 'genomes with nore than 50 percent completeness.')
    
    # Filter out entries from samples with fewer than ten MAGs
    counts = df.groupby('metagenome_id').apply(len)
    ids_to_keep = counts.index[counts > 10]
    df = df[df.metagenome_id.isin(ids_to_keep)]
    print('filter_data:', len(df), 'genomes from samples with more than ten genomes.')

    # Filter our habitats with fewer than ten samples.
    counts = df.groupby('habitat').apply(len)
    habitats_to_keep = counts.index[counts > 10]
    df = df[df.habitat.isin(habitats_to_keep)]
    print('filter_data:', len(df), 'genomes from habitats with more than ten samples')
    
    return df


def get_aerobe_anaerobe_fraction_df(predictions_df:pd.DataFrame, metadata_df:pd.DataFrame):

    # Combine the predictions and metadata DataFrames. 
    df = pd.concat([metadata_df, predictions_df], axis=1)
    # Apply the habitat map to the data. 
    df.habitat = df.habitat.replace(load_habitat_map(merge_human_habitats=True))
    # Apply some quality controls to the data. 
    df = filter_data(df)

    # Calculate the aerobe-anaerobe-facultative fractions for each sample.
    classes = ['aerobe', 'anaerobe', 'facultative'] # Assuming ternary classification. 

    aerobe_anaerobe_fraction_df = []
    # Iterate through the groups of habitats. Get the fraction of each organism time in the habitat.
    for habitat, habitat_df in df.groupby('habitat'):
        counts = habitat_df.prediction.value_counts()
        row = counts.astype(float) / counts.sum()
        row = row.to_dict()
        row = {f'{k}_fraction':v for k, v in row.items()}
        row.update(counts.astype(float).to_dict())
        row['total'] = counts.sum()
        row['habitat'] = habitat
        aerobe_anaerobe_fraction_df.append(row)

    aerobe_anaerobe_fraction_df = pd.DataFrame(aerobe_anaerobe_fraction_df).fillna(0)
    aerobe_anaerobe_fraction_df['aerobe_anaerobe_ratio'] = aerobe_anaerobe_fraction_df['aerobe_fraction'] / aerobe_anaerobe_fraction_df['anaerobe_fraction']
    
    return aerobe_anaerobe_fraction_df


if __name__ == '__main__':

    # Load in the genome metadata, which includes taxonomy.  
    metadata_df = pd.read_csv(os.path.join(DATA_PATH, 'metadata.tsv'), delimiter='\t', index_col=0, dtype={'metagenome_id':str})
    predictions_df = pd.read_csv(os.path.join(RESULTS_PATH, 'earth_microbiome_predict_nonlinear_aa_3mer_ternary.csv'), index_col=0)

    aerobe_anaerobe_fraction_df =  get_aerobe_anaerobe_fraction_df(predictions_df, metadata_df)
    print('Writing aerobe-anaerobe fraction data to', os.path.join(DATA_PATH, 'aerobe_anaerobe_fraction.csv'))
    aerobe_anaerobe_fraction_df.to_csv(os.path.join(DATA_PATH, 'aerobe_anaerobe_fraction.csv'))

    taxonomy_coverage_df = get_taxonomy_coverage_df(metadata_df)
    print('Writing taxonomy converage data to', os.path.join(DATA_PATH, 'taxonomy_coverage.csv'))
    taxonomy_coverage_df.to_csv(os.path.join(DATA_PATH, 'taxonomy_coverage.csv'))