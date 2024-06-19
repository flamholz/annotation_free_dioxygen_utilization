import pandas as pd 
import numpy as np 
from aerobot.utils import AMINO_ACIDS, NUCLEOTIDES, FEATURES_PATH 
import os 


def load_chemistry(seq_type:str):
    if seq_type == 'cds':
        nt_df = pd.read_csv(os.path.join(FEATURES_PATH, 'nt_chemical_features.csv'))
        nt_df = nt_df[nt_df.type == 'RNA'].set_index('letter_code')
        # return nt_df # Includes DNA and RNA names
        return nt_df.loc[NUCLEOTIDES]
    elif seq_type == 'aa':
        aa_df = pd.read_csv(os.path.join(FEATURES_PATH, 'aa_chemical_features.csv'), index_col=0)
        return aa_df


def from_counts(counts_df:pd.DataFrame, seq_type:str=None) -> pd.Series:
    '''Calculate chemical features of the RNA coding sequences. Calculates the formal 
    C oxidation state of mRNAs, as well as the C, N, O and S content. 

    :param counts_df: A DataFrame containing the counts of single nucleotides, either RNA or DNA. Assumed to single stranded.
    :return: A pd.Series containing the formal C oxidation state of the RNA coding sequences.
    '''

    chemistry_df = load_chemistry(seq_type=seq_type).loc[counts_df.columns]

    num_carbons = counts_df @ chemistry_df.num_carbon
    ox_state_of_carbon = counts_df @ (chemistry_df.num_carbon * chemistry_df.ox_state_carbon)
    mean_ox_state_of_carbon = ox_state_of_carbon / num_carbons
    mean_ox_state_of_carbon.name = f'{seq_type}_mean_ox_state_carbon'

    columns = []

    elements = ['carbon', 'oxygen', 'nitrogen']
    elements = elements + ['sulfur'] if seq_type == 'aa' else elements
    for element in elements:
        totals = counts_df @ chemistry_df[f'num_{element}']
        means = totals / counts_df.sum(axis=1)
        means.name = f'{seq_type}_mean_num_{element}'
        columns.append(means)

    return pd.concat(columns, axis=1)


def from_features(number_of_genes_df:pd.DataFrame=None, cds_1mer_df:pd.DataFrame=None, aa_1mer_df:pd.DataFrame=None, nt_1mer_df=None) -> pd.DataFrame:
    '''Compute chemical features using other feature DataFrames and the metadata.

    :param metadata_df: DataFrame containing the gene metadata.
    :param nt_1mer_df: DataFrame containing the nt_1mer feature data.
    :param aa_1mer_df: DataFrame containing the aa_1mer feature data.
    :param cds_1mer_df: DataFrame containing the cds_1mer feature data.
    :return: A DataFrame containing the chemical feature data.
    '''
    n_genes = number_of_genes_df['number_of_genes']

    gc_content = nt_1mer_df[['G', 'C']].sum(axis=1) / nt_1mer_df.sum(axis=1)
    gc_content.name = 'gc_content'

    aa_features = from_counts(aa_1mer_df[[c for c in AMINO_ACIDS if c in aa_1mer_df.columns]], seq_type='aa')
    rna_features = from_counts(cds_1mer_df[[c for c in NUCLEOTIDES if c in cds_1mer_df.columns]], seq_type='cds')

    return pd.concat([gc_content, n_genes, aa_features, rna_features], axis=1) #.dropna(axis=0)
