import pandas as pd
import numpy as np 
from aerobot.io import RESULTS_PATH, DATA_PATH
import os


DATA_PATH = os.path.join(DATA_PATH, 'black_sea')

if __name__ == '__main__':
    # Load the black sea metadata
    metadata_df = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'), index_col=0)
    metadata_df['depth_m'] = metadata_df['Depth'].str.strip('m').astype(float)

    # Chemical measurements are from Sollai et al. 2019 Geobiology. In their Figure 1, it's clear these are in uM units 
    # despite what it says in our metadata file, which was downloaded from the SRA at this link:
    # https://www.ncbi.nlm.nih.gov/Traces/study/?query_key=3&WebEnv=MCID_6582027b68a595196d9af8e7&o=acc_s%3Aa
    # uM units make sense as the Henry's law equilibrium of O2 in seawater is ~200 uM.
    metadata_df['o2_uM'] = metadata_df['diss_oxygen'].str.strip('mmol / kg').astype(float)
    metadata_df['h2s_uM'] = metadata_df['hydrogen_sulfide'].str.strip('mM').astype(float)
    metadata_df = metadata_df.sort_values('depth_m')


    # Load the Black Sea MAG inference results. I am a little confused by what the format of this data is. What do the columns indicate?
    predictions_df = pd.read_csv(os.path.join(RESULTS_PATH, f'black_sea_predict_nonlinear_aa_3mer_ternary.csv'), index_col=0)
    mag_fraction_df = pd.read_csv(os.path.join(DATA_PATH, 'mag_fraction.csv'), index_col=0).drop('*')

    # What is this doing?
    mag_fraction_df['prediction'] = predictions_df.loc[mag_fraction_df.index]
    black_sea_df = mag_fraction_df.groupby('prediction').sum().T * 100

    black_sea_df['depth_m'] = metadata_df.loc[black_sea_df.index].depth_m
    black_sea_df['o2_uM'] = metadata_df.loc[black_sea_df.index].o2_uM
    black_sea_df['h2s_uM'] = metadata_df.loc[black_sea_df.index].h2s_uM
    black_sea_df = black_sea_df.sort_values('depth_m')

    print('Writing Black Sea data to', os.path.join(DATA_PATH, 'black_sea.csv'))
    black_sea_df.to_csv(os.path.join(DATA_PATH, 'black_sea.csv'))