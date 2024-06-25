import wget
import os
import gzip 
from aerobot.utils import DATA_PATH
import zipfile

URLS = dict()
URLS['suppressed_genomes.csv'] = 'https://figshare.com/ndownloader/files/47142253'
URLS['rna16s.zip'] = 'https://figshare.com/ndownloader/files/47142751'
URLS['black_sea.zip'] = 'https://figshare.com/ndownloader/files/47186161'
URLS['contigs.zip'] = 'https://figshare.com/ndownloader/files/47186182'
URLS['earth_microbiome.zip'] = 'https://figshare.com/ndownloader/files/47186173'
URLS['features.zip'] = 'https://figshare.com/ndownloader/files/47186155'
URLS['jablonska_datasets.h5'] = 'https://figshare.com/ndownloader/files/47186158'
URLS['madin_datasets.h5'] = 'https://figshare.com/ndownloader/files/47186188'
URLS['training_datasets.h5'] = 'https://figshare.com/ndownloader/files/47186179'
URLS['testing_datasets.h5'] = 'https://figshare.com/ndownloader/files/47186185'
URLS['validation_datasets.h5'] = 'https://figshare.com/ndownloader/files/47186176'


def extract(name:str):
    # Check to see if the file exists before extracting.
    if not os.path.exists(os.path.join(DATA_PATH, name.replace('.zip', ''))):
        with zipfile.ZipFile(os.path.join(DATA_PATH, name), 'r') as zf:
            # name = name.replace('.zip', '') # Remove the zip file extension.
            print(f'\nextract: Extracting {name} to {DATA_PATH}')
            zf.extractall(DATA_PATH)

if __name__ == '__main__':

    # Make the data directory in the project root directory.
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    for name, url in URLS.items():
        if name not in os.listdir(DATA_PATH):
            print(f'\nDownloading {name}...')
            wget.download(url, DATA_PATH)
        if 'zip' in name:
            extract(name)


    

    