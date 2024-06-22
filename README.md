# Annotation-free prediction of microbial di-oxygen utilization 

Predicting whether or not a microorganism is aerobic (i.e. requires oxygen to grow) or anaerobic from genomic data can be a daunting task, as many anaerobic organisms produce non-essential enzymes which use oxygen as a substrate. One approach to determining physiology is to annotate genome sequences, and using the predicted gene functions to build models for predicting aerobicity. However, genome annotation is computationally-intensive, and relies on existing knowledge of protein structure and functions. In this work, we assess the ability of simple, annotation-free genomic features (e.g. amino acid trimers) to predict microbial physiology. 

This repository contains the code associated with the following pre-print, which will soon be published in the mSystems journal. 

[Flamholz, A. I., Goldford, J. E., Larsson, E. M., Jinich, A., Fischer, W. W., & Newman, D. K. (2024). Annotation-free prediction of microbial dioxygen utilization. bioRxiv, 2024-01.](https://www.biorxiv.org/content/10.1101/2024.01.16.575888v1)

## Installation
To use the code in this repository, simply clone this GitHub repository and use the Python package manager to install the package into your coding environment (see the coding environments section below for how to set up your environment). 

```
git clone https://github.com/flamholz/annotation_free_dioxygen_utilization.git
pip install -e /path/to//annotation_free_dioxygen_utilization
```
## Coding environments

This project uses two separate coding environments, which are described by the aerobot.yml and aerobot-16s.yml files. The majority of the codebase runs within the aerobot environment. The aerobot-16s environment was created to resolve dependency issues with the code used to generate embeddings of 16S RNA sequences (see 16S RNA embeddings), which necessitates use of an older version of Python. The coding environments can be set up using conda, as shown below.

```
conda env create -f aerobot.yml
conda activate aerobot
```

## Data availability
Prior to running any of the scripts below, data must be downloaded and stored in a data folder in the root directory of the repository. The expected file structure is given below. All data can be downloaded from [FigShare](https://figshare.com/articles/dataset/Annotation-free_prediction_of_microbial_dioxygen_utilization/26065345).

```
├── aerobot
│   ├── features
├── data
│   ├── black_sea
│   ├── contigs
│   │   └── genomes
│   ├── earth_microbiome
│   ├── features
│   ├── original
│   └── rna16s
├── figures
├── models
├── notebooks
├── results
├── scripts
└── tests
```


