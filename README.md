<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 4
* WARNINGs: 0
* ALERTS: 4

Conversion time: 0.724 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β36
* Tue Jun 25 2024 09:40:21 GMT-0700 (PDT)
* Source doc: methodology_v3

ERROR:
undefined internal link to this URL: "#heading=h.t506upm5rpw1".link text: coding environments
?Did you generate a TOC with blue links?

* Tables are currently converted to HTML tables.

ERROR:
undefined internal link to this URL: "#heading=h.cg9tlhhvhl9b".link text: 16S RNA embeddings
?Did you generate a TOC with blue links?


ERROR:
undefined internal link to this URL: "#heading=h.cg9tlhhvhl9b".link text: 16S RNA embeddings
?Did you generate a TOC with blue links?


ERROR:
undefined internal link to this URL: "#heading=h.t1fibvu5kj3o".link text: Building the datasets
?Did you generate a TOC with blue links?


WARNING:
You have 8 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 4; WARNINGs: 1; ALERTS: 4.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



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

This project uses two separate coding environments, which are described by the `aerobot.yml `and `aerobot-16s.yml` files. The majority of the codebase runs within the aerobot environment. The aerobot-16s environment was created to resolve dependency issues with the code used to generate embeddings of 16S RNA sequences (see 16S RNA embeddings section), which necessitates use of an older version of Python. The coding environments can be set up using conda, as shown below.


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



## Feature types


<table>
  <tr>
   <td><strong>feature type</strong>
   </td>
   <td><strong>symbol</strong>
   </td>
   <td><strong>description</strong>
   </td>
  </tr>
  <tr>
   <td>Gene families
   </td>
   <td><code>ko</code>
   </td>
   <td>A ~10000-dimensional vector of integers, where each integer represents the number of hits for a particular KEGG gene family in the genome. 
   </td>
  </tr>
  <tr>
   <td>Terminal oxidase gene families
   </td>
   <td><code>ko_terminal_oxidase_genes</code>
   </td>
   <td>The same as the gene families feature set, but only counts for a select group terminal oxidase gene families were included (see <code>/data/features/terminal_oxidase_genes.csv</code> for the complete list).
   </td>
  </tr>
  <tr>
   <td>Genome embedding
   </td>
   <td><code>embedding_genome</code>
   </td>
   <td>Numerical embeddings of the genomes created by passing each protein through a <a href="https://github.com/agemagician/ProtTrans">large language model</a>, and averaging each protein embedding to obtain a representation of the entire genome. 
   </td>
  </tr>
  <tr>
   <td>Oxygen gene embeddings
   </td>
   <td><code>embeddings_oxygen_genes</code>
   </td>
   <td>The same as the genome embedding feature set, but only proteins associated with oxygen metabolism were included.
   </td>
  </tr>
  <tr>
   <td>Chemical features
   </td>
   <td><code>chemical</code>
   </td>
   <td>A vector of values which describe the chemical characteristics of a genome’s nucleotides and amino acids. Chemical features included in this feature set are: genomic GC content, number of genes, the mean number of carbon, oxygen, nitrogen, and sulfur atoms on amino acids, and the mean number of carbon, oxygen, and nitrogen atoms on coding-sequence nucleotides.
   </td>
  </tr>
  <tr>
   <td>Number of oxygen genes
   </td>
   <td><code>number_of_oxygen_genes</code>
   </td>
   <td>A single integer indicating the total number of oxygen-associated genes in the genome. 
   </td>
  </tr>
  <tr>
   <td>Number of genes
   </td>
   <td><code>number_of_genes</code>
   </td>
   <td>A single integer indicating the total number of genes in the genome.
   </td>
  </tr>
  <tr>
   <td>Percent oxygen genes
   </td>
   <td><code>percent_oxygen_genes</code>
   </td>
   <td>A float representing the fraction of genes in the genome which are associated with oxygen metabolism
   </td>
  </tr>
  <tr>
   <td>16S RNA embedding
   </td>
   <td><code>embedding_rna16s</code>
   </td>
   <td>Embeddings of each genome’s 16S RNA sequences, generated using a large-language model (see the 16S RNA embeddings</a> section for more details).
   </td>
  </tr>
  <tr>
   <td>Nucleotide k-mers
   </td>
   <td><code>nt_1mer, nt_2mer, nt_3mer, nt_4mer, nt_5mer</code>
   </td>
   <td>A vector of floats representing the normalized count of each nucleotide k-mer in the genome sequence, including both coding and non-coding regions.
   </td>
  </tr>
  <tr>
   <td>Amino acid k-mers
   </td>
   <td><code>aa_1mer, aa_2mer, aa_3mer</code>
   </td>
   <td>A vector of floats representing the normalized count of each amino acid k-mer in a genome’s proteins.
   </td>
  </tr>
  <tr>
   <td>Coding sequence k-mers
   </td>
   <td><code>cds_1mer, cds_2mer, cds_3mer, cds_4mer, cds_5mer</code>
   </td>
   <td>A vector of floats representing the normalized count of each nucleotide k-mer in the genome sequence, only accounting for nucleotides in the coding regions.
   </td>
  </tr>
</table>



## Building the datasets

The MAGs and physiology labels used in this project were obtained from __. Genomic features were extracted from Jablonska and Madin MAGs and stored in HDF files, and the two data sources were merged into a single dataset, and duplicate genomes were removed. Additionally, several of the genomes present in the datasets have been “suppressed” by NCBI due to a failure to meet [quality control criteria.](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/policies-annotation/genome-processing/genome_notes/)  A list of these suppressed genomes was obtained from the NCBI FTP site, and suppressed genomes were removed from the merged dataset.

The merged dataset was then divided into three disjoint partitions for model training, validation, and testing. Each dataset was created to be phylogenetically-representative at the class level, with the aim of testing the models’ abilities to learn phylogenetic information contained in the genome features. This was done by sorting the genomes by class, and withholding 20 percent of each class for the testing dataset. The remaining genomes were partitioned a second time into the training and validation datasets by repeating this procedure, with 20 percent of the remaining genomes in each class being withheld for the validation dataset. The resulting training, validation, and testing datasets contained 2084, 465, and 587 genomes, respectively. 

The training, testing, and validation datasets can be generated from scratch from the `madin_datasets.h5 `and` jablonska_datasets.h5 `files by running the script below.


```
python build.py
```



## Model training


### Logistic regression classifiers

The logistic regression-based classifiers use Python’s Scikit-Learn implementation of logistic regression. The `max_iter` parameter was set to 10000, which was the maximum number of iterations required for the regressors to converge on all feature type data. The `C` parameter, which specifies regularization strength, was set to 100. The trained model was then used to predict the physiologies of organisms in the testing dataset, and the balanced accuracy and confusion matrix was computed for this prediction. 

Logistic regression classifiers were trained using the code snippet below, where `n_classes` specifies binary or ternary classification.


```
python train.py logistic [feature-type] --n-classes=n
```



### Nonlinear classifiers

The nonlinear classifiers are simple neural networks, consisting of an input layer, ReLU activation function, a 512-dimensional hidden layer, an output layer, and a softmax layer. Weighted mean-squared error was used for the loss function, with the weights for each class set to be the inverse frequency of each class in the training dataset. An Adam optimizer was used for model training, with the learning rate parameter set to 0.0001 and the weight decay was set to 0.01 – weight decay sets the inverse regularization strength, which is equivalent to the regularization strength used for the logistic regression classifiers. 

Each model was trained for a maximum of 100 epochs on the training dataset, using batches of size 16. After each epoch, model performance was evaluated on the validation dataset. At the end of the 100 epochs, the model weights which resulted in the highest accuracy on the validation data were saved, effectively implementing early stopping of the training process. The trained model was then used to predict the physiologies of organisms in the testing dataset, and the balanced accuracy and confusion matrix was computed for this prediction. 

Logistic regression classifiers were trained using the code snippet below, where `n_classes` specifies binary or ternary classification.


```
python train.py nonlinear [feature-type] --n-classes=n
```



## Model evaluation


### Phylogenetic cross-validation

Because models were trained on class-balanced datasets, the results of model testing were indicative of the models’ abilities to learn phylogenetic information. However, we also sought to assess how well the models were able to generalize to unknown phylogenetic groups. To do this, we implemented “phylogenetic cross-validation,” using a procedure similar to that described in [this preprint](https://www.biorxiv.org/content/10.1101/2024.03.22.586313v1.full.pdf). 

We first combined the disjoint training and validation datasets into a single training set. Using Scikit-Learn’s GroupShuffleSplit interface, we randomly split the merged data into training and testing partitions such that no phylogenetically-related taxa were present in both partitions. This was performed for relatives at each taxonomic level, e.g. Phylum, Class, Family, etc. For example, take a split at the Class level The merged dataset contained multiple genomes from the family Chlorobia. When splitting the merged dataset, we ensured that _all _members of Chlorobia were present in either the testing _or_ training partitions, i.e. if any Chlorobia taxon was present in the training partition, none were present in the testing partition. 

For each taxonomic level, the merged dataset was randomly split 5 times, with ~20 percent of the merged data being withheld for the testing set. A model was then trained from scratch on each training partition. For the nonlinear classifiers, which require an additional validation set for the early-stopping protocol, the training data was further divided into a training and validation dataset. After completion of model training, we computed balanced accuracy on the holdout testing set as an evaluation metric. 

Phylogenetic cross-validation can be performed using the code snippet below. 


```
python phylo-cv.py [model-class] [feature-type] --n-splits=5 --n-classes=n
```



#### Random relative baseline

In addition to the nonlinear and logistic regression-based classifiers, we created a baseline classifier which makes physiological predictions using only phylogenetic information. This model takes a genome and its associated phylogenetic information and locates all relatives of the organism at the specified taxonomic level. At the Class level, for example, if the model receives a genome of an organism in the class Chlorobia, it will first find all members of Chlorobia in the merged dataset. It then selects one of these relatives at random, and uses the physiology label of the selected organism to predict the physiology of the input genome.


```
python phylo-cv.py randrel none --n-splits=5 --n-classes=n
```



### Earth Microbiome Project MAGs

We applied the nonlinear model trained on amino acid trimers to MAGs collected from diverse environments through the Earth Microbiome Project (EMP). We first extracted the amino acid trimer features from EMP genomes, which are stored under `data/earth_microbiome/aa_3mer.csv`. The code snippet below generates the physiology predictions.


```
#!/bin/bash

model=path/to/models/nonlinear_aa_3mer_ternary.joblib
input=path/to/data/earth_microbiome/aa_3mer.csv
output=/path/to/results/output.csv

python predict.py - m $model -i $input -o $output -f aa_3mer -t csv
```



#### Post-processing

Prior to analyzing the prediction results, a series of filtering steps were applied to the EMP data to control for sample and genome quality. The initial dataset contained __ samples and __ genomes. First, __ all genomes with less than 50 percent completeness were removed, leaving the high-quality genomes. Genomes assembled from samples, i.e. metagenomes, with fewer than 10 high-quality MAGs were also discarded. Finally, __ genomes from habitats for which fewer than 10 high-quality samples  had been collected were discarded.


### Black Sea MAGs


```
#!/bin/bash

model=path/to/models/nonlinear_aa_3mer_ternary.joblib
input=path/to/data/black_sea/aa_3mer.csv
output=/path/to/results/output.csv

python predict.py -m $model -i $input -o $output -f aa_3mer -t csv
```



#### Post-processing


### Synthetic contigs

We sought to determine whether or not annotation-free genomic features could be used to predict microbial physiology from raw contig data, which would preclude the need to bin and assemble MAGs – a process which discards a large amount of genetic data. 


#### Building the datasets

We first constructed a dataset of synthetic contigs by dividing 100 randomly-selected MAGs from the testing dataset into contigs of sizes ranging from 2000 to 50000 base pairs. 1000 contigs were sampled from random locations in each genome. We then extracted nucleotide 3-mer, 4-mer, and 5-mer features from each contig, to which we applied the corresponding nonlinear models for ternary classification.

The synthetic contig datasets were generated using the script below. **In order to run this script in its entirety, the NCBI datasets tool must be installed, which is used to obtain the complete nucleotide sequences of the genomes from NCBI. These genomes we used are also available on FigShare. ** 


```
python build-contigs.py --n-genomes=100 --max-n-contigs=100
```



### 16S RNA embeddings

In order to assess the predictive power of phylogeny, we build an additional classifier based on embeddings of 16S sequences. Embeddings were generated using a [pre-trained large language model](https://github.com/ramanathanlab/genslm?tab=readme-ov-file) (LLM), specifically the `genslm_25M_patric` architecture. **Note that running the embedding code requires the <code>aerobot-16s</code> conda environment.</strong>


#### Building the datasets

16S sequences were not available for all genomes in the Madin and Jablonska source datasets, so the training, testing, and validation sets for the 16S classifier were generated from scratch. First, we downloaded all available RNA sequences (1031)  from NCBI using the BioPython Entrez wrapper, as well as the NCBI taxonomy associated with each. We then partitioned the sequences into training, testing, and validation datasets using the same Class-balanced approach described in the *Building the datasets* section. The results of this partitioning were testing, training, and validation datasets of 188, 693, and 150 sequences respectively. 

Each RNA sequence in the datasets was then passed into the LLM, producing an embedding vector of n rows and 512 columns, where n is the length of the RNA sequence. To standardize the size of the embeddings, each was mean-pooled over sequence length. This resulted in 512-dimensional vectors, where each vector element is the average of the column vector in the original embedding. These embeddings were saved in HDF files in the `rna16s` subdirectory.


#### Model training

A simple linear model was used for classifying genomes using the 16S sequences, consisting of only a single linear layer followed by a softmax activation function. We used an Adam optimizer with a weight decay of 0.01 and learning rate of 0.01. The model was trained for a maximum of 100 epochs on the training data, using batches of size 16. As with the multi-layer models, the validation dataset was used to implement early stopping  during the training process.


## Aerobot tool

We are in the process of adapting this annotation-free approach for predicting microbial physiology into a command-line tool called aerobot. We hope that this tool will be complete in the next couple of months, and will update this repository with a link to the tool when it is available.
