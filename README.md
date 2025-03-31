# OGTFinder: optimal growth temperature prediction for prokaryotes

## Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)  
   3.1 [Input Data](#input_data)  
   3.2 [Output Data](#output_data)  
   3.3 [Usage Command](#usage_command)  
   3.4 [Examples](#examples)
4. [License](#license)
5. [Contact](#contact)


## Introduction <a name="introduction"></a>
OGTFinder is a tool that predicts the optimal growth temperature (OGT) for a user-provided (partial) proteome based on mean amino acid descriptors and taxonomic domain. For more details, please refer to:

Colette, S., François, J., De Moor, B., & Van Noort, V. (2025, March). *Machine learning for optimal growth temperature prediction of prokaryotes using amino acid descriptors*. doi:10.1101/2025.03.03.640802

## Installation <a name="installation"></a>
System Requirements: Python (>= 3.9)

### Installation with pip
```
pip install OGTFinder
```

## Usage <a name="usage"></a>

### Input Data <a name="input_data"></a>
The input is a FASTA file containing proteome sequences of the organism of interest. Comment lines starting with '!', '#', or ';' are allowed. Besides the proteome, the user also needs to specify the taxonomic domain of the organism of interest, either Archaea or Bacteria (default) with `--domain`.

### Output Data <a name="output_data"></a>
The prediction is detailed in the file `results.tsv` in the output directory, which may be specified with `--outdir`. The result file is a tab-delimited file consisting of the following 4 columns: 
- `filename`: Input FASTA file
- `domain`: User-specified taxonomic domain
- `prediction [°C]`: OGT prediction
- `class`: Corresponding thermophilicity class

In addition, Debug mode `--debug` outputs a file `descriptors.tsv` containing the feature values inputted to the model. 

### Usage command <a name="usage_command"></a>
```
ogtfinder tests/test_data/Haloarcula_marismortui/GCF_000011085.1.faa --outdir my_outdir --domain Archaea
```

### Examples <a name="examples"></a>
*Example 1: Fervidobacterium pennivorans*

Fervidobacterium pennivorans is an anaerobe, thermophilic bacterium from the phylum Thermotogota. Its optimal growth temperature is 70°C according to the database ThermoBase and the supplementary information from Lyubetsky et al. (2020), and 80°C according to the database Tempura. In this work, we consider the median OGT value as the true OGT, which in this case is 70°C. Starting from the genome `GCF_000235405.2`, the proteome can be predicted with a genome annotation tool such as Prokka or Bakta. Alternatively, the proteome can be downloaded from NCBI or UniProt if available. In this work, we used Prokka to annotate all genomes with the following command:
```
prokka --outdir Prokka --force --prefix GCF_000235405.2 --genus Fervidobacterium --usegenus --kingdom Bacteria --evalue 0.001  GCF_000235405.2.fna
```

The resulting output directory `Prokka` contains the predicted proteome in the output file `GCF_000235405.2.faa`. This output file is also provided here under the `tests` directory.

Now, the OGT can be predicted with the following command:
```
ogtfinder tests/test_data/Fervidobacterium_pennivorans/GCF_000235405.2.faa --outdir my_outdir --domain Bacteria
```

The predicted OGT is both printed to the screen and stored in `my_outdir/results.tsv`. The predicted OGT is 71.3°C, close to the median OGT of 70°C.
