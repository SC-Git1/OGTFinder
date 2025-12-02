"""
This script contains code modified from GitHub repository: https://github.com/DavidBSauer/OGT_prediction
"""


from sys import argv
from ast import literal_eval
import logging
logger = logging.getLogger('feature_calculation')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('feature_calculation.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import csv
import multiprocessing as mp
import random
import sauer_scripts.genomic as genomic
import sauer_scripts.protein as protein
import sauer_scripts.ORFs as ORFs
import sauer_scripts.external_tools as external_tools
import operator
import os
import pandas as pd
from pathlib import Path
import sys
import os.path



def calc_features(species, genome_files):

  result = {}
  result['species']=species

  def features_per_genome(genome_file):

    result_genome = {"species": result["species"]}
    species = result_genome['species']

    # check if already exists:
    existing_files = {'genomic':False, 'ORF': False, "protein": False}
    for feature_class in ['genomic','ORF','protein']:
        if Path('./sauer_features/'+species+'_'+feature_class+'.txt').exists():
            existing_files[feature_class] = True


    #calculate genomic features
    if not existing_files["genomic"]:
        logger.info('Working on '+genome_file)
        result_genome['genomic'] = genomic.analysis((genome_file,species))

    if not existing_files["protein"]:
        #calculate ORF and proteome features
        logger.info('Identifying and analyzing ORFs for '+genome_file)
        (ORF_test,ORF_seqs) = external_tools.ORF(genome_file)
        (protein_test,protein_seqs) = external_tools.proteins(genome_file)

        if ORF_test and not existing_files["ORF"]:
            t_size = result_genome['genomic']['Total Size']
            result_genome['ORF']=ORFs.analysis(ORF_seqs,t_size)

        if protein_test:
            result_genome['protein']=protein.analysis(protein_seqs)

    return (genome_file,result_genome)

  random.shuffle(genome_files)
  to_analyze = genome_files
  logger.info('Analyzing Genomic features')

  ##single thread for trouble shooting
  results = list(map(features_per_genome,to_analyze))
  results = {x[0]:x[1] for x in results}

  #write out the results by feature class
  for feature_class in ['genomic','ORF','protein']:
  # get parameters
    all_params = []
    for genome in results.keys():
        if feature_class in results[genome].keys():
            all_params = results[genome][feature_class].keys()
            break

    if feature_class in results[genome].keys():
        g= open('./sauer_features/'+species+'_'+feature_class+'.txt','w')
        g.write('Genome\tspecies\t'+'\t'.join([feature_class+' '+x for x in all_params])+'\n')
        for genome in results.keys():
            if feature_class in results[genome].keys():
                g.write(genome+'\t'+results[genome]['species']+'\t')
                for param in all_params:
                    g.write(str(results[genome][feature_class][param])+'\t')
                g.write('\n')
        g.close()

  logger.info('Exiting normally')


if __name__ == "__main__":

  # read in the species-genome pairs.
  df = pd.read_csv("species_genomes.tsv", sep = "\t", header = 0)

  # from string to list
  df["Genomes"] = df["Genomes"].apply(literal_eval)

  # path to genomes folder
  base_path = Path("/path/to/genomes")

  # for each taxid:
  for i in range(len(df)):
    species = str(df.iloc[i]["ncbiTaxID_new"])
    genomes = df.iloc[i]["Genomes"]
    
    # get the specific genome file paths
    genome_files = []
    for genome in genomes:
      numeric_id = genome.split("_")[1].split(".")[0]
      matches = list(base_path.glob(f"*_{numeric_id}*.fna"))
      if matches:
        genome_files.append(str(matches[0]))
      # If no genome file could be found, raise error
      else:
        print("Genome file missing!")
        sys.exit(1)

    # Calculate the genomic, ORF and protein features
    calc_features(species, genome_files)
