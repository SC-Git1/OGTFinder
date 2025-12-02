import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import shutil
import numpy as np
import subprocess
import logging
logger = logging.getLogger('feature_calculation')
from BCBio import GFF
import gzip
import tarfile


#load external tool commands
commands = {}
f = open('./sauer_scripts/external_tools.txt','r')
for line in f.readlines():
  commands[line.split()[0].strip()]=line.split()[1].strip()
f.close()

#record version
def versions():
  global commands
  p = subprocess.Popen([commands['prodigal']+' -v'],shell=True,executable='/bin/bash',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = p.communicate()
  logger.info('prodigal version info: '+err.decode('utf-8').strip())
  logger.info('Numpy version: '+np.__version__)
  import Bio
  logger.info('Biopython version: '+Bio.__version__)
  del(Bio)
  import sys
  logger.info('Python version: '+sys.version)
  del(sys)
  import platform
  logger.info('Platform: '+platform.platform())
  del(platform)

versions()

#using prodigal ORFfinder
def ORF(genome_file):
  genome = os.path.basename(genome_file).replace('.fna','')
  global commands

  if not os.path.exists('./sauer_proteome/'+genome+'.faa'):
    command = (f"{commands['prodigal']} -i {genome_file} -d ./sauer_mrna/{genome}.fna -a ./sauer_proteome/{genome}.faa -o ./sauer_prodigal/{genome}.gbk")

    p = subprocess.Popen([command],shell=True,executable='/bin/bash',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = p.communicate()
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    if not('Error:' in err): #prodigal uses stderr for all output, need to catch errors
      if os.path.isfile('./sauer_mrna/'+genome+'.fna'):
        if len(SeqIO.index('./sauer_mrna/'+genome+'.fna','fasta'))>0:
          return (True,SeqIO.index('./sauer_mrna/'+genome+'.fna','fasta'))
        else:
          logger.info('Predicted zero ORFs for '+genome_file)
          return (False,None)
      else:
        logger.info('could not find predicted ORFs for '+genome_file)
        return (False,None)
    else:
      logger.info('error on '+genome_file+' prodigal step with a message of\n'+err)
      return (False,None)

  else:
    if os.path.isfile('./sauer_mrna/'+genome+'.fna'):
        if len(SeqIO.index('./sauer_mrna/'+genome+'.fna','fasta'))>0:
          return (True,SeqIO.index('./sauer_mrna/'+genome+'.fna','fasta'))

def proteins(genome_file):
  genome = os.path.basename(genome_file).replace('.fna','')
  if os.path.isfile('./sauer_proteome/'+genome+'.faa'):
    if len(SeqIO.index('./sauer_proteome/'+genome+'.faa','fasta'))>0:
      return (True,SeqIO.index('./sauer_proteome/'+genome+'.faa','fasta'))
    else:
      print('Predicted zero proteins for '+genome_file)
      logger.info('Predicted zero proteins for '+genome_file)
      return (False,None)
  else:
    print('could not find predicted proteins for '+genome_file)
    logger.info('could not find predicted proteins for '+genome_file)
    return (False,None)
