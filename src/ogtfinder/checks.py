from Bio import SeqIO
from pathlib import Path
import logging
import sys
import os

def create_dir(dir: Path):
    Path(dir).mkdir(parents=True, exist_ok=True)


def check_dir(outdir: Path) -> bool:
    # check if directory exists
    return os.path.isdir(outdir)


def get_proteome(infile: Path) -> list:
    """
    This function checks heuristically whether the provided FASTA file is protein.
    A FASTA file with in total >= 75% [ACGTNU] is considered dna/rna.
    It returns True if the checks pass, False otherwise.

    Parameters
    ----------
    fasta: Path
        Path to FASTA file containing query proteome.
    """

    fasta_sequences = SeqIO.parse(open(infile), 'fasta-blast')

    sequences = []

    for fasta in fasta_sequences:
        sequences.append(str(fasta.seq).upper())

    if len(sequences) == 0:
        logging.error(f"No sequences detected in the FASTA file: {filepath}. Exiting...")
        sys.exit()

    if not is_proteome(sequences):
        logging.warning(f"The FASTA file {infile} contains over 75% DNA/RNA-specific letters (ACGTNU). " +
                             "Please verify that the input consists of protein sequences.")

    return sequences


def is_proteome(seqs: list) -> bool:
    allseq = ''.join(seqs)
    if len([letter for letter in allseq if letter in "ACGTNU"]) < 0.75*len(allseq):
        return True
    else:
        return False


def is_valid_fasta_file(fasta: Path) -> bool:
    """
    This function checks whether the provided input FASTA file is a valid FASTA file.
    It returns True if the checks pass, False otherwise.

    Parameters
    ----------
    fasta: Path
        Path to FASTA file containing query proteome.

    """
    try:
        SeqIO.parse(fasta, "fasta-blast")
        return True
    except Exception:
        return False


def check_fasta(infile):
    """
    This function checks whether the provided input FASTA file exists and is a valid FASTA file.
    If not, it throws an error.

    Parameters
    ----------
    fasta: Path
        Path to FASTA file containing genome.

    """
    if not os.path.exists(infile):
        logging.error(f"Input file {infile} does not exist. Please provide the correct path to the input file. Exiting ...")
        sys.exit()

    if not is_valid_fasta_file(infile):
        logging.error(f"Input file {infile} is not a valid FASTA file. Exiting ...")
        sys.exit()
