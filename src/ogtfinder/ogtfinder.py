#!/usr/bin/env python3

from .extractData import calc_mean_descriptors, ogt_predict
from .checks import check_fasta, check_dir, create_dir, get_proteome
import os
import sys, argparse
import logging
import traceback
from datetime import datetime
from importlib.metadata import version



def get_category(ogt: float) -> str:
    if ogt < 15:
        return "psycrophile"
    elif ogt >= 15 and ogt <= 45:
        return "mesophile"
    elif ogt > 45 and ogt <= 80:
        return "thermophile"
    else:
        return "hyperthermophile"


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help
        if action.default is False and not action.required:
            help_text += " (default: OFF)"
        elif action.default is not argparse.SUPPRESS and not action.required:
            if "default: " not in help_text:
                help_text += f" (default: {action.default})"
        return help_text


def main():

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                         add_help = False, description = "Optimal growth temperature prediction using proteome-derived features")

    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                    help="Show this help message and exit")
    parser.add_argument('infile', help="", metavar='<proteome.fasta>')
    parser.add_argument('--domain', choices=['Bacteria','Archaea'], default = "Bacteria", help = 'Taxonomic domain name')
    parser.add_argument("-o","--outdir", help="Output directory (default: [auto])", default = argparse.SUPPRESS)
    parser.add_argument('--force', action='store_true', default=False, help='Overwrite output directory')
    parser.add_argument('--verbose', type = int, choices = [0,1,2], default=2, help="Output verbosity. " +
                             "Levels: 0 (silent), 1 (warnings), 2 (verbose)")
    parser.add_argument('-v', '--version', action='version', version=version('ogtfinder'), help = "Show the version and exit")
    parser.add_argument('--debug', action = 'store_true', default = False, help = "Debug mode: also keep intermediate results")

    args = parser.parse_args()

    infile = args.infile
    domain = args.domain
    force = args.force
    verbose = args.verbose
    debug = args.debug

    # set logging level
    if args.debug:
        print("Debug logging activated")
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    elif verbose == 0:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    elif verbose == 1:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    elif verbose == 2:
        logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')


    if hasattr(args,"outdir"):
        outdir = args.outdir
    else:
        # auto create output directory
        outdir = os.path.join(os.getcwd(), f"OGTFINDER_{datetime.now().strftime('%Y%m%d')}")

    # If outdir does not exist yet
    if not check_dir(outdir):
        create_dir(outdir)
        logging.debug(f"Created directory {outdir}")

    elif check_dir(outdir) and not force:
        logging.error(f"Folder {outdir} already exists. Please change --outdir or use --force to overwrite. Exiting ...")
        sys.exit()

    # else: overwrite - i.e., do nothing, overwrite files 'w'.

    # check validity input file
    check_fasta(infile)

    logging.debug("Setup program completed.")
    logging.debug("Read in proteome...")

    # get sequences from input FASTA file
    sequences = get_proteome(infile)

    # calculate mean predictors
    logging.debug("Calculating features ...")
    dfvalues = calc_mean_descriptors(sequences, domain, outdir)

    # if debug, write intermediary results to descriptors.tsv file in output directory
    if debug:
        # write to file
        dfvalues.to_csv(os.path.join(outdir,"descriptors.tsv"), index = False, sep  = "\t")

    # predict ogt
    pred = ogt_predict(dfvalues, domain)
    logging.info(f"The predicted OGT is {pred:.1f}°C.")


    # write result
    with open(os.path.join(outdir, "results.tsv"), "w") as fout:
        fout.write("\t".join(["filename", "domain", "prediction [°C]", "class"]) + "\n")
        fout.write("\t".join([os.path.basename(infile), domain, f"{pred:.1f}", get_category(pred)]) + "\n")



if __name__ == "__main__":

    try:
         main()

    except Exception as e:
        logging.error(traceback.format_exc())
