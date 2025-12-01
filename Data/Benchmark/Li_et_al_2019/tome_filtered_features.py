#!/usr/bin/env python3

"""
This script contains (modified) functions of the Tome package (source: https://github.com/EngqvistLab/Tome)
"""

# packages
import os
from collections import Counter
import numpy as np
from Bio import SeqIO
import pandas as pd


# Test if the proteome contains more than 10^5 amino acids
def test_length(fasta_file):
    seqs = [str(rec.seq).upper() for rec in SeqIO.parse(fasta_file,'fasta')]
    total_len = np.sum([len(seq) for seq in seqs])
    print(total_len)
    if total_len >= 100000:
        return True
    else:
        return False

# Function from Tome package
def do_count(seq):
    dimers = Counter()
    for i in range(len(seq)-1): dimers[seq[i:i+2]] += 1.0
    return dimers

# Function modified from Tome package
def count_dimer(fasta_file):
    seqs = [str(rec.seq).upper() for rec in SeqIO.parse(fasta_file,'fasta')]

    results = list(map(do_count, seqs))
    dimers = sum(results, Counter())
    return dict(dimers)

# Function from Tome package
def get_dimer_frequency(fasta_file):
    dimers = count_dimer(fasta_file)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    dimers_fq = dict()

    # this is to remove dimers which contains letters other than these 20 amino_acids,
    # like *
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimers_fq[a1+a2] = dimers.get(a1+a2,0.0)
    number_of_aa_in_fasta = sum(dimers_fq.values())
    for key,value in dimers_fq.items(): dimers_fq[key] = value/number_of_aa_in_fasta
    return dimers_fq

# If the proteome contains more than 10^5 amino acids, calculate and save the dipeptide frequencies 
def get_features(fasta_file, outfile):
    res = test_length(fasta_file)
    if res == True:
        dimers_fq = get_dimer_frequency(fasta_file)

        Xs = list()
        features = ['AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL', 'AM', 'AN', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AV', 'AW', 'AY', 'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH',
                'CI', 'CK', 'CL', 'CM', 'CN', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CV', 'CW', 'CY', 'DA', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DK', 'DL', 'DM', 'DN', 'DP', 'DQ',
                'DR', 'DS', 'DT', 'DV', 'DW', 'DY', 'EA', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EK', 'EL', 'EM', 'EN', 'EP', 'EQ', 'ER', 'ES', 'ET', 'EV', 'EW', 'EY', 'FA',
                'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FK', 'FL', 'FM', 'FN', 'FP', 'FQ', 'FR', 'FS', 'FT', 'FV', 'FW', 'FY', 'GA', 'GC', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI',
                'GK', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GV', 'GW', 'GY', 'HA', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HK', 'HL', 'HM', 'HN', 'HP', 'HQ', 'HR',
                'HS', 'HT', 'HV', 'HW', 'HY', 'IA', 'IC', 'ID', 'IE', 'IF', 'IG', 'IH', 'II', 'IK', 'IL', 'IM', 'IN', 'IP', 'IQ', 'IR', 'IS', 'IT', 'IV', 'IW', 'IY', 'KA', 'KC',
                'KD', 'KE', 'KF', 'KG', 'KH', 'KI', 'KK', 'KL', 'KM', 'KN', 'KP', 'KQ', 'KR', 'KS', 'KT', 'KV', 'KW', 'KY', 'LA', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LK',
                'LL', 'LM', 'LN', 'LP', 'LQ', 'LR', 'LS', 'LT', 'LV', 'LW', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI', 'MK', 'ML', 'MM', 'MN', 'MP', 'MQ', 'MR', 'MS',
                'MT', 'MV', 'MW', 'MY', 'NA', 'NC', 'ND', 'NE', 'NF', 'NG', 'NH', 'NI', 'NK', 'NL', 'NM', 'NN', 'NP', 'NQ', 'NR', 'NS', 'NT', 'NV', 'NW', 'NY', 'PA', 'PC', 'PD',
                'PE', 'PF', 'PG', 'PH', 'PI', 'PK', 'PL', 'PM', 'PN', 'PP', 'PQ', 'PR', 'PS', 'PT', 'PV', 'PW', 'PY', 'QA', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QK', 'QL',
                'QM', 'QN', 'QP', 'QQ', 'QR', 'QS', 'QT', 'QV', 'QW', 'QY', 'RA', 'RC', 'RD', 'RE', 'RF', 'RG', 'RH', 'RI', 'RK', 'RL', 'RM', 'RN', 'RP', 'RQ', 'RR', 'RS', 'RT',
                'RV', 'RW', 'RY', 'SA', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SK', 'SL', 'SM', 'SN', 'SP', 'SQ', 'SR', 'SS', 'ST', 'SV', 'SW', 'SY', 'TA', 'TC', 'TD', 'TE',
                'TF', 'TG', 'TH', 'TI', 'TK', 'TL', 'TM', 'TN', 'TP', 'TQ', 'TR', 'TS', 'TT', 'TV', 'TW', 'TY', 'VA', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI', 'VK', 'VL', 'VM',
                'VN', 'VP', 'VQ', 'VR', 'VS', 'VT', 'VV', 'VW', 'VY', 'WA', 'WC', 'WD', 'WE', 'WF', 'WG', 'WH', 'WI', 'WK', 'WL', 'WM', 'WN', 'WP', 'WQ', 'WR', 'WS', 'WT', 'WV',
                'WW', 'WY', 'YA', 'YC', 'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YK', 'YL', 'YM', 'YN', 'YP', 'YQ', 'YR', 'YS', 'YT', 'YV', 'YW', 'YY']

        for fea in features:
            Xs.append(dimers_fq[fea])

        Xs = np.array(Xs).reshape([1,len(Xs)])

        # write to file
        with open(outfile, "a") as f:
            f.write(str(os.path.basename(fasta_file)).replace(".faa", "") + "," + ",".join(map(str, Xs[0])) + "\n")

if __name__ == "__main__":
    # output
    genomefile = "mean_descriptors.csv" # from ogt_final_runs/data/raw
    outputfile = "genome_tome_descriptors.csv"

    # read in genomes:
    df = pd.read_csv(genomefile, header = None)
    Genomes = df.iloc[:,0].to_list()

    dir = "/path/to/proteomes"

    for file in os.listdir(dir):
        if file.endswith(".faa"):
            if file.replace('.faa','') in Genomes:
                pass

            else:
                get_features(os.path.join(dir, file), outputfile)



