#!/bin/bash

#SBATCH --job-name=embed_a100_esm2_t33_650m
#SBATCH --output=src/slurm/logfiles/embed_a100_esm2_t33_650m.out
#SBATCH --error=src/slurm/logfiles/embed_a100_esm2_t33_650m.err
#SBATCH --account=lp_cosubi
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=9
#SBATCH --mem-per-cpu=10000
#SBATCH --time=3-00:00:00

# Make sure log directory exists
mkdir -p src/slurm/logfiles
mkdir -p src/out

# Load conda and activate env
source /vsc-hard-mounts/leuven-data/331/vsc33189/env_setup/conda_base/etc/profile.d/conda.sh
conda activate ml_env

# Go to your benchmark folder
cd /staging/leuven/stg_00142/projects/ogt-paper-dev/benchmark_llm

python src/embed_benchmark.py \
  --model_name esm2_t33_650M_UR50D \
  --hardware a100 \
  --num_gpus $SLURM_GPUS_PER_NODE \
  --batch_size 1 \
  --proteome_dir ./benchmark_proteomes \
  --output_csv src/out/timings.csv
