#!/bin/bash

#SBATCH --job-name=embed_cpu_genius_esm2_t36_3b
#SBATCH --output=src/slurm/logfiles/embed_cpu_genius_esm2_t36_3b.out
#SBATCH --error=src/slurm/logfiles/embed_cpu_genius_esm2_t36_3b.err
#SBATCH --account=lp_cosubi
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=3-00:00:00

# Make sure log directory exists
mkdir -p src/slurm/logfiles
mkdir -p src/out

# Bind threads to the cores we requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load conda and activate env
source /vsc-hard-mounts/leuven-data/331/vsc33189/env_setup/conda_base/etc/profile.d/conda.sh
conda activate ml_env

# Go to your benchmark folder
cd /staging/leuven/stg_00142/projects/ogt-paper-dev/benchmark_llm

CPU_MODEL=$(lscpu | awk -F: '/Model name/ {print $2}' | sed 's/^ *//')

python src/embed_benchmark_cpu.py \
  --model_name esm2_t36_3B_UR50D \
  --cpu_model "$CPU_MODEL" \
  --num_cores $SLURM_CPUS_PER_TASK \
  --batch_size 1 \
  --proteome_dir ./benchmark_proteomes \
  --output_csv src/out/timings.csv
