#!/bin/bash
#SBATCH --job-name=emf_dl_transformer_attention
#SBATCH --output=logfiles/emf_dl_transformer_attention.out
#SBATCH --error=logfiles/emf_dl_transformer_attention.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaldert.francois@student.kuleuven.be
#SBATCH --account=lp_cosubi
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00

# Prevent OpenBLAS/MKL thread oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# Change to the working directory
cd /scratch/leuven/331/vsc33189/projects/ogtfinder/ensemble_models_fixed/implementation

# Set up conda path
export PATH="/data/leuven/331/vsc33189/env_setup/conda_base/bin:${PATH}"

# Activate the ml_env environment
source /data/leuven/331/vsc33189/env_setup/conda_base/bin/activate
conda activate /data/leuven/331/vsc33189/env_setup/envs/ml_env

# Run the DL stacking ensemble script
python reg_dlstack_transformer_attention.py
