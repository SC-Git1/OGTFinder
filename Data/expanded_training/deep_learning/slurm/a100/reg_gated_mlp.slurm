#!/bin/bash
#SBATCH --job-name=dl_src_gated_mlp
#SBATCH --output=logfiles/dl_src_gated_mlp.out
#SBATCH --error=logfiles/dl_src_gated_mlp.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaldert.francois@student.kuleuven.be
#SBATCH --account=lp_cosubi
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00

# Prevent OpenBLAS/MKL thread oversubscription (let Optuna handle parallelism)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# Change to the working directory
cd /scratch/leuven/331/vsc33189/projects/ogtfinder/deep_learning

# Set up conda path
export PATH="/data/leuven/331/vsc33189/env_setup/conda_base/bin:${PATH}"

# Activate the ml_env environment
source /data/leuven/331/vsc33189/env_setup/conda_base/bin/activate
conda activate /data/leuven/331/vsc33189/env_setup/envs/ml_env

# Run the training script
python reg_gated_mlp.py ../data/train_genus.csv ../data/test_genus.csv --weights_path ../data/weights_train_genus.json
