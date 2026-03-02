#!/bin/bash
#SBATCH --job-name=dl_src_wide_deep
#SBATCH --output=logfiles/dl_src_wide_deep.out
#SBATCH --error=logfiles/dl_src_wide_deep.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jaldert.francois@student.kuleuven.be
#SBATCH --account=lp_cosubi
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00

# Prevent OpenBLAS/MKL thread oversubscription (let Optuna handle parallelism)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# H100 CUDA setup
module load CUDA/12.6.0

export CUDA_HOME=${EBROOTCUDA:-/apps/leuven/rocky8/zen4-h100/2024a/software/CUDA/12.6.0}
export CUDA_DIR=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Required: tell XLA where libdevice is
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Optional: keep auto-JIT disabled (but libdevice still needed)
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# Optional: reduce numerical variability
export TF_ENABLE_ONEDNN_OPTS=0
export NVIDIA_TF32_OVERRIDE=0
export TF_DETERMINISTIC_OPS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optional: silence known warnings
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONWARNINGS="ignore::pydantic._internal._generate_schema.UnsupportedFieldAttributeWarning"

# Change to the working directory
cd /scratch/leuven/331/vsc33189/projects/ogtfinder/deep_learning

# Set up conda path
export PATH="/data/leuven/331/vsc33189/env_setup/conda_base/bin:${PATH}"

# Activate the ml_env environment
source /data/leuven/331/vsc33189/env_setup/conda_base/bin/activate
conda activate /data/leuven/331/vsc33189/env_setup/envs/ml_env

# Run the training script
python reg_wide_deep.py ../data/train_genus.csv ../data/test_genus.csv --weights_path ../data/weights_train_genus.json
