SLURM job scripts for DL stacking ensemble models (GPU).

Partition: gpu_a100
Cluster: wice
GPUs: 1
CPUs/node: 8
Time limit: 3-00:00:00

These jobs train DL base models from scratch using best_params.json
from deep_learning/out/, then tune a meta-learner with Optuna.

Before running, create logfiles directory on the cluster:
  mkdir -p /scratch/leuven/331/vsc33189/projects/ogtfinder/ensemble_models_fixed/implementation/logfiles
