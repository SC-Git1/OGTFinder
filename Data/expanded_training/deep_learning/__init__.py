"""
Tabular Deep Learning Architectures

Alternative neural network architectures optimized for tabular data regression.
Designed for ~8k samples with ~1000 features.

Available architectures by category:

A. "Start Here" - High probability of working well:
  A1. reg_baseline_mlp.py     - Regularized Baseline MLP (2-4 layers)
  A2. reg_rtdl_resnet.py      - RTDL ResNet MLP (residual blocks)
  A3. reg_realmlp.py          - RealMLP (modern default-tuned MLP)
  A4. reg_tabm.py             - TabM (parameter-efficient MLP ensemble)
  A5. reg_gated_mlp.py        - Gated MLP (feature-wise gates)
  A6. reg_sparse_mlp.py       - Sparse/Locally-Sparse MLP

B. Feature Selection / Interpretability:
  B7. reg_tabnet.py           - TabNet (sequential attentive feature selection)
  B8. reg_lassonet.py         - LassoNet (global feature selection path)
  B9. reg_gandalf.py          - GANDALF (gated feature learning units)
  B10. reg_danets.py          - DANets (feature grouping/abstraction)

C. Retrieval / Memory-based:
  C11. reg_tabr.py            - TabR (retrieval-augmented MLP with kNN)
  C12. reg_hopular.py         - Hopular (Hopfield memory access)

D. Attention / Transformers:
  D13. reg_ft_transformer.py  - FT-Transformer (feature tokenizer transformer)
  D14. reg_saint.py           - SAINT (self + intersample attention, contrastive pretrain)

E. Neural Trees / Boosting:
  E15. reg_node.py            - NODE (differentiable oblivious decision ensembles)
  E16. reg_grownet.py         - GrowNet (gradient boosting with neural nets)

F. Pretraining Strategies:
  F17. reg_vime.py            - VIME (self-supervised: reconstruction + mask prediction)
  F18. reg_saint.py --variant pretrain  - SAINT contrastive pretraining

Other architectures:
  - reg_resnet_mlp.py         - Simple ResNet MLP variant
  - reg_wide_deep.py          - Wide & Deep network
  - reg_tabnet_inspired.py    - Simplified TabNet-inspired model
  - reg_attention_mlp.py      - Self-attention MLP
  - reg_snn.py                - Self-normalizing networks (SELU)

Usage:
    python reg_<architecture>.py train_data.feather test_data.feather [--output_dir ./output]

All scripts now use Optuna for hyperparameter tuning with N_TRIALS=300.
The scripts will:
    1. Load training data and split into train/val
    2. Run Optuna hyperparameter optimization
    3. Train final model with best parameters on train+val
    4. Evaluate on test data and save results

Shared utilities are in utils.py:
    - Data loading and preprocessing
    - Model evaluation and metrics
    - Optuna callbacks and study management
    - Plotting and result saving
"""

__version__ = '0.3.0'
