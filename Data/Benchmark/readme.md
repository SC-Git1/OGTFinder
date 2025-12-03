**Li et al. (2019)**
- `Tome.zip`: contains the source code of Tome available at https://github.com/EngqvistLab/Tome. Only modifications made are in setup.py (sklearn -> scikit-learn) and in tome/core/predOGT.py (from sklearn.externals import joblib -> import joblib)
- `tome_model`: folder containing the retrained model
- `evaluate_tome.py`: source code for part of Table 3.
- `tome_train_w_opt_params.py`: script for retraining
- `tome_filtered_features.py`: script to calculate tome input

**Sauer and Wang (2019)**
- `sauer_scripts`: folder containing code excerpts originating from https://github.com/DavidBSauer/OGT_prediction/ for feature calculation.
- `OGT_prediction`: archived code from https://github.com/DavidBSauer/OGT_prediction/ used in model calculation
- `calculate_features`: script to calculate genomic, ORF and protein features.
- `pipeline.py`: script to retrain models. 
- `files.zip`: output files
Training command:
```python3 ./pipeline.py ./species-ogt.tsv ./train-test.tsv ./species-taxon.tsv sauer_features.tsv```

**Licenses**
Licenses from the two repositories of which code was included in this benchmark were copied verbatim from the repositories to the files `LICENSE_OGT_Prediction` and `LICENSE_Tome` respectively.
