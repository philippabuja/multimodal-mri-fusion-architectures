# Multimodal MRI fusion for Alzheimer's disease classification

Code for the master's thesis: a 3D CNN trained with a relevance-guided (RG) loss and
Deep Taylor Decomposition (Tinauer et al., 2020), extended from T1 to the quantitative maps
FLAIR, R2* and MTR. The thesis compares early, late, decision-level and dynamic fusion architectures 
and explains the results with different Explainability experiments.

## Code structure

| Folder | Content |
|---|---|
| `1_data/` | age/sex propensity matching, patient-grouped stratified 10-fold splits |
| `2_preprocess/` | HD-BET brain mask, register all modalities into T1 space, 2 mm downsampling |
| `3_model/` | CNN with DTD, early / late / dynamic fusion models, RG loss, data loader and normalisation |
| `4_train/` | training of one split (`_engine/train.py`),`run_experiments.py` for all experiments |
| `5_evaluation/` | out-of-fold predictions, relevance maps, Explainability plots|


