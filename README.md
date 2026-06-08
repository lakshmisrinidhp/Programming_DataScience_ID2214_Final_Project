# Molecular Activity Prediction — ID2214 Final Project

Binary classification of small molecules as active or inactive using cheminformatics features and machine learning.

## Problem

Given a molecule's SMILES string, predict whether it is biologically **active** (1) or **inactive** (0). The dataset is imbalanced and models are evaluated by **ROC-AUC**.

## Repository Structure

```
├── Datasets/
│   ├── training_smiles.csv      # SMILES + ACTIVE labels
│   └── test_smiles.csv          # SMILES for final predictions
│
├── Feature_Generation/
│   ├── fingerprints/            # Morgan fingerprint arrays (.npy)
│   └── rdmoldescriptor/         # RDKit descriptor arrays (.npy)
│
├── Figures/
│   ├── Morgan_fingerprint/      # ROC curves for fingerprint models
│   └── rdDescriptor/            # ROC curves for descriptor models
│
└── code/
    ├── Morgan-fingerprints.ipynb              # Feature generation: Morgan fingerprints
    ├── rdMolDescriptors-checkpoint.ipynb      # Feature generation: RDKit descriptors
    ├── Fingerprints_with_LR_and_RF.ipynb      # Models on Morgan fingerprints
    ├── Descriptors_8_with_LR_and_RF.ipynb     # Models on top-8 RDKit descriptors
    └── Descriptors_20_with_LR_and_RF.ipynb    # Models on top-20 RDKit descriptors
```

## Approach

Two feature representations were explored:

**Morgan Fingerprints** — circular fingerprints (radius=2, 1024 bits) generated with RDKit's `MorganGenerator`. Each molecule becomes a binary vector encoding substructural patterns.

**RDKit Molecular Descriptors** — physicochemical properties (e.g. molecular weight, logP, ring counts) computed with RDKit. Top 8 and top 20 most informative descriptors were selected.

Both feature sets were fed into two classifiers:

- **Logistic Regression** — with `StandardScaler` preprocessing and `class_weight="balanced"`
- **Random Forest** — 500 trees, `max_features="sqrt"`, `class_weight="balanced_subsample"`

Models were evaluated using 5-fold stratified cross-validation (ROC-AUC).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn rdkit matplotlib
```

Run notebooks in order:
1. Feature generation notebooks (`Morgan-fingerprints.ipynb`, `rdMolDescriptors-checkpoint.ipynb`)
2. Model training notebooks (`Fingerprints_with_LR_and_RF.ipynb`, `Descriptors_*_with_LR_and_RF.ipynb`)

> **Note:** The generated `.npy` feature files are excluded from this repo (`.gitignore`) due to their size. Run the feature generation notebooks first to recreate them.

