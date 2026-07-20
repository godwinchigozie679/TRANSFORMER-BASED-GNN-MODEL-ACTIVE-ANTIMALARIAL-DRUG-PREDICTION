# Transformer-Based GNN Model for Active Antimalarial Drug Prediction

Graph neural network pipeline for predicting antimalarial activity from
molecular structure (SMILES), comparing four GNN architectures (GCN,
GAT, GCN-Transformer, GAT-Transformer) against five classical
baselines, with full explainability support (GNNExplainer, Integrated
Gradients, Feature Ablation, Saliency, SHAP) across both the atomic
graph and molecular descriptor branches.

## Usage

```bash
python run.py cv              # cross-validate all models
python run.py train-final     # 80/20 split, tune, train, save
python run.py compare         # generate comparison plots
python run.py explain         # run all explainability methods
python run.py all             # run everything in sequence
```

See `run_pipeline.ipynb` for the notebook version.

## Setup

```bash
pip install -r requirements.txt
```
