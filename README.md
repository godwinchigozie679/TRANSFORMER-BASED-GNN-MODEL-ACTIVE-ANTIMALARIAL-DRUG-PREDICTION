#ABSTRACT
Malaria remains one of the greatest global threats to human health, and the emergence of drug-resistant Plasmodium strains has intensified the urgency for novel therapeutic interventions. Advances in computational drug discovery have been made, yet many predictive models lack transparency, limiting their utility in guiding rational antimalarial design. This study presents a deep learning framework based on Graph Neural Network (GNN) for predicting antimalarial activity directly from SMILES representations. Using a curated 123,745 compounds from ChEMBL and PubChem datasets, we benchmarked four architectures: a Graph Convolutional Network (GCN), a Graph Attention Network (GAT), a GAT-Transformer hybrid, and a GCN-Transformer, compared with baseline models: random forest, gradient boosting, logistic regression, linear Support Vector Machine (SVM), and K-nearest neighbour (KNN). With the Bemis-Murcko scaffold split, cross-validation, bootstrap confidence intervals, and the DeLong test, an AUROC range of 0.9061--0.9108 was achieved for the GNN models, while 0.8894--0.8958 was recorded for the baseline models. Statistically, the GAT-Transformer model was slightly higher than GCN and GCN-Transformer in terms of accuracy, precision, F1-score, AUROC, and PR-AUC metrics values. For post-hoc interpretability, we integrated SHAP, integrated gradient, and gradient-based saliency mapping. SHAP and integrated gradients strongly share similar features important for both atomic and molecular-level descriptors. This study not only described the behaviour of GNN and transformer-enhanced GNN models in antimalarial activity prediction but also offered hypotheses for chemical follow--up. These hypotheses have not been validated experimentally.

# Antimalarial activity prediction: complete pipeline

Everything sits in ONE folder: the `.py` files, the notebook, and `main_data_2/` with your three data files
(`Drug_activity.csv` with columns `Smiles`, `Labels`; `molecule_graphs.pt`; `enhanced_drug_data_with_descriptors.csv`).

## Where things are saved (all in folders, nothing loose)
| Folder | Content |
|---|---|
| `cache/` | prepared data (graphs, splits) so later stages start fast |
| `eda/` | data visualisation before training: class_balance, descriptor_distributions, correlation_heatmap, sample_molecules, extreme_molecules (PNG), `data_split_overview.pdf`, `descriptor_statistics.csv` (Mann-Whitney, FDR), `dataset_summary.csv`, `splits.csv` |
| `tuning/` | `<model>_tuning_results.csv`, `tuning_runs.csv`, `hyperparameter_table.csv`, `best_params.json` |
| `cross_validation/` | `cv_folds.csv`, `cv_mean_std_table.csv` |
| `trained_models/` | weights for every model and seed (`<kind>_seed<N>.pt`, plus `gnn_model.pt`, `gat_model.pt`, `gcn_transformer_model.pt`, `gat_transformer_model.pt` for the first seed) and `<name>_best_params.json` |
| `training_history/` | per-epoch loss, AUC, accuracy, learning rate for every final model and seed (`*_history.pkl`, `*_seed<N>_history.csv`) |
| `test_results/` | `<model>_results.pkl`, `test_predictions.csv`, `test_set_metrics_table.csv`, `bootstrap_ci_table.csv`, `delong_pairwise.csv`, `seed_repeat_table.csv`, `seeds/` |
| `learning_curves/` | `learning_curve_folds.csv`, `learning_curve_summary.csv` |
| `explainability/` | all explainability CSVs, agreement and deletion-check tables |
| `figures/` | confusion_matrices, roc_curves, pr_curves, training_curves, tuning_results, metrics_comparison, learning_curves, prauc_ci, delong_heatmap, explainability figures |
| `logs/` | `training_log.txt`, `run_log.csv`, `run_info.json` |

## Accuracy settings (config.CFG, all on by default)
* `QUICK = False` is the default; `QUICK = True` (or `--quick`) writes to `quick_test/` so it never mixes with real results.
* `resample_each_epoch`: a fresh 1:1 undersample of the training pool every epoch (uses all inactives over training, test set untouched).
* `embed_categorical`: learned embeddings for atomic number, hybridisation, period and group instead of raw numbers.
* AdamW with gradient clipping; longer tuning budget and patience.
* `ensemble`: the reported GNN result is the mean probability over the training seeds (`--seeds 42 43 44`). Baselines are single models. State this in Methods.
* Probabilities from 1:1 undersampled training are shifted upward, so precision/F1/MCC at cutoff 0.5 can look low; the "(val-tuned cutoff)" rows use a cutoff chosen on validation data only and are the fairer threshold metrics. AUROC/PR-AUC are unaffected.
* Never tune anything on the test set.

## Explainability (same as the original analysis)
Five methods (GNNExplainer, Integrated Gradients, Feature Ablation, Saliency, SHAP), each on both branches
(atomic graph features and molecular descriptors). In the notebook: first the **saved models are loaded** from
`trained_models/` (no retraining), then each method runs in its own cell:
`explain.load_trained_models()` -> `explain.make_setup(...)` -> `run_gnnexplainer`, `run_integrated_gradients`,
`run_feature_ablation`, `run_saliency`, `run_shap` -> `run_agreement` (rank agreement between the methods) ->
`run_deletion_check`. CSV names are the original ones (`saliency_importance.csv`, `descriptor_saliency_importance.csv`, ...)
and the figures are `figures/<method>_atomic_vs_molecular.pdf`.
The explanations are post-hoc hypotheses about what the model uses; they are not validated experimentally.

## Files
| File | What it holds |
|---|---|
| `config.py` | settings, hyperparameter grid, folder layout, quick-test mode, log file / run info |
| `eda.py` | stage 2b `eda` (data visualisation before training) |
| `graphs.py` | atom/bond features and names, scaffolds, RDKit descriptors, loading your graph file |
| `splits.py` | scaffold splits, 1:1 undersampling (training data only) |
| `data_prep.py` | stage 1 `prepare()` |
| `descriptors.py` | descriptor imputation/filter/standardisation (fit on training data only) |
| `models.py` | GCN, GAT, GCN-Transformer, GAT-Transformer with shared classifier head |
| `train.py` | train one model (records per-epoch history), predict, metrics, baselines |
| `tune.py` | stage 2 `tune()` |
| `train_final.py` | stage 3 `final()` |
| `evaluate.py` | stage 4 `report()` |
| `curves.py` | stage 5 `curves()` (learning curves, 5-fold CV, all nine models) |
| `plots.py` | stage 6 `plots()` (figures, original publication style) |
| `explain.py` | stage 7: the five explainability methods, agreement, deletion check |
| `run.py` | command line |
| `make_table.py`, `make_cv_table.py` | LaTeX tables from `test_results/` and `cross_validation/` CSVs (write `test_results/table_*.tex`) |

## Run
Notebook: set `QUICK = True`, run all cells (smoke test), then `QUICK = False` and run again (GPU recommended).
Terminal: `python run.py all --data main_data_2/Drug_activity.csv --graphs main_data_2/molecule_graphs.pt --descriptors main_data_2/enhanced_drug_data_with_descriptors.csv --out . --seeds 42 43 44`

Training AUC/accuracy in `training_history` are running values over each epoch's training batches (dropout on);
validation values use the validation set.
