"""Settings shared by every stage: paths, hyperparameter grid, model names, device, quick-test mode."""
import functools
import json
import os
import sys
import time
from datetime import datetime

import torch


CFG = dict(
    data_path="main_data_2/Drug_activity.csv",   # CSV with a SMILES column and a 0/1 label column
    smiles_col="Smiles",
    label_col="Labels",
    out_dir="run",
    seed=42,
    test_frac=0.20,          # outer scaffold split: external test set
    inner_val_frac=0.15,     # final fit: validation share of the development set (early stopping, cutoff)
    n_folds=5,               # scaffold CV folds inside the development set
    corr_threshold=0.85,     # descriptor redundancy filter (|r| >= threshold is dropped)
    n_configs=20,            # random-search configurations, shared by all four GNNs
    halving_keep=5,          # configs that go on to all folds after the first-fold screen (0 = full CV)
    tune_epochs=50,
    tune_patience=10,
    screen_epochs=20,        # short first-fold screen of all configurations (successive halving)
    final_epochs=200,
    final_patience=30,
    batch_size=128,
    eval_batch_size=2048,
    heads=8,
    n_boot=1000,
    knn_k=15,
    workers=0,               # DataLoader workers (0 is fastest on CPU)
    explain_n=50,            # compounds explained (random, class-balanced sample of the test set)
    shap_nsamples="auto",
    ig_steps=50,
    gnnexplainer_epochs=200,
    gnnexplainer_desc_epochs=100,
    curve_folds=5,           # learning curves: scaffold CV folds (as in the original analysis)
    resample_each_epoch=True,  # fresh 1:1 undersample of the training data every epoch (the model sees all inactives over time)
    embed_categorical=True,    # learned embeddings for atomic number, hybridisation, period, group (inside the model)
    ensemble=True,             # reported predictions = mean over the final-training seeds
    grad_clip=5.0,
    amp=False,               # mixed precision, CUDA only
)


GRID = dict(
    hidden=[64, 128, 256],
    dropout=[0.1, 0.2, 0.3],
    layers=[2, 3, 4],
    lr=[1e-3, 5e-4, 1e-4],
    weight_decay=[1e-5, 1e-4, 5e-4],
    pool=["mean", "mean_max"],
)


KINDS = ["gcn", "gat", "gcn_transformer", "gat_transformer"]


BASELINES = ["baseline_gradient_boosting", "baseline_random_forest", "baseline_logistic_regression",
             "baseline_linear_svm", "baseline_knn"]


DISPLAY = {
    "gcn": "GCN", "gat": "GAT", "gcn_transformer": "GCN-Transformer", "gat_transformer": "GAT-Transformer",
    "baseline_gradient_boosting": "Gradient Boosting", "baseline_random_forest": "Random Forest",
    "baseline_logistic_regression": "Logistic Regression", "baseline_linear_svm": "Linear SVM",
    "baseline_knn": "k-NN",
}


PKL_NAME = {"gcn": "gnn_model", "gat": "gat_model", "gcn_transformer": "gcn_transformer_model",
            "gat_transformer": "gat_transformer_model"}


PRIORITY_DESCRIPTORS = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds",
                        "NumAromaticRings", "HeavyAtomCount", "FractionCsp3", "RingCount"]


METRIC_COLS = ["accuracy", "balanced_accuracy", "auc", "pr_auc", "precision", "recall", "f1", "mcc"]


CURVE_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Everything that is saved goes into a folder (same layout as the original analysis):
#   cache/             prepared data (graphs, splits) so later stages start fast
#   eda/               data visualisation before training, dataset summary, splits.csv, descriptor statistics
#   tuning/            tuning runs, best hyperparameters, hyperparameter table, per-model tuning results
#   cross_validation/  cv_folds.csv, cv_mean_std_table.csv
#   trained_models/    model weights (<kind>_seed<N>.pt, <name>.pt) and best hyperparameters per model
#   training_history/  per-epoch loss / AUC / accuracy of every final model
#   test_results/      result pickles, test predictions, test metrics, bootstrap CIs, DeLong, seed repeats
#   learning_curves/   learning-curve tables
#   explainability/    importance CSVs, agreement and deletion checks
#   figures/           every figure (PDF)
#   logs/              training_log.txt, run_log.csv, run_info.json
LOG_FILES = {"training_log.txt", "run_log.csv", "run_info.json"}


def folder_for(name):
    """One folder per kind of output (see the table above)."""
    if name in LOG_FILES:
        return "logs"
    if name.startswith(("tuning_runs", "best_params", "hyperparameter_table")) or name.endswith("_tuning_results.csv"):
        return "tuning"
    if name.startswith(("cv_folds", "cv_mean_std")):
        return "cross_validation"
    if name.startswith("learning_curve"):
        return "learning_curves"
    if name.startswith("explainability_") or name.endswith("_importance.csv"):
        return "explainability"
    if name.startswith(("descriptor_statistics", "dataset_summary", "splits")):
        return "eda"
    return "test_results"


def path(*parts):
    """path("x.csv") -> <out>/<folder for x>/x.csv ; path("folder", "x") -> <out>/folder/x."""
    if len(parts) == 1:
        parts = (folder_for(parts[0]), parts[0])
    p = os.path.join(CFG["out_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def pkl_key(key):
    return PKL_NAME.get(key, key)


def apply_quick():
    """Tiny settings for a smoke test of the whole pipeline (minutes, not hours)."""
    CFG.update(n_configs=2, halving_keep=1, tune_epochs=3, tune_patience=2, final_epochs=4, final_patience=2,
               n_boot=50, screen_epochs=2)
    GRID.update(hidden=[64], layers=[2, 3])
    CURVE_FRACTIONS[:] = [0.5, 1.0]
    CFG["curve_folds"] = 2
    if os.path.basename(os.path.normpath(CFG["out_dir"])) != "quick_test":
        CFG["out_dir"] = os.path.join(CFG["out_dir"], "quick_test")   # never mixes with the real results
    print("=" * 70 + "\nQUICK TEST MODE: tiny epochs, results go to " + CFG["out_dir"]
          + "\nThese numbers are NOT for the paper. Set QUICK = False for the real run.\n" + "=" * 70)
    CFG.update(explain_n=6, shap_nsamples=100, ig_steps=5, gnnexplainer_epochs=10, gnnexplainer_desc_epochs=10)


# ----------------------------------------------------------------------------- logging
class _Tee:
    """Copies everything printed to training_log.txt as well as the screen."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log():
    """Everything printed from now on is also appended to <out_dir>/training_log.txt."""
    if isinstance(sys.stdout, _Tee):
        return
    sys.stdout = _Tee(sys.stdout, open(path("training_log.txt"), "a", encoding="utf-8"))


def save_run_info(**extra):
    """Settings, grid, package versions and device, so the run can be reported and reproduced."""
    import numpy, pandas, rdkit, sklearn, torch_geometric
    info = dict(time=datetime.now().isoformat(timespec="seconds"), device=str(DEVICE),
                versions=dict(torch=torch.__version__, torch_geometric=torch_geometric.__version__,
                              rdkit=rdkit.__version__, sklearn=sklearn.__version__, numpy=numpy.__version__,
                              pandas=pandas.__version__),
                config={k: v for k, v in CFG.items()}, grid=GRID, **extra)
    with open(path("run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def logged(fn):
    """Stage decorator: log file, start/end time, and one line per stage in run_log.csv."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        start_log()
        print(f"\n===== {fn.__name__} started {datetime.now():%Y-%m-%d %H:%M:%S} =====")
        t0 = time.time()
        out = fn(*a, **k)
        sec = time.time() - t0
        new = not os.path.exists(path("run_log.csv"))
        with open(path("run_log.csv"), "a") as f:
            if new:
                f.write("stage,finished,minutes\n")
            f.write(f"{fn.__name__},{datetime.now():%Y-%m-%d %H:%M:%S},{sec / 60:.2f}\n")
        print(f"===== {fn.__name__} finished in {sec / 60:.1f} min =====")
        return out
    return wrapper
