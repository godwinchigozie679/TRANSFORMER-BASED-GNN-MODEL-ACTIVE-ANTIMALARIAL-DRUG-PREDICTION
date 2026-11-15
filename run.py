"""Command-line entry point: python run.py all --data Drug_activity.csv --graphs molecule_graphs.pt --descriptors enhanced_drug_data_with_descriptors.csv --out .

Stages (run in this order; `all` runs them in sequence):
    prepare  eda  tune  final  report  curves  plots  explain
Examples:
    python run.py all --quick                  # smoke test (minutes)
    python run.py all                          # full run
    python run.py final --seeds 42 43 44       # repeat final training with several seeds
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from config import CFG, apply_quick, KINDS
from curves import curves
import eda
from data_prep import prepare
from evaluate import report
import explain
from plots import plots
from train_final import final
from tune import tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "eda", "tune", "final", "report", "curves", "explain", "plots", "all"])
    ap.add_argument("--data", default=None, help="CSV path (overrides data_path)")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--graphs", default=None, help="your graph file (dict SMILES -> Data, or list of Data with .smiles)")
    ap.add_argument("--descriptors", default=None, help="your descriptor CSV (SMILES column + numeric columns)")
    ap.add_argument("--build-missing", action="store_true", help="build graphs that are missing from --graphs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="training seeds for the final stage")
    ap.add_argument("--full-cv-tuning", action="store_true", help="every configuration on all folds (slower)")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--limit", type=int, default=None, help="use a random subset of N molecules")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--curve-folds", type=int, default=5, help="scaffold CV folds for the learning curves")
    ap.add_argument("--explain-model", default="gat_transformer", choices=KINDS)
    ap.add_argument("--explain-n", type=int, default=50, help="compounds explained (random, class-balanced)")
    a = ap.parse_args()
    if a.data:
        CFG["data_path"] = a.data
    if a.out:
        CFG["out_dir"] = a.out
    CFG["graph_path"], CFG["desc_path"], CFG["build_missing"] = a.graphs, a.descriptors, a.build_missing
    if a.full_cv_tuning:
        CFG["halving_keep"] = 0
    if a.amp:
        CFG["amp"] = True
    if a.limit:
        CFG["limit"] = a.limit
    CFG["curve_folds"], CFG["explain_n"] = a.curve_folds, a.explain_n
    if a.quick:
        apply_quick()
        a.explain_n = 6
    torch.set_num_threads(a.threads or os.cpu_count() or 1)
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    stages = (["prepare", "eda", "tune", "final", "report", "curves", "plots", "explain"] if a.stage == "all"
              else [a.stage])
    for s in stages:
        t0 = time.time()
        {"prepare": prepare, "eda": eda.run_all, "tune": tune, "final": lambda: final(a.seeds), "report": report,
         "curves": curves,
         "explain": lambda: explain.run_all(a.explain_model),
         "plots": plots}[s]()
        print(f"[{s}] finished in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
