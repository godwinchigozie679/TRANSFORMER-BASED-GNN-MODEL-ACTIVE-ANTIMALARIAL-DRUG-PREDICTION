"""Stage 5 (curves): learning curves, validation performance versus training-set size (all nine models)."""
import copy
import json
import os

import numpy as np
import pandas as pd

from config import BASELINES, CFG, CURVE_FRACTIONS, GRID, KINDS, logged, path
from data_prep import Ctx
from descriptors import FoldData
from train import fit_baseline, fit_gnn, metrics

LC_METRICS = ("auc", "accuracy", "pr_auc", "f1", "mcc")


def stratified_subsample(train, y, fraction, rng):
    """Stratified random subsample of the (already undersampled) training fold."""
    if fraction >= 1.0:
        return train
    keep = []
    for c in (0, 1):
        idx = train[y[train] == c]
        keep.append(rng.choice(idx, size=max(int(round(fraction * len(idx))), 2), replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


@logged
def curves():
    """Each model is trained on a growing share of every cross-validation training fold (undersampled 1:1) and
    scored on the matching validation fold, so the curves use the development set only. GNNs use their tuned
    hyperparameters. Writes test_results/learning_curve_folds.csv and learning_curve_summary.csv."""
    ctx = Ctx()
    with open(path("best_params.json")) as f:
        best = json.load(f)
    out_path = path("learning_curve_folds.csv")
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    done = {(r["model"], r["fraction"], r["fold"]) for r in rows}
    names = {**{k: k for k in KINDS}, **{b: b.replace("baseline_", "") for b in BASELINES}}
    for k in range(CFG["curve_folds"]):
        tr, va = ctx.cv[k]
        full = FoldData(ctx, tr, va, CFG["seed"] + k)
        yv = ctx.y[full.val]
        for frac in CURVE_FRACTIONS:
            todo = [key for key in names if (names[key], frac, k) not in done]
            if not todo:
                continue
            fd = copy.copy(full)
            fd.train = stratified_subsample(full.train, ctx.y, frac, np.random.default_rng(CFG["seed"] + k))
            fd.pool = fd.train                                  # the subsample is this fit's whole training pool
            for key in todo:
                if key in KINDS:
                    params = {a: b for a, b in best[key].items() if a in GRID}
                    _, _, _, p = fit_gnn(ctx, fd, key, params, CFG["tune_epochs"], CFG["tune_patience"], CFG["seed"] + k)
                else:
                    try:
                        _, p = fit_baseline(ctx, fd, key, CFG["seed"] + k)
                    except Exception as err:
                        print(f"  {key} failed ({err}), skipped")
                        continue
                m = metrics(yv, p)
                rows.append(dict(model=names[key], fraction=frac, n_train=len(fd.train), fold=k,
                                 **{c: m[c] for c in LC_METRICS}))
                pd.DataFrame(rows).to_csv(out_path, index=False)                    # resumable
                print(f"  fold {k + 1} fraction {frac}: {names[key]} n={len(fd.train)} AUROC {m['auc']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    g = table.groupby(["model", "fraction"])
    summary = g.agg(n_train=("n_train", "mean"), n_folds=("fold", "nunique"),
                    **{f"{m}_mean": (m, "mean") for m in LC_METRICS},
                    **{f"{m}_sd": (m, "std") for m in LC_METRICS}).reset_index()
    summary.fillna({c: 0.0 for c in summary.columns if c.endswith("_sd")}).to_csv(
        path("learning_curve_summary.csv"), index=False)
