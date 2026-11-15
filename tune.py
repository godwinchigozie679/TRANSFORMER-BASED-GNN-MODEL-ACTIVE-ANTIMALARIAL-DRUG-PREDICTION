"""Stage 2 (tune): random search with five-fold scaffold cross-validation on the development set only."""
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd

from config import BASELINES, CFG, DISPLAY, GRID, KINDS, METRIC_COLS, logged, path, pkl_key
from data_prep import Ctx
from descriptors import FoldData
from train import fit_baseline, fit_gnn, metrics


def sample_configs():
    keys = list(GRID)
    space = list(itertools.product(*[GRID[k] for k in keys]))
    rng = random.Random(CFG["seed"])
    chosen = rng.sample(space, min(CFG["n_configs"], len(space)))
    return [dict(zip(keys, c)) for c in chosen]


@logged
def tune():
    ctx = Ctx()
    configs = sample_configs()
    runs_path = path("tuning_runs.csv")
    runs = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()
    folds = {}

    def fold(k):
        if k not in folds:
            tr, va = ctx.cv[k]
            folds[k] = FoldData(ctx, tr, va, CFG["seed"] + k)
        return folds[k]

    def done(kind, cid, k):
        return len(runs) and ((runs["model"] == kind) & (runs["cfg_id"] == cid) & (runs["fold"] == k)).any()

    def run(kind, cid, k, screen=False):
        """screen=True: short run on the first fold, recorded as fold -1 (used only to pick finalists)."""
        nonlocal runs
        rec = -1 if screen else k
        if done(kind, cid, rec):
            return float(runs[(runs["model"] == kind) & (runs["cfg_id"] == cid) & (runs["fold"] == rec)]["auc"].iloc[0])
        t0 = time.time()
        fd = fold(0 if screen else k)
        _, auc, ep, p = fit_gnn(ctx, fd, kind, configs[cid], CFG["screen_epochs"] if screen else CFG["tune_epochs"],
                                CFG["tune_patience"], CFG["seed"] + (0 if screen else k))
        row = dict(model=kind, cfg_id=cid, fold=rec, epochs=ep, seconds=round(time.time() - t0, 1),
                   **{f"p_{a}": b for a, b in configs[cid].items()}, **metrics(ctx.y[fd.val], p))
        runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
        runs.to_csv(runs_path, index=False)                     # saved after every run, so a stop can resume
        print(f"  {kind} cfg {cid} fold {k}: AUROC {auc:.4f} ({ep} epochs, {row['seconds']}s)", flush=True)
        return auc

    best = {}
    for kind in KINDS:
        print(f"tuning {kind}")
        n_cfg, keep = len(configs), CFG["halving_keep"]
        if keep and keep < n_cfg:
            first = {cid: run(kind, cid, 0, screen=True) for cid in range(n_cfg)}   # short screen, first fold
            finalists = sorted(first, key=first.get, reverse=True)[:keep]
        else:
            finalists = list(range(n_cfg))
        mean, std = {}, {}
        for cid in finalists:
            sc_ = [run(kind, cid, k) for k in range(CFG["n_folds"])]
            mean[cid], std[cid] = float(np.mean(sc_)), float(np.std(sc_))
        pd.DataFrame([dict(**configs[c], mean_val_auc=mean[c], std_val_auc=std[c]) for c in finalists]).to_csv(
            path(f"{pkl_key(kind)}_tuning_results.csv"), index=False)   # candidates that reached full CV
        cid = max(mean, key=mean.get)
        scores = [run(kind, cid, k) for k in range(CFG["n_folds"])]
        best[kind] = dict(cfg_id=cid, mean_val_auc=float(np.mean(scores)), std_val_auc=float(np.std(scores)),
                          **configs[cid])
        print(f"  best {kind}: {configs[cid]}  CV AUROC {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    with open(path("best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    rows = []                                                  # tuned cross-validation metrics, per fold
    for kind in KINDS:
        sub = runs[(runs["model"] == kind) & (runs["cfg_id"] == best[kind]["cfg_id"]) & (runs["fold"] >= 0)]
        rows += [dict(model=kind, fold=int(r.fold), **{m: getattr(r, m) for m in METRIC_COLS}) for r in sub.itertuples()]
    for key in BASELINES:
        print(f"cross-validating {key}")
        for k in range(CFG["n_folds"]):
            _, p = fit_baseline(ctx, fold(k), key, CFG["seed"] + k)
            rows.append(dict(model=key, fold=k, **{m: metrics(ctx.y[fold(k).val], p)[m] for m in METRIC_COLS}))
    cvdf = pd.DataFrame(rows)
    cvdf.to_csv(path("cv_folds.csv"), index=False)
    table = []
    for key in KINDS + BASELINES:
        sub = cvdf[cvdf["model"] == key]
        table.append({"model": key, **{m: f"{sub[m].mean():.3f} ± {sub[m].std(ddof=1):.3f}" for m in METRIC_COLS}})
    pd.DataFrame(table).to_csv(path("cv_mean_std_table.csv"), index=False)
    print(pd.DataFrame(table).to_string(index=False))
    hp = pd.DataFrame([{"Model": DISPLAY[k], "Hidden size": v["hidden"], "Layers": v["layers"],
                        "Dropout": v["dropout"], "Learning rate": v["lr"], "Weight decay": v["weight_decay"],
                        "Pooling": v["pool"], "CV AUROC": f"{v['mean_val_auc']:.3f} ± {v['std_val_auc']:.3f}"}
                       for k, v in best.items()])
    hp.to_csv(path("hyperparameter_table.csv"), index=False)
