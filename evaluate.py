"""Stage 4 (report): test metrics, bootstrap confidence intervals, DeLong tests with Holm correction, seed repeats."""
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from config import BASELINES, CFG, DISPLAY, KINDS, logged, path, pkl_key
from train import metrics


def delong_matrix(y, scores):
    """Paired DeLong test (fast midrank version) for every pair of models."""
    order = np.argsort(-y, kind="stable")
    y = y[order]
    S = np.array([s[order] for s in scores])
    m = int(y.sum())
    n = len(y) - m
    k = S.shape[0]
    tx = np.array([rankdata(S[i, :m]) for i in range(k)])
    ty = np.array([rankdata(S[i, m:]) for i in range(k)])
    tz = np.array([rankdata(S[i]) for i in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


@logged
def report():
    keys = KINDS + BASELINES
    res = {}
    for key in keys:
        with open(path("test_results", f"{pkl_key(key)}_results.pkl"), "rb") as f:
            res[key] = pickle.load(f)
    yt = res[keys[0]]["true_labels"]
    prev = float(yt.mean())

    rows = []                                                  # test metrics at 0.5 and at the validation cutoff
    for key in keys:
        r = res[key]
        for thr, name in [(0.5, DISPLAY[key]), (r["cutoff"], f"{DISPLAY[key]} (val-tuned cutoff)")]:
            rows.append(dict(model=name, cutoff=thr, **metrics(yt, r["probabilities"], thr),
                             prevalence_active=prev, n_test=len(yt)))
    pd.DataFrame(rows).to_csv(path("test_set_metrics_table.csv"), index=False)

    rng = np.random.default_rng(CFG["seed"])                   # bootstrap confidence intervals, same resamples
    boots = [rng.integers(0, len(yt), len(yt)) for _ in range(CFG["n_boot"])]
    boots = [b for b in boots if 0 < yt[b].sum() < len(b)]
    out = []
    for key in keys:
        p = res[key]["probabilities"]
        v = {"AUC": [], "PR-AUC": [], "F1": []}
        for b in boots:
            v["AUC"].append(roc_auc_score(yt[b], p[b]))
            v["PR-AUC"].append(average_precision_score(yt[b], p[b]))
            v["F1"].append(f1_score(yt[b], (p[b] >= 0.5).astype(int), zero_division=0))
        m = metrics(yt, p)
        row = {"Model": DISPLAY[key], "AUC": m["auc"]}
        for name, est in [("AUC", m["auc"]), ("PR-AUC", m["pr_auc"]), ("F1", m["f1"])]:
            lo, hi = np.percentile(v[name], [2.5, 97.5])
            row[name] = est
            row[f"{name} 95% CI"] = f"({lo:.3f}, {hi:.3f})"
        out.append({"Model": row["Model"], "AUC": row["AUC"], "AUC 95% CI": row["AUC 95% CI"],
                    "PR-AUC": row["PR-AUC"], "PR-AUC 95% CI": row["PR-AUC 95% CI"], "F1": row["F1"],
                    "F1 95% CI": row["F1 95% CI"], "n_test": len(yt), "prevalence_active": prev})
    pd.DataFrame(out).to_csv(path("bootstrap_ci_table.csv"), index=False)

    aucs, cov = delong_matrix(yt, [res[k]["probabilities"] for k in keys])
    pairs = []
    for i, j in itertools.combinations(range(len(keys)), 2):
        var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
        z = (aucs[i] - aucs[j]) / np.sqrt(var) if var > 0 else 0.0
        pairs.append(dict(a=DISPLAY[keys[i]], b=DISPLAY[keys[j]], auc_a=aucs[i], auc_b=aucs[j],
                          diff=aucs[i] - aucs[j], z=z, p=2 * norm.sf(abs(z))))
    adj = holm([r["p"] for r in pairs])
    dl = pd.DataFrame([{"Model A": r["a"], "Model B": r["b"], "AUC A": r["auc_a"], "AUC B": r["auc_b"],
                        "diff": r["diff"], "z": r["z"], "p": r["p"], "p_holm": a} for r, a in zip(pairs, adj)])
    dl.sort_values("p").to_csv(path("delong_pairwise.csv"), index=False)

    seed_rows = []                                             # variation over repeated final training runs
    for key in KINDS:
        seed_dir = os.path.join(CFG["out_dir"], "test_results", "seeds")
        files = sorted(f for f in (os.listdir(seed_dir) if os.path.isdir(seed_dir) else [])
                       if f.startswith(pkl_key(key) + "_seed"))
        stats = []
        for f in files:
            with open(path("test_results", "seeds", f), "rb") as fh:
                r = pickle.load(fh)
            stats.append((roc_auc_score(r["true_labels"], r["probabilities"]),
                          average_precision_score(r["true_labels"], r["probabilities"])))
        if len(stats) > 1:
            a = np.array(stats)
            seed_rows.append({"Model": DISPLAY[key], "n_seeds": len(stats),
                              "AUROC": f"{a[:, 0].mean():.3f} ± {a[:, 0].std(ddof=1):.3f}",
                              "PR-AUC": f"{a[:, 1].mean():.3f} ± {a[:, 1].std(ddof=1):.3f}"})
    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(path("seed_repeat_table.csv"), index=False)
    print(pd.DataFrame(out)[["Model", "AUC", "AUC 95% CI", "PR-AUC", "PR-AUC 95% CI"]].to_string(index=False))
    print("tables written to", CFG["out_dir"])
