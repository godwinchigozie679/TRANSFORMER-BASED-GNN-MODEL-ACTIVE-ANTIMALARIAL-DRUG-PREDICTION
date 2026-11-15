"""Stage 3 (final): train the final models on the development set and predict the external test set."""
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from config import BASELINES, CFG, DISPLAY, GRID, KINDS, logged, path, pkl_key, save_run_info
from data_prep import Ctx
from descriptors import FoldData
from train import best_cutoff, fit_baseline, fit_gnn, predict


@logged
def final(seeds):
    save_run_info(seeds=list(seeds))
    ctx = Ctx()
    with open(path("best_params.json")) as f:
        best = json.load(f)
    fd = FoldData(ctx, ctx.inner_train, ctx.inner_val, CFG["seed"])
    print(f"final fit: {len(fd.train)} training (undersampled), {len(fd.val)} validation, "
          f"{len(ctx.test_idx)} test compounds, {fd.table.shape[1]} descriptors kept")
    te = DataLoader([ctx.graphs[i] for i in ctx.test_idx], batch_size=CFG["eval_batch_size"], shuffle=False)
    yt, yv = ctx.y[ctx.test_idx], ctx.y[fd.val]

    test_preds, per_seed = {}, {}

    def write(key, tag, seed, p_val, p_test, folder):
        cut = best_cutoff(yv, p_val)
        res = dict(true_labels=yt, probabilities=p_test, predictions=(p_test >= 0.5).astype(int),
                   predictions_tuned=(p_test >= cut).astype(int), val_labels=yv, val_probabilities=p_val,
                   cutoff=cut, seed=seed, ensemble_size=tag)
        with open(folder, "wb") as f:
            pickle.dump(res, f)

    def save_seed(key, seed, p_val, p_test):
        per_seed.setdefault(key, []).append((p_val, p_test))
        write(key, 1, seed, p_val, p_test, path("test_results", "seeds", f"{pkl_key(key)}_seed{seed}.pkl"))

    def save_main(key):
        """Reported result: the mean over the seeds (an ensemble) if CFG['ensemble'], otherwise the first seed."""
        runs = per_seed[key] if CFG["ensemble"] else per_seed[key][:1]
        p_val, p_test = (np.mean([r[i] for r in runs], axis=0) for i in (0, 1))
        write(key, len(runs), seeds[0], p_val, p_test, path("test_results", f"{pkl_key(key)}_results.pkl"))
        test_preds[DISPLAY[key]] = p_test
        return p_test

    for kind in KINDS:
        params = {k: v for k, v in best[kind].items() if k in GRID}
        for n, seed in enumerate(seeds):
            t0 = time.time()
            model, auc, ep, p_val = fit_gnn(ctx, fd, kind, params, CFG["final_epochs"], CFG["final_patience"], seed)
            p_test = predict(model, te)
            save_seed(kind, seed, p_val, p_test)
            hist = pd.DataFrame(model.history)
            hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
            hist.to_csv(path("training_history", f"{kind}_seed{seed}_history.csv"), index=False)
            if n == 0:
                with open(path("training_history", f"{pkl_key(kind)}_history.pkl"), "wb") as fh:
                    pickle.dump(model.history, fh)
            torch.save(dict(state=model.state_dict(), params=params, kind=kind,
                            x_stats=fd.x_stats, e_stats=fd.e_stats, desc_names=list(fd.pipe.names)),
                       path("trained_models", f"{kind}_seed{seed}.pt"))
            if n == 0:                                         # first seed is also saved under the original names
                torch.save(torch.load(path("trained_models", f"{kind}_seed{seed}.pt"), weights_only=False),
                           path("trained_models", f"{pkl_key(kind)}.pt"))
                with open(path("trained_models", f"{pkl_key(kind)}_best_params.json"), "w") as fh:
                    json.dump(params, fh, indent=2)
            print(f"  {kind} seed {seed}: val AUROC {auc:.4f}, test AUROC {roc_auc_score(yt, p_test):.4f}, "
                  f"AP {average_precision_score(yt, p_test):.4f} ({ep} epochs, {time.time() - t0:.0f}s)", flush=True)
        p_ens = save_main(kind)
        print(f"  {kind}: reported result (mean of {len(per_seed[kind]) if CFG['ensemble'] else 1} seed(s)): "
              f"test AUROC {roc_auc_score(yt, p_ens):.4f}, AP {average_precision_score(yt, p_ens):.4f}", flush=True)
    for key in BASELINES:
        model, p_val = fit_baseline(ctx, fd, key, CFG["seed"])
        p_test = model.predict_proba(fd.table[ctx.test_idx])[:, 1]
        save_seed(key, CFG["seed"], p_val, p_test)
        save_main(key)
        print(f"  {key}: test AUROC {roc_auc_score(yt, p_test):.4f}, AP {average_precision_score(yt, p_test):.4f}")
    out = pd.DataFrame({"smiles": [ctx.smiles[i] for i in ctx.test_idx], "label": yt, **test_preds})
    out.to_csv(path("test_results", "test_predictions.csv"), index=False)   # every model's test probability per compound
