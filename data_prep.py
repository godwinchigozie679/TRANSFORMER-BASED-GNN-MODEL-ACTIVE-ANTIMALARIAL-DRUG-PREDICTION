"""Stage 1 (prepare): clean the data, load your graphs and descriptors, build the scaffold splits and cache everything."""
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

from config import CFG, logged, path, save_run_info
from graphs import featurise, load_graph_file
from splits import group_by_scaffold, scaffold_holdout, scaffold_kfold


@logged
def prepare():
    save_run_info()
    t0 = time.time()
    df = pd.read_csv(CFG["data_path"])
    sc, lc = CFG["smiles_col"], CFG["label_col"]
    df = df.dropna(subset=[lc]).copy()
    df[lc] = df[lc].astype(int)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{n_raw} labelled rows, {len(df)} after removing exact duplicate rows")
    if CFG.get("limit"):
        df = df.sample(n=min(CFG["limit"], len(df)), random_state=CFG["seed"]).reset_index(drop=True)
    own_graphs = load_graph_file(CFG["graph_path"]) if CFG.get("graph_path") else None
    ext = None
    if CFG.get("desc_path"):                                   # your own descriptor table (SMILES + numeric columns)
        ext = pd.read_csv(CFG["desc_path"]).drop_duplicates(subset=sc).set_index(sc)
        ext_cols = [c for c in ext.columns if c != lc and pd.api.types.is_numeric_dtype(ext[c])
                    and not str(c).startswith("Unnamed")]
        print(f"using your descriptor file: {len(ext_cols)} numeric columns")
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        feats = pool.map(partial(featurise, graph=own_graphs is None, desc=ext is None),
                         df[sc].tolist(), chunksize=256)
    valid = np.array([f is not None for f in feats])
    if own_graphs is not None:
        missing = np.array([smi not in own_graphs for smi in df[sc]])
        if CFG.get("build_missing") and missing.any():
            print(f"building {int(missing.sum())} graphs missing from your graph file")
            extra = [featurise(smi, graph=True, desc=False) if m else None for smi, m in zip(df[sc], missing)]
            for i, e in enumerate(extra):
                if e is not None:
                    feats[i] = (e[0], e[1], e[2], feats[i][3] if feats[i] else e[3], feats[i][4] if feats[i] else None)
            missing = np.zeros(len(df), bool)
        print(f"{int(missing.sum())} molecules have no graph in your file and are dropped")
        valid &= ~missing
    if ext is not None:
        absent = ~df[sc].isin(ext.index).to_numpy()
        print(f"{int(absent.sum())} molecules have no row in your descriptor file and are dropped")
        valid &= ~absent
    df = df[valid].reset_index(drop=True)
    feats = [f for f, v in zip(feats, valid) if v]
    y = df[lc].to_numpy()
    scaffolds = [f[3] for f in feats]
    gx, gei, gea = [], [], []
    for smi, f in zip(df[sc], feats):
        if own_graphs is not None and smi in own_graphs:       # your graph, used exactly as it is
            g = own_graphs[smi]
            ea_ = g.edge_attr.numpy() if getattr(g, "edge_attr", None) is not None else np.zeros((g.edge_index.shape[1], 5))
            gx.append(g.x.numpy().astype(np.float32))
            gei.append(g.edge_index.numpy().astype(np.int64))
            gea.append(ea_.astype(np.float32).reshape(-1, 5))
        else:                                                  # built here (no own file, or --build-missing)
            gx.append(f[0]); gei.append(f[1]); gea.append(f[2])
    if ext is not None:
        desc_all, desc_names = ext.loc[df[sc], ext_cols].to_numpy(dtype=np.float64), np.array(ext_cols)
    else:
        desc_all, desc_names = np.stack([f[4] for f in feats]), np.array([n for n, _ in Descriptors.descList])
    conflict = df.groupby(sc)[lc].nunique()
    print(f"{len(df)} valid molecules ({y.sum()} active, {len(y) - y.sum()} inactive, {100 * y.mean():.1f}% active)")
    print(f"SMILES with conflicting labels: {(conflict > 1).sum()}")

    groups = group_by_scaffold(scaffolds, range(len(df)))
    dev_idx, test_idx = scaffold_holdout(groups, CFG["test_frac"], CFG["seed"])
    dev_groups = group_by_scaffold(scaffolds, dev_idx.tolist())
    folds_val = scaffold_kfold(dev_groups, CFG["n_folds"], CFG["seed"])
    cv = [(np.setdiff1d(dev_idx, v), v) for v in folds_val]
    inner_train, inner_val = scaffold_holdout(dev_groups, CFG["inner_val_frac"], CFG["seed"])
    shared = set(np.array(scaffolds, dtype=object)[test_idx]) & set(np.array(scaffolds, dtype=object)[dev_idx])
    assert not shared, "scaffold overlap between development and test sets"
    prepared = dict(
        smiles=df[sc].tolist(), y=y, scaffolds=scaffolds, desc=desc_all,
        desc_names=desc_names, x=gx, ei=gei, ea=gea,
        dev_idx=dev_idx, test_idx=test_idx, cv=cv, inner_train=inner_train, inner_val=inner_val,
    )
    with open(path("cache", "prepared.pkl"), "wb") as f:
        pickle.dump(prepared, f, protocol=4)
    split = np.full(len(y), "development", dtype=object)       # one row per compound: which part it belongs to
    split[test_idx] = "test"
    fold_of = np.full(len(y), -1)
    for k, (_, v) in enumerate(cv):
        fold_of[v] = k
    inner = np.full(len(y), "", dtype=object)
    inner[inner_train], inner[inner_val] = "inner_train", "inner_val"
    pd.DataFrame(dict(smiles=df[sc], label=y, scaffold=scaffolds, split=split, cv_fold=fold_of,
                      final_fit_part=inner)).to_csv(path("splits.csv"), index=False)
    rows = []
    for name, idx in [("Full dataset", np.arange(len(y))), ("Development set", dev_idx), ("Test set", test_idx)]:
        rows.append(dict(subset=name, compounds=len(idx), active=int(y[idx].sum()),
                         inactive=int(len(idx) - y[idx].sum()), active_pct=round(100 * y[idx].mean(), 2)))
    pd.DataFrame(rows).to_csv(path("dataset_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"{len(desc_names)} descriptors; prepare took {time.time() - t0:.0f}s")


def load_prepared():
    """The cached data as a plain dict (no graph objects built): smiles, y, scaffolds, desc, desc_names, splits."""
    with open(path("cache", "prepared.pkl"), "rb") as f:
        return pickle.load(f)


class Ctx:
    """Loads the cached data once and builds the graph objects."""

    def __init__(self):
        self.__dict__.update(load_prepared())
        self.graphs = [Data(x=torch.from_numpy(self.x[i]), edge_index=torch.from_numpy(self.ei[i]),
                            edge_attr=torch.from_numpy(self.ea[i]), y=torch.tensor([int(self.y[i])]),
                            idx=torch.tensor([i])) for i in range(len(self.y))]
