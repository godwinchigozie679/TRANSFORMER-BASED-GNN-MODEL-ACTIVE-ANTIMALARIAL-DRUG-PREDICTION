"""Descriptor preprocessing (imputation, redundancy filter, standardisation, fitted on training data only) and the per-fit data bundle."""
import numpy as np
import torch

from config import CFG, DEVICE, PRIORITY_DESCRIPTORS
from splits import undersample


class DescPipe:
    """Imputation, redundancy filter and standardisation, fitted on training data only."""

    def fit(self, X, names, threshold):
        X = np.where(np.isfinite(X), X, np.nan)
        cols = np.where(np.isnan(X).mean(0) <= 0.5)[0]
        X = X[:, cols]
        med = np.nanmedian(X, axis=0)
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
        ok = X.var(0) > 1e-10
        cols, X, med = cols[ok], X[:, ok], med[ok]
        corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        nm = list(names[cols])
        order = [nm.index(p) for p in PRIORITY_DESCRIPTORS if p in nm]
        order += [j for j in range(len(nm)) if j not in order]
        chosen = []
        for j in order:
            if all(corr[j, k] < threshold for k in chosen):
                chosen.append(j)
        chosen = np.array(sorted(chosen))
        self.cols, self.med = cols[chosen], med[chosen]
        Xs = X[:, chosen]
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-8
        self.names = names[self.cols]
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X[:, self.cols]), X[:, self.cols], np.nan)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.med[c]
        return np.clip((X - self.mu) / self.sd, -10, 10).astype(np.float32)


class FoldData:
    """Everything one fit needs. `pool` = all training compounds of this fit; `train` = one 1:1 undersample of
    the pool (used by the classical baselines); GNNs draw a fresh 1:1 undersample of the pool every epoch.
    Descriptor cleaning and input scalers are fitted on the training pool only."""

    def __init__(self, ctx, train_idx, val_idx, seed):
        rng = np.random.default_rng(seed)
        self.val = np.asarray(val_idx)
        self.pool = np.asarray(train_idx)
        self.train = undersample(self.pool, ctx.y, rng)
        self.pipe = DescPipe().fit(ctx.desc[self.pool], ctx.desc_names, CFG["corr_threshold"])
        self.table = self.pipe.transform(ctx.desc)               # standardised descriptors, all molecules
        xa = torch.cat([ctx.graphs[i].x for i in self.pool])
        ea = torch.cat([ctx.graphs[i].edge_attr for i in self.pool])
        self.x_stats = (xa.mean(0), xa.std(0).clamp_min(1e-6))
        self.e_stats = (ea.mean(0), ea.std(0).clamp_min(1e-6))
        self.table_t = torch.from_numpy(self.table).to(DEVICE)
