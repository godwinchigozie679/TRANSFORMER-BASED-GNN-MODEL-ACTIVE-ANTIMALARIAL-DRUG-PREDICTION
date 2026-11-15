"""Scaffold splits and 1:1 undersampling (undersampling is only ever applied to training data)."""
import random

import numpy as np


def group_by_scaffold(scaffolds, indices):
    groups = {}
    for i in indices:
        groups.setdefault(scaffolds[i], []).append(i)
    return list(groups.values())


def scaffold_holdout(groups, frac, seed):
    """Whole scaffold groups go to the held-out part until it holds about `frac` of the compounds."""
    n = sum(len(g) for g in groups)
    for attempt in range(100):
        rng = random.Random(seed + attempt)
        order = groups[:]
        rng.shuffle(order)
        held, size = [], 0
        for g in order:
            if size >= frac * n:
                break
            held.append(g)
            size += len(g)
        if abs(size / n - frac) <= 0.02:
            break
    held_idx = np.sort(np.concatenate(held))
    held_set = set(held_idx.tolist())
    rest_idx = np.sort(np.array([i for g in groups for i in g if i not in held_set]))
    return rest_idx, held_idx


def scaffold_kfold(groups, k, seed):
    """Scaffold groups are spread over k folds of similar size (largest groups first)."""
    rng = random.Random(seed)
    order = groups[:]
    rng.shuffle(order)
    order.sort(key=len, reverse=True)
    folds, sizes = [[] for _ in range(k)], [0] * k
    for g in order:
        j = int(np.argmin(sizes))
        folds[j].extend(g)
        sizes[j] += len(g)
    return [np.sort(np.array(f)) for f in folds]


def undersample(idx, y, rng):
    """1:1 random undersampling of the majority class; applied to training data only."""
    idx = np.asarray(idx)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    small, large = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
    keep = rng.choice(large, size=len(small), replace=False)
    out = np.concatenate([small, keep])
    rng.shuffle(out)
    return out
