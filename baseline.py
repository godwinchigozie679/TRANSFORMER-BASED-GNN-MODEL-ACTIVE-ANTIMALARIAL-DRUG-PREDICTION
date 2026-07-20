import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

from data_prep import cross_validation_splits
from models import BASELINE_REGISTRY


def cross_validate_one_baseline(dataset_df, model_name, n_folds=5, seed=42):
    """Runs a single named baseline (see models.BASELINE_REGISTRY for the
    options) through the same k-fold splits the GNNs get, so the
    comparison is fair."""
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"unknown baseline '{model_name}', options are: {list(BASELINE_REGISTRY.keys())}")

    build_model = BASELINE_REGISTRY[model_name]
    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}
    n_failed = 0

    for fold_idx, train_df, val_df, descriptor_cols, _ in cross_validation_splits(dataset_df, n_folds, seed):
        X_train, y_train = train_df[descriptor_cols].values, train_df['Labels'].values
        X_val, y_val = val_df[descriptor_cols].values, val_df['Labels'].values

        model = build_model(seed)

        # knn's n_neighbors can't exceed the training fold size — on a
        # small or heavily undersampled fold, the default of 15 can be
        # more than there are rows to look at
        if hasattr(model, 'n_neighbors'):
            model.n_neighbors = max(1, min(model.n_neighbors, len(X_train) - 1))

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]
        except Exception as e:
            # one baseline hiccupping on one fold (mlp's internal early-stopping
            # split needs a couple of samples per class, which a small or
            # unusually undersampled fold can fail to provide) shouldn't take
            # down a run that's also training three GNNs and five other
            # classical models — skip this fold for this model and move on
            n_failed += 1
            print(f"  fold {fold_idx + 1}: {model_name} failed on this fold ({e}), skipping it")
            continue

        fold_metrics['accuracy'].append(accuracy_score(y_val, preds))
        fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_val, preds))
        fold_metrics['auc'].append(roc_auc_score(y_val, probs))
        fold_metrics['precision'].append(precision_score(y_val, preds, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_val, preds, zero_division=0))
        fold_metrics['f1'].append(f1_score(y_val, preds, zero_division=0))

        print(f"  fold {fold_idx + 1}: acc={fold_metrics['accuracy'][-1]:.3f} auc={fold_metrics['auc'][-1]:.3f}")

    if not fold_metrics['accuracy']:
        print(f"{model_name}: failed on every fold, no metrics to report")
        return fold_metrics

    if n_failed:
        print(f"{model_name}: {n_failed}/{n_folds} folds failed and were skipped — "
              f"summary below is over the remaining {n_folds - n_failed}")

    print(f"{model_name} summary:")
    for key, values in fold_metrics.items():
        print(f"  {key}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    return fold_metrics


def cross_validate_baseline(dataset_df, n_folds=5, seed=42, model_name='random_forest'):
    """Old single-model entry point, kept for backward compatibility —
    defaults to random forest like this used to always do. Use
    cross_validate_all_baselines below to run more than one."""
    return cross_validate_one_baseline(dataset_df, model_name, n_folds, seed)


def cross_validate_all_baselines(dataset_df, n_folds=5, seed=42, model_names=None):
    """Runs every baseline in model_names (default: all of them) through
    the same k-fold splits. Returns {model_name: fold_metrics},
    compare all six classical baselines against each other and
    against the GNNs in one table.

    """
    model_names = model_names or list(BASELINE_REGISTRY.keys())
    results = {}

    for name in model_names:
        print(f"\n=== baseline: {name} ===")
        try:
            results[name] = cross_validate_one_baseline(dataset_df, name, n_folds, seed)
        except Exception as e:
            print(f"{name} failed entirely ({e}), skipping it and continuing with the rest")
            results[name] = {'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}

    return results
