"""
80/20 train/test split, then 5-fold CV inside the 80% for tuning, then
one final model on the full 80%, tested once on the 20% at the end.

Test set stays out of tuning entirely - outer_split() sets it aside and
train_with_best_params() is the only thing that touches it, right at
the end.

Small grid on purpose, not trying to be exhaustive.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_prep import cross_validation_splits, prepare_split, outer_split
from graphs import build_graph_dataset
from train import train_model, evaluate_model
from config import TRAIN_CONFIG

PARAM_GRID = {
    'hidden_dim': [64, 128],
    'dropout': [0.2, 0.4],
    'num_layers': [2, 4],  # want to know if fewer layers actually generalizes better
}


def _param_combos(param_grid):
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        yield dict(zip(keys, values))


def tune_hyperparameters(model_class, outer_train_df, n_folds=3, seed=42,
                          param_grid=None, tuning_epochs=25, tuning_patience=6, verbose=True):
    """CV inside outer_train_df only, never touches the test set.
    Fewer epochs than the real run since we're just comparing combos.

    Returns results_df sorted best-first and best_params."""
    param_grid = param_grid or PARAM_GRID
    rows = []

    for combo in _param_combos(param_grid):
        fold_aucs = []
        for fold_idx, train_df, val_df, descriptor_cols, _ in cross_validation_splits(outer_train_df, n_folds, seed):
            train_graphs = build_graph_dataset(train_df, descriptor_cols)
            val_graphs = build_graph_dataset(val_df, descriptor_cols)

            model = model_class(**combo)
            tune_config = {**TRAIN_CONFIG, 'epochs': tuning_epochs, 'early_stop_patience': tuning_patience}
            _, _, best_val_auc = train_model(model, train_graphs, val_graphs, config=tune_config, verbose=False)
            fold_aucs.append(best_val_auc)

        row = {**combo, 'mean_val_auc': float(np.mean(fold_aucs)), 'std_val_auc': float(np.std(fold_aucs))}
        rows.append(row)
        if verbose:
            print(f"  {combo}: mean val auc {row['mean_val_auc']:.3f} +/- {row['std_val_auc']:.3f}")

    results_df = pd.DataFrame(rows).sort_values('mean_val_auc', ascending=False).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_params = {}
    for key in param_grid.keys():
        value = best_row[key]
        # pandas upcasts to float when mixed with the auc columns, cast back
        best_params[key] = int(value) if isinstance(param_grid[key][0], int) else float(value)

    return results_df, best_params


def prepare_final_split(outer_train_df, outer_test_df, seed=42):
    """Same split train_with_best_params uses internally. Pulled out on
    its own so anything reloading a checkpoint later can rebuild the
    same descriptor_cols it was trained with. Change 0.15 here, change
    it there too."""
    inner_train_df, inner_val_df = train_test_split(
        outer_train_df, test_size=0.15, stratify=outer_train_df['Labels'], random_state=seed
    )
    return prepare_split(
        inner_train_df.reset_index(drop=True), inner_val_df.reset_index(drop=True), outer_test_df, seed
    )


def train_with_best_params(model_class, best_params, outer_train_df, outer_test_df,
                            seed=42, save_as=None, config_overrides=None, model_dir='trained_models'):
    """Final fit on the full 80%, one eval on the 20%. This is the
    number to report, not anything from tuning above.

    Saves best_params alongside the checkpoint if save_as is set -
    needed later to rebuild the model with the right shapes before
    load_state_dict."""
    train_df, val_df, test_df, descriptor_cols, _ = prepare_final_split(outer_train_df, outer_test_df, seed)

    train_graphs = build_graph_dataset(train_df, descriptor_cols)
    val_graphs = build_graph_dataset(val_df, descriptor_cols)
    test_graphs = build_graph_dataset(test_df, descriptor_cols)

    model = model_class(**best_params)
    train_config = {**TRAIN_CONFIG, **(config_overrides or {})}
    model, history, best_val_auc = train_model(model, train_graphs, val_graphs, config=train_config, save_as=save_as)

    if save_as:
        os.makedirs(model_dir, exist_ok=True)
        params_path = os.path.join(model_dir, f'{save_as}_best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        print(f"saved best_params to {params_path}")

    model_device = next(model.parameters()).device
    result = evaluate_model(model, test_graphs, device=model_device)

    return model, history, result