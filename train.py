import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from torch_geometric.loader import DataLoader

import config as config_module
from config import TRAIN_CONFIG, set_seed


def run_epoch(model, loader, criterion, optimizer=None, device=None):
    device = device or config_module.DEVICE
    # no optimizer = eval mode, otherwise one training pass
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    preds, labels, probs = [], [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad()

            out = model(batch).squeeze()
            loss = criterion(out, batch.y.float())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.extend((out > 0.5).float().detach().cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
            probs.extend(out.detach().cpu().numpy())

    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, probs),
    }
    return metrics, np.array(preds), np.array(labels), np.array(probs)


def train_model(model, train_graphs, val_graphs, config=None, verbose=True, device=None, save_as=None):
    """If MPS chokes on an unsupported op, catches it and retries the
    whole run on CPU instead of crashing partway through. Only retries
    once -- a second failure means something else is wrong.

    save_as (e.g. 'gnn_model') writes {save_as}_best.pth and
    {save_as}_history.pkl to the usual model/history dirs."""
    device = device or config_module.DEVICE
    train_config = {**TRAIN_CONFIG, **(config or {})}

    try:
        model, history, best_val_auc = _train_model_on_device(
            model, train_graphs, val_graphs, train_config, verbose, device)
    except RuntimeError as e:
        if device.type == 'mps':
            print(f"MPS hit an unsupported operation ({e}); retrying this run on CPU instead.")
            model, history, best_val_auc = _train_model_on_device(
                model.to('cpu'), train_graphs, val_graphs, train_config, verbose, torch.device('cpu'))
        else:
            raise

    if save_as:
        _save_checkpoint(model, history, save_as)

    return model, history, best_val_auc


def _save_checkpoint(model, history, name):
    os.makedirs(config_module.MODEL_DIR, exist_ok=True)
    os.makedirs(config_module.HISTORY_DIR, exist_ok=True)

    model_path = os.path.join(config_module.MODEL_DIR, f'{name}_best.pth')
    torch.save(model.state_dict(), model_path)

    history_path = os.path.join(config_module.HISTORY_DIR, f'{name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"saved checkpoint to {model_path}, history to {history_path}")


def load_checkpoint(model, name, device=None):
    """Loads a saved {name}_best.pth into model in place. Handy for
    reloading weights for eval/explainability without retraining."""
    device = device or config_module.DEVICE
    path = os.path.join(config_module.MODEL_DIR, f'{name}_best.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"loaded {path}")
    return model


def load_history(name):
    path = os.path.join(config_module.HISTORY_DIR, f'{name}_history.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"no history at {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def _train_model_on_device(model, train_graphs, val_graphs, config, verbose, device):
    model = model.to(device)
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['batch_size'], shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {
        'train_auc': [], 'val_auc': [],
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate': [],
    }

    for epoch in range(config['epochs']):
        train_metrics, *_ = run_epoch(model, train_loader, criterion, optimizer, device=device)
        val_metrics, *_ = run_epoch(model, val_loader, criterion, device=device)
        scheduler.step(val_metrics['loss'])

        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == config['epochs'] - 1):
            print(f"epoch {epoch:3d} | train auc {train_metrics['auc']:.3f} | "
                  f"val auc {val_metrics['auc']:.3f} | best so far {best_val_auc:.3f}")

        if patience_counter >= config['early_stop_patience']:
            if verbose:
                print(f"no improvement for {config['early_stop_patience']} epochs, stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_auc


def evaluate_model(model, graphs, batch_size=32, device=None):
    device = device or next(model.parameters()).device  # wherever the model actually lives
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    metrics, preds, labels, probs = run_epoch(model, loader, nn.BCELoss(), device=device)

    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)
    
    metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
    metrics['predictions'] = preds
    metrics['true_labels'] = labels
    metrics['probabilities'] = probs

    return metrics


def cross_validate_gnn(model_fn, dataset_df, build_graphs_fn, n_folds=5, seed=42, config=None):
    """model_fn should return a fresh untrained model each call.
    Returns per-fold metric lists -- report mean +/- std, not any one
    fold alone."""
    from data_prep import cross_validation_splits
    set_seed(seed)

    fold_metrics = {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}

    for fold_idx, train_df, val_df, descriptor_cols, _ in cross_validation_splits(dataset_df, n_folds, seed):
        print(f"\n--- fold {fold_idx + 1}/{n_folds} ---")
        train_graphs = build_graphs_fn(train_df, descriptor_cols)
        val_graphs = build_graphs_fn(val_df, descriptor_cols)

        model = model_fn()
        model, _, _ = train_model(model, train_graphs, val_graphs, config=config, verbose=False)
        result = evaluate_model(model, val_graphs)

        for key in fold_metrics:
            fold_metrics[key].append(result[key])

        print(f"fold {fold_idx + 1}: acc={result['accuracy']:.3f} auc={result['auc']:.3f} f1={result['f1']:.3f}")

    print("\ncross-validation summary:")
    for key, values in fold_metrics.items():
        print(f"  {key}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    return fold_metrics