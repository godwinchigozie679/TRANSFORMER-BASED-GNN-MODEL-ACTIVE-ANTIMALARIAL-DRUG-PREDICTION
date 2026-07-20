import os
import random

import numpy as np
import torch

try:
    import tomllib  # stdlib 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli on older python

from rdkit import RDLogger

# suppresses RDKit's deprecation warnings; parsing errors still show
RDLogger.DisableLog('rdApp.*')


def _load_toml(path='config.toml'):
    if not os.path.exists(path):
        print(f"no {path} found, falling back to hardcoded defaults")
        return {}
    with open(path, 'rb') as f:
        return tomllib.load(f)


_toml = _load_toml()

# falls back to these defaults if config.toml or a given key is missing
SEED = _toml.get('run', {}).get('seed', 42)
RAW_DATA_PATH = _toml.get('data', {}).get('path', 'main_data_2/Drug_activity.csv')
CACHE_PATH = _toml.get('data', {}).get('cache_path', 'enhanced_drug_data_with_descriptors.csv')
USE_CACHE = _toml.get('data', {}).get('use_cache', True)
GRAPH_CACHE_PATH = _toml.get('data', {}).get('graph_cache_path', 'molecule_graphs.pt')
BALANCE_FULL_DATASET = _toml.get('data', {}).get('balance_full_dataset', True)
BALANCED_PATH = _toml.get('data', {}).get('balanced_path', 'main_data_2/Drug_activity_balanced.csv')
BALANCED_CACHE_PATH = _toml.get('data', {}).get('balanced_cache_path', 'balanced_descriptors.csv')
BALANCED_GRAPH_CACHE_PATH = _toml.get('data', {}).get('balanced_graph_cache_path', 'balanced_molecule_graphs.pt')
CORRELATION_THRESHOLD = _toml.get('crossval', {}).get('correlation_threshold', 0.85)
N_FOLDS = _toml.get('crossval', {}).get('folds', 5)
CV_MODELS = _toml.get('crossval', {}).get('models') or None  # empty list -> run all
BASELINE_MODELS = _toml.get('crossval', {}).get('baseline_models') or None  # empty -> run all five
SKIP_BASELINE = _toml.get('crossval', {}).get('skip_baseline', False)
EXPLAIN_N_MOLECULES = _toml.get('explain', {}).get('n_molecules', 20)
TUNE_TEST_SIZE = _toml.get('tuning', {}).get('test_size', 0.2)
TUNE_FOLDS = _toml.get('tuning', {}).get('folds', 5)
TUNE_EPOCHS = _toml.get('tuning', {}).get('epochs', 40)
TUNE_PATIENCE = _toml.get('tuning', {}).get('patience', 8)
VISUALIZE_N_MOLECULES = _toml.get('visualize', {}).get('n_molecules', 3)

OUTPUT_DIR = _toml.get('paths', {}).get('output_dir', 'test_results')
MODEL_DIR = _toml.get('paths', {}).get('model_dir', 'trained_models')
HISTORY_DIR = _toml.get('paths', {}).get('history_dir', 'training_history')
FIGURE_DIR = _toml.get('paths', {}).get('figure_dir', 'figures')
for d in (OUTPUT_DIR, MODEL_DIR, HISTORY_DIR, FIGURE_DIR):
    os.makedirs(d, exist_ok=True)

_train_toml = _toml.get('training', {})
TRAIN_CONFIG = {
    'epochs': _train_toml.get('epochs', 200),
    'initial_lr': _train_toml.get('initial_lr', 0.001),
    'weight_decay': _train_toml.get('weight_decay', 1e-4),
    'early_stop_patience': _train_toml.get('early_stop_patience', 25),
    'batch_size': _train_toml.get('batch_size', 32),
    'hidden_dim': _train_toml.get('hidden_dim', 128),
    'dropout': _train_toml.get('dropout', 0.2),
}


def _select_device(preference='auto'):
    """'auto' picks CUDA if available, otherwise CPU; MPS is skipped
    even on Apple Silicon, since torch_geometric's scatter ops hit an
    MPS-unsupported operator that PYTORCH_ENABLE_MPS_FALLBACK does not
    cover. preference='mps' can still be set explicitly in
    config.toml."""
    if preference == 'cpu':
        return torch.device('cpu')
    if preference == 'cuda':
        return torch.device('cuda')
    if preference == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # covers some ops, not the one that mattered here
        return torch.device('mps')
    # cuda if available, else cpu; mps skipped, see docstring
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE_PREFERENCE = _toml.get('run', {}).get('device', 'auto')
DEVICE = _select_device(DEVICE_PREFERENCE)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # not bit-for-bit deterministic on GPU; use
    # torch.use_deterministic_algorithms(True) for that, at a speed cost


def reload_config(path='config.toml'):
    """Re-reads config.toml without restarting Python. Does not affect
    TRAIN_CONFIG values already in use by an in-progress training
    run."""
    global _toml, SEED, RAW_DATA_PATH, CACHE_PATH, USE_CACHE, GRAPH_CACHE_PATH, CORRELATION_THRESHOLD, N_FOLDS
    global CV_MODELS, BASELINE_MODELS, SKIP_BASELINE, EXPLAIN_N_MOLECULES, VISUALIZE_N_MOLECULES, TRAIN_CONFIG
    global BALANCE_FULL_DATASET, BALANCED_PATH, BALANCED_CACHE_PATH, BALANCED_GRAPH_CACHE_PATH
    global TUNE_TEST_SIZE, TUNE_FOLDS, TUNE_EPOCHS, TUNE_PATIENCE
    global DEVICE_PREFERENCE, DEVICE

    _toml = _load_toml(path)

    SEED = _toml.get('run', {}).get('seed', 42)
    RAW_DATA_PATH = _toml.get('data', {}).get('path', 'main_data_2/Drug_activity.csv')
    CACHE_PATH = _toml.get('data', {}).get('cache_path', 'enhanced_drug_data_with_descriptors.csv')
    USE_CACHE = _toml.get('data', {}).get('use_cache', True)
    GRAPH_CACHE_PATH = _toml.get('data', {}).get('graph_cache_path', 'molecule_graphs.pt')
    BALANCE_FULL_DATASET = _toml.get('data', {}).get('balance_full_dataset', True)
    BALANCED_PATH = _toml.get('data', {}).get('balanced_path', 'main_data_2/Drug_activity_balanced.csv')
    BALANCED_CACHE_PATH = _toml.get('data', {}).get('balanced_cache_path', 'balanced_descriptors.csv')
    BALANCED_GRAPH_CACHE_PATH = _toml.get('data', {}).get('balanced_graph_cache_path', 'balanced_molecule_graphs.pt')
    CORRELATION_THRESHOLD = _toml.get('crossval', {}).get('correlation_threshold', 0.85)
    N_FOLDS = _toml.get('crossval', {}).get('folds', 5)
    CV_MODELS = _toml.get('crossval', {}).get('models') or None
    BASELINE_MODELS = _toml.get('crossval', {}).get('baseline_models') or None
    SKIP_BASELINE = _toml.get('crossval', {}).get('skip_baseline', False)
    EXPLAIN_N_MOLECULES = _toml.get('explain', {}).get('n_molecules', 20)
    TUNE_TEST_SIZE = _toml.get('tuning', {}).get('test_size', 0.2)
    TUNE_FOLDS = _toml.get('tuning', {}).get('folds', 5)
    TUNE_EPOCHS = _toml.get('tuning', {}).get('epochs', 40)
    TUNE_PATIENCE = _toml.get('tuning', {}).get('patience', 8)
    VISUALIZE_N_MOLECULES = _toml.get('visualize', {}).get('n_molecules', 3)
    DEVICE_PREFERENCE = _toml.get('run', {}).get('device', 'auto')
    DEVICE = _select_device(DEVICE_PREFERENCE)

    _train_toml = _toml.get('training', {})
    TRAIN_CONFIG = {
        'epochs': _train_toml.get('epochs', 200),
        'initial_lr': _train_toml.get('initial_lr', 0.001),
        'weight_decay': _train_toml.get('weight_decay', 1e-4),
        'early_stop_patience': _train_toml.get('early_stop_patience', 25),
        'batch_size': _train_toml.get('batch_size', 32),
        'hidden_dim': _train_toml.get('hidden_dim', 128),
        'dropout': _train_toml.get('dropout', 0.2),
    }

    print("config reloaded from", path)