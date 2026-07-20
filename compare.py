"""
Comparison plots and tables for cv/train-final results: bar charts
with error bars across folds, and ROC curves from the held-out test
set.
"""

import math
import os
import pickle
import string

import matplotlib
try:
    get_ipython()
    _in_notebook = True
except NameError:
    _in_notebook = False
if not _in_notebook:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

ROMAN_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

# readable labels for internal keys; covers both naming schemes in use
# (cv_results uses 'gcn', train-final's saved files use 'gnn_model')
DISPLAY_NAMES = {
    'baseline_random_forest': 'Random Forest',
    'baseline_logistic_regression': 'Logistic Regression',
    'baseline_linear_svm': 'Linear SVM',
    'baseline_gradient_boosting': 'Gradient Boosting',
    'baseline_knn': 'k-NN',
    'baseline_mlp': 'MLP',
    'gcn': 'GCN',
    'gat': 'GAT',
    'gat_transformer': 'GAT-Transformer',
    'gnn_model': 'GCN',
    'gat_model': 'GAT',
    'gcn_transformer': 'GCN-Transformer',
    'gcn_transformer_model': 'GCN-Transformer',
    'gat_transformer_model': 'GAT-Transformer',
}


def comparison_table(cv_results):
    """cv_results is {model_name: {metric: [fold values]}}, from cv or
    reloaded via load_cv_summary. Returns one row per model, mean +/-
    std per metric, sorted by AUC."""
    rows = []
    for name, metrics in cv_results.items():
        row = {'model': DISPLAY_NAMES.get(name, name), 'type': 'baseline' if name.startswith('baseline_') else 'GNN'}
        for metric, values in metrics.items():
            values = [v for v in values if v is not None]
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if 'auc_mean' in df.columns:
        df = df.sort_values('auc_mean', ascending=False).reset_index(drop=True)
    return df


def load_cv_summary(path='test_results/cross_validation_summary.json'):
    """Reloads cross_validation_summary.json into the same shape
    cv_results has in memory."""
    import json
    with open(path) as f:
        summary = json.load(f)
    return {name: {metric: v['values'] for metric, v in metrics.items()} for name, metrics in summary.items()}


def plot_metric_comparison(cv_results, metric='auc', save_path=None, title=None):
    """Bar chart, one bar per model, error bars from std across folds;
    baselines and GNNs colored differently."""
    table = comparison_table(cv_results)
    table = table.dropna(subset=[f'{metric}_mean'])

    colors = ['#4C72B0' if t == 'baseline' else '#DD8452' for t in table['type']]

    plt.figure(figsize=(max(8, len(table) * 1.1), 5))
    plt.bar(table['model'], table[f'{metric}_mean'], yerr=table[f'{metric}_std'],
            color=colors, capsize=4, alpha=0.85)
    plt.ylabel(metric.upper() if metric == 'auc' else metric.capitalize())
    plt.title(title or f'{metric.upper() if metric == "auc" else metric.capitalize()} across models (mean \u00b1 std over folds)')
    plt.xticks(rotation=35, ha='right')

    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color='#4C72B0', label='classical baseline'),
                         Patch(color='#DD8452', label='GNN')])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    return table


def plot_all_metrics_comparison(cv_results, metrics=('accuracy', 'auc', 'precision', 'recall', 'f1'),
                                 output_path='figures/metrics_comparison.pdf', ncols=2):
    """Accuracy, AUC, Precision, Recall, F1 by default, 2 columns,
    labeled A-E. Baselines and GNNs keep their blue/orange coding
    within each panel."""
    table = comparison_table(cv_results)

    n = len(metrics)
    nrows = math.ceil(n / ncols)
    labels = string.ascii_uppercase

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, metric in enumerate(metrics):
        sub = table.dropna(subset=[f'{metric}_mean'])
        colors = ['#4C72B0' if t == 'baseline' else '#DD8452' for t in sub['type']]
        ax = axes[i]
        ax.bar(sub['model'], sub[f'{metric}_mean'], yerr=sub[f'{metric}_std'], color=colors, capsize=4, alpha=0.85)
        letter = labels[i] if i < len(labels) else str(i + 1)
        metric_label = metric.upper() if metric == 'auc' else metric.capitalize()
        ax.set_title(f'{letter}. {metric_label}')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['model'], rotation=35, ha='right')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, edgecolor='black', linewidth=1.0))

    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#4C72B0', label='classical baseline'),
                         Patch(color='#DD8452', label='GNN')],
               loc='upper right', bbox_to_anchor=(0.99, 0.99))

    fig.suptitle('Model comparison across metrics')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.show()
    plt.close()

    print(f"saved {n}-metric comparison ({labels[0]}-{labels[min(n, len(labels)) - 1]}) to {output_path}")
    return table


def plot_training_curves(model_name, history_dir='training_history', save_path=None):
    """Train vs. validation loss, accuracy, and AUC per epoch for one
    model, from {model_name}_history.pkl. A widening gap between the
    two lines indicates overfitting."""
    path = os.path.join(history_dir, f'{model_name}_history.pkl')
    if not os.path.exists(path):
        print(f"no {path} found — run train-final first (or check the model name)")
        return None

    with open(path, 'rb') as f:
        history = pickle.load(f)

    epochs = range(len(history['train_loss']))
    panels = [
        ('loss', 'train_loss', 'val_loss'),
        ('accuracy', 'train_acc', 'val_acc'),
        ('AUC', 'train_auc', 'val_auc'),
    ]
    # older saved histories (before accuracy tracking was added) won't
    # have train_acc/val_acc -- skip that panel rather than crash
    panels = [p for p in panels if p[1] in history and p[2] in history]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label='train')
        ax.plot(epochs, history[val_key], label='validation')
        ax.set_xlabel('epoch')
        ax.set_ylabel(label)
        ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}: {label}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    return history


def plot_all_training_curves(model_names=None, history_dir='training_history', save_dir='figures'):
    """Runs plot_training_curves for every GNN with a saved history;
    does not apply to the classical baselines."""
    model_names = model_names or ['gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model']
    for name in model_names:
        save_path = os.path.join(save_dir, f'{name}_training_curves.png') if save_dir else None
        plot_training_curves(name, history_dir=history_dir, save_path=save_path)


def save_training_curves_pdf(model_names=None, history_dir='training_history',
                              output_path='figures/training_curves.pdf'):
    """All GNNs' train/val curves in one PDF, one row per model,
    columns for loss/accuracy/AUC. Panels labeled A-i, A-ii, A-iii per
    row: letter identifies the model, numeral identifies the metric."""
    model_names = model_names or ['gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model']
    letters = string.ascii_uppercase

    available, histories = [], []
    for name in model_names:
        path = os.path.join(history_dir, f'{name}_history.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                histories.append(pickle.load(f))
            available.append(name)

    if not available:
        print(f"no *_history.pkl files found in {history_dir} — run train-final first")
        return None

    ncols = 3
    nrows = len(available)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes)
    if axes.shape != (nrows, ncols):
        axes = axes.reshape(nrows, ncols)

    for row, (name, history) in enumerate(zip(available, histories)):
        epochs = range(len(history['train_loss']))
        panels = [
            ('loss', 'train_loss', 'val_loss'),
            ('accuracy', 'train_acc', 'val_acc'),
            ('AUC', 'train_auc', 'val_auc'),
        ]
        panels = [p for p in panels if p[1] in history and p[2] in history]

        for col, (metric_label, train_key, val_key) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(epochs, history[train_key], label='train')
            ax.plot(epochs, history[val_key], label='validation')
            ax.set_xlabel('epoch')
            ax.set_ylabel(metric_label)
            roman = ROMAN_NUMERALS[col] if col < len(ROMAN_NUMERALS) else str(col + 1)
            ax.set_title(roman)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for col in range(len(panels), ncols):
            axes[row, col].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(top=0.94, hspace=0.5)

    # row letter headers, positioned from the actual rendered axes so
    # they stay correct regardless of layout
    for row in range(nrows):
        letter = letters[row] if row < len(letters) else str(row + 1)
        left_pos = axes[row, 0].get_position()
        right_pos = axes[row, ncols - 1].get_position()
        mid_x = (left_pos.x0 + right_pos.x1) / 2
        fig.text(mid_x, left_pos.y1 + 0.012, letter, ha='center', va='bottom',
                  fontsize=15, fontweight='bold')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.show()
    plt.close()

    print(f"saved training curves for {len(available)} models to {output_path}")
    return output_path


def plot_roc_curves(model_names=None, results_dir='test_results', save_path=None):
    """ROC curves from the held-out test set; needs the per-model
    {name}_results.pkl files train-final produces."""
    from models import BASELINE_REGISTRY
    model_names = model_names or (
        [f'baseline_{name}' for name in BASELINE_REGISTRY.keys()]
        + ['gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model']
    )

    plt.figure(figsize=(7, 6))
    plotted_any = False

    for name in model_names:
        path = os.path.join(results_dir, f'{name}_results.pkl')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            result = pickle.load(f)
        if 'true_labels' not in result or 'probabilities' not in result:
            continue

        fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
        roc_auc = auc(fpr, tpr)
        label = DISPLAY_NAMES.get(name, name)
        plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
        plotted_any = True

    if not plotted_any:
        print(f"no {{name}}_results.pkl files with probabilities found in {results_dir} — "
              f"run train-final first")
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves, held-out test set')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_confusion_matrix(model_name, results_dir='test_results', save_path=None):
    """Reads {model_name}_results.pkl and plots the confusion matrix as
    a heatmap."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    path = os.path.join(results_dir, f'{model_name}_results.pkl')
    if not os.path.exists(path):
        print(f"no {path} found — run train-final first")
        return None

    with open(path, 'rb') as f:
        result = pickle.load(f)

    cm = confusion_matrix(result['true_labels'], result['predictions'])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.title(f'{DISPLAY_NAMES.get(model_name, model_name)}: confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    return cm


def plot_balanced_confusion_matrix(model_name, results_dir='test_results', save_path=None, seed=42):
    """Supplementary diagnostic, not a replacement for
    plot_confusion_matrix. Subsamples the majority class in the
    existing predictions to match the minority count; nothing is
    retrained. Kept separate from the primary result since the real
    test set's natural imbalance is what determines whether the
    reported metrics reflect actual screening performance."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    path = os.path.join(results_dir, f'{model_name}_results.pkl')
    if not os.path.exists(path):
        print(f"no {path} found — run train-final first")
        return None

    with open(path, 'rb') as f:
        result = pickle.load(f)

    true_labels = np.array(result['true_labels'])
    predictions = np.array(result['predictions'])

    rng = np.random.default_rng(seed)
    class_0_idx = np.where(true_labels == 0)[0]
    class_1_idx = np.where(true_labels == 1)[0]
    n = min(len(class_0_idx), len(class_1_idx))

    if n == 0:
        print(f"{model_name}: one class has zero examples in the test set, can't balance")
        return None

    sampled_idx = np.concatenate([
        rng.choice(class_0_idx, size=n, replace=False),
        rng.choice(class_1_idx, size=n, replace=False),
    ])

    cm = confusion_matrix(true_labels[sampled_idx], predictions[sampled_idx])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.title(f'{DISPLAY_NAMES.get(model_name, model_name)}: confusion matrix\n(SUPPLEMENTARY -- balanced subsample, n={n} per class, not the primary result)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    return cm


def plot_all_confusion_matrices(model_names=None, results_dir='test_results',
                                 output_path='figures/confusion_matrices.pdf', ncols=2, balanced=False):
    """All confusion matrices in one PDF, 2-column grid by default,
    labeled A, B, C... Set balanced=True for the supplementary
    balanced-subsample version instead."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from models import BASELINE_REGISTRY

    model_names = model_names or (
        [f'baseline_{name}' for name in BASELINE_REGISTRY.keys()]
        + ['gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model']
    )

    available = [name for name in model_names
                 if os.path.exists(os.path.join(results_dir, f'{name}_results.pkl'))]

    if not available:
        print(f"no *_results.pkl files found in {results_dir} — run train-final first")
        return None

    n = len(available)
    nrows = math.ceil(n / ncols)
    labels = string.ascii_uppercase
    if n > len(labels):
        print(f"warning: {n} models but only {len(labels)} letters -- panels beyond Z will be unlabeled")

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    rng = np.random.default_rng(42)

    for i, name in enumerate(available):
        with open(os.path.join(results_dir, f'{name}_results.pkl'), 'rb') as f:
            result = pickle.load(f)

        true_labels = np.array(result['true_labels'])
        predictions = np.array(result['predictions'])

        if balanced:
            class_0_idx = np.where(true_labels == 0)[0]
            class_1_idx = np.where(true_labels == 1)[0]
            k = min(len(class_0_idx), len(class_1_idx))
            sampled_idx = np.concatenate([
                rng.choice(class_0_idx, size=k, replace=False),
                rng.choice(class_1_idx, size=k, replace=False),
            ])
            cm = confusion_matrix(true_labels[sampled_idx], predictions[sampled_idx])
        else:
            cm = confusion_matrix(true_labels, predictions)

        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if not balanced else 'Oranges',
                    xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'],
                    ax=ax, cbar=False)
        label = labels[i] if i < len(labels) else str(i + 1)
        ax.set_title(f'{label}. {DISPLAY_NAMES.get(name, name)}')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        # panel outline, drawn in axes coordinates since seaborn's
        # heatmap can otherwise cover the spines
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, edgecolor='black', linewidth=1.3))

    for j in range(len(available), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Confusion matrices' + (' (balanced subsample, supplementary)' if balanced else ''))
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.show()
    plt.close()

    print(f"saved {n} confusion matrices ({labels[0]}-{labels[min(n, len(labels)) - 1]}) to {output_path}")
    return output_path


def plot_all_balanced_confusion_matrices(model_names=None, results_dir='test_results', save_dir='figures', seed=42):
    """Supplementary balanced-subsample versions of every confusion
    matrix above."""
    from models import BASELINE_REGISTRY
    model_names = model_names or (
        [f'baseline_{name}' for name in BASELINE_REGISTRY.keys()]
        + ['gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model']
    )
    for name in model_names:
        path = os.path.join(results_dir, f'{name}_results.pkl')
        if not os.path.exists(path):
            continue
        save_path = os.path.join(save_dir, f'{name}_balanced_confusion_matrix.png') if save_dir else None
        plot_balanced_confusion_matrix(name, results_dir=results_dir, save_path=save_path, seed=seed)


def plot_tuning_results(results_df, model_name='model', save_path=None):
    """Bar chart of mean validation AUC per hyperparameter combination,
    sorted best first."""
    labels = [', '.join(f'{k}={v}' for k, v in row.items() if k not in ('mean_val_auc', 'std_val_auc'))
              for _, row in results_df.iterrows()]

    plt.figure(figsize=(max(6, len(results_df) * 1.3), 4.5))
    plt.bar(labels, results_df['mean_val_auc'], yerr=results_df['std_val_auc'], capsize=4,
            color='#55A868', alpha=0.85)
    plt.ylabel('mean validation AUC (CV inside 80% train)')
    plt.title(f'{DISPLAY_NAMES.get(model_name, model_name)}: hyperparameter tuning')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def save_feature_importance_pdf(importance_dfs, output_path='figures/feature_importance.pdf',
                                 ncols=2, top_n=10, feature_col='feature'):
    """Horizontal bar charts of feature importance, one panel per
    explainability method, 2 columns, labeled A, B, C...

    importance_dfs is {method_name: dataframe}, each with a
    feature_col column and an 'importance' column (feature_col=
    'feature' by default, or 'descriptor' for descriptor-level
    results)."""
    n = len(importance_dfs)
    if n == 0:
        print("no importance dataframes provided")
        return None

    nrows = math.ceil(n / ncols)
    labels = string.ascii_uppercase

    # extends the palette used elsewhere so each panel gets a distinct color
    palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
               '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, (method_name, df) in enumerate(importance_dfs.items()):
        top = df.sort_values('importance', ascending=False).head(top_n)
        # reversed so the top feature plots at the top -- barh draws bottom-up
        top = top.iloc[::-1]

        ax = axes[i]
        ax.barh(top[feature_col], top['importance'], color=palette[i % len(palette)], alpha=0.85)
        letter = labels[i] if i < len(labels) else str(i + 1)
        ax.set_title(f'{letter}. {method_name}')
        ax.set_xlabel('importance')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, edgecolor='black', linewidth=1.0))

    for j in range(len(importance_dfs), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.show()
    plt.close()

    print(f"saved {n}-method feature importance comparison ({labels[0]}-{labels[min(n, len(labels)) - 1]}) to {output_path}")
    return output_path


def save_tuning_results_pdf(tuning_results, output_path='figures/tuning_results.pdf', ncols=2,
                             results_dir='test_results',
                             model_names=('gnn_model', 'gat_model', 'gcn_transformer_model', 'gat_transformer_model')):
    """All hyperparameter-tuning bar charts in one PDF, 2 columns,
    labeled A, B, C...

    tuning_results is {model_name: results_df}. Pass None to load each
    {name}_tuning_results.csv from results_dir instead."""
    if tuning_results is None:
        tuning_results = {}
        for name in model_names:
            path = os.path.join(results_dir, f'{name}_tuning_results.csv')
            if os.path.exists(path):
                tuning_results[name] = pd.read_csv(path)

    if not tuning_results:
        print(f"no tuning results found (dict was empty, and no *_tuning_results.csv in {results_dir}) "
              f"— run train-final first")
        return None

    n = len(tuning_results)
    nrows = math.ceil(n / ncols)
    labels = string.ascii_uppercase

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, (name, results_df) in enumerate(tuning_results.items()):
        combo_labels = [', '.join(f'{k}={v}' for k, v in row.items() if k not in ('mean_val_auc', 'std_val_auc'))
                        for _, row in results_df.iterrows()]
        ax = axes[i]
        ax.bar(combo_labels, results_df['mean_val_auc'], yerr=results_df['std_val_auc'], capsize=4,
               color='#55A868', alpha=0.85)
        ax.set_ylabel('mean validation AUC')
        letter = labels[i] if i < len(labels) else str(i + 1)
        ax.set_title(f'{letter}. {DISPLAY_NAMES.get(name, name)}')
        ax.tick_params(axis='x', rotation=20)
        for tick_label in ax.get_xticklabels():
            tick_label.set_ha('right')

    for j in range(len(tuning_results), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Hyperparameter tuning results, all GNNs')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.show()
    plt.close()

    print(f"saved tuning results for {n} models to {output_path}")
    return output_path