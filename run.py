#!/usr/bin/env python3
"""
Command-line tool for the antimalarial activity pipeline.

Usage:
    python run.py eda                     dataset overview, distributions, correlation heatmaps
    python run.py visualize               draw sample/extreme molecules
    python run.py cv                      cross-validate all four GNNs + the classical baselines
    python run.py cv --folds 10           more folds
    python run.py cv --models gcn         just one model, skip the others
    python run.py stats                   descriptor statistics, FDR corrected
    python run.py train-final             80/20 split, tune, train, and save every architecture
    python run.py compare                 plots from whatever cv/train-final already saved
    python run.py explain                 run GNNExplainer, IG, Feature Ablation, Saliency, and SHAP
    python run.py balance-data            save a class-balanced copy of the dataset
    python run.py all                     eda, visualize, cv, stats, train-final, compare, explain
    python run.py --recompute-descriptors cv    ignore the cached CSV, recompute from scratch

--data, --seed, and --recompute-descriptors go BEFORE the subcommand
(cv/eda/stats/etc.), same as any argparse tool with subcommands --
`run.py --recompute-descriptors cv`, not `run.py cv --recompute-descriptors`.

Descriptors get cached after the first run, so later runs load from
there instead of recomputing 104 descriptors across the whole dataset
again. Delete the cache file, or pass --recompute-descriptors, if your
source data changed.

Expects main_data_2/Drug_activity.csv with Smiles and Labels columns,
unless you point --data somewhere else.
"""

import argparse
import json
import os
import pickle

import pandas as pd

import config
from config import set_seed
import data_prep
from data_prep import build_full_dataset, make_holdout_split, save_balanced_dataset
from graphs import build_graph_dataset
from models import GCNBaseline, GATNet, GCNTransformerNet, GATTransformerNet, BASELINE_REGISTRY
from train import cross_validate_gnn, train_model, evaluate_model
from baseline import cross_validate_all_baselines
from stats_analysis import compare_descriptors_by_activity
from explain import (
    explain_molecule_set, integrated_gradients_on_batch, feature_ablation_on_batch, saliency_on_batch,
    shap_on_batch, integrated_gradients_descriptors, feature_ablation_descriptors, saliency_descriptors,
    gnnexplainer_descriptors, shap_descriptors,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from torch_geometric.loader import DataLoader

MODEL_REGISTRY = {
    'gcn': lambda: GCNBaseline(),
    'gat': lambda: GATNet(),
    'gcn_transformer': lambda: GCNTransformerNet(),
    'gat_transformer': lambda: GATTransformerNet(),
}

# matches the checkpoint/result filenames these get saved under
CHECKPOINT_NAMES = {
    'gcn': 'gnn_model',
    'gat': 'gat_model',
    'gcn_transformer': 'gcn_transformer_model',
    'gat_transformer': 'gat_transformer_model',
}


def cmd_cv(args, dataset_df):
    results = {}
    skip_baseline = getattr(args, 'skip_baseline', False)
    models_to_run = getattr(args, 'models', None) or list(MODEL_REGISTRY.keys())
    baselines_to_run = getattr(args, 'baselines', None) or config.BASELINE_MODELS

    if not skip_baseline:
        baseline_results = cross_validate_all_baselines(
            dataset_df, n_folds=args.folds, seed=args.seed, model_names=baselines_to_run
        )
        for name, metrics in baseline_results.items():
            results[f'baseline_{name}'] = metrics

    for name in models_to_run:
        if name not in MODEL_REGISTRY:
            print(f"skipping unknown model '{name}', options are: {list(MODEL_REGISTRY.keys())}")
            continue
        print(f"\n=== {name} ===")
        results[name] = cross_validate_gnn(
            MODEL_REGISTRY[name], dataset_df, build_graph_dataset, n_folds=args.folds, seed=args.seed
        )

    summary = {
        name: {metric: {'mean': float(sum(v) / len(v)) if v else None, 'values': [float(x) for x in v]}
               for metric, v in metrics.items()}
        for name, metrics in results.items()
    }
    out_path = f'{config.OUTPUT_DIR}/cross_validation_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")


def _evaluate_sklearn_model(model, X_val, y_val):
    """Same shape of result dict evaluate_model() returns for the GNNs,
    so downstream code (ROC plotting, comparison tables) doesn't need
    to care whether a model came from sklearn or torch."""
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    return {
        'accuracy': accuracy_score(y_val, preds),
        'balanced_accuracy': balanced_accuracy_score(y_val, preds),
        'auc': roc_auc_score(y_val, probs),
        'precision': precision_score(y_val, preds, zero_division=0),
        'recall': recall_score(y_val, preds, zero_division=0),
        'f1': f1_score(y_val, preds, zero_division=0),
        'predictions': preds,
        'true_labels': y_val,
        'probabilities': probs,
    }


def cmd_train_final(args, dataset_df):
    """80% train / 20% test outer split, 5-fold CV inside the 80% for
    tuning (GNNs only -- baselines fit directly, no tuning needed),
    then one final model per architecture trained on the full 80% and
    evaluated once on the untouched 20%. Saves everything: checkpoints,
    training history, tuning results, final test-set predictions.

    This is what you cite as reported test performance per
    architecture. The cross-validated numbers from `cv` are a separate,
    complementary claim about how robust the approach is across
    splits -- report both, and say which is which."""
    import tune

    outer_train_df, outer_test_df = data_prep.outer_split(dataset_df, test_size=config.TUNE_TEST_SIZE, seed=args.seed)
    print(f"outer split: {len(outer_train_df)} train (80%), {len(outer_test_df)} test (20%, held out until final eval)")

    results = {}

    print("\n--- classical baselines (fit directly on the 80%, no tuning) ---")
    train_df, test_df, descriptor_cols, _ = data_prep.prepare_two_way_split(outer_train_df, outer_test_df, args.seed)
    X_train, y_train = train_df[descriptor_cols].values, train_df['Labels'].values
    X_test, y_test = test_df[descriptor_cols].values, test_df['Labels'].values

    for name, build_model in BASELINE_REGISTRY.items():
        print(f"training {name}")
        model = build_model(args.seed)
        if hasattr(model, 'n_neighbors'):
            model.n_neighbors = max(1, min(model.n_neighbors, len(X_train) - 1))

        try:
            model.fit(X_train, y_train)
            result = _evaluate_sklearn_model(model, X_test, y_test)
        except Exception as e:
            print(f"  {name} failed ({e}), skipping")
            continue

        results[f'baseline_{name}'] = result
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        result_path = os.path.join(config.OUTPUT_DIR, f'baseline_{name}_results.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  test acc={result['accuracy']:.3f} auc={result['auc']:.3f} f1={result['f1']:.3f} -> {result_path}")

    print("\n--- GNNs: 5-fold CV inside the 80% for tuning, then one final fit + one final test ---")
    for key, model_fn in MODEL_REGISTRY.items():
        name = CHECKPOINT_NAMES[key]
        model_class = type(model_fn())  # grabs the class itself, no need for a separate class registry

        print(f"\n=== {name}: tuning (5-fold CV inside the 80% train) ===")
        tuning_results_df, best_params = tune.tune_hyperparameters(
            model_class, outer_train_df, n_folds=args.tune_folds, seed=args.seed,
            tuning_epochs=args.tune_epochs, tuning_patience=args.tune_patience,
        )
        print(f"best params for {name}: {best_params}")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        tuning_path = os.path.join(config.OUTPUT_DIR, f'{name}_tuning_results.csv')
        tuning_results_df.to_csv(tuning_path, index=False)
        print(f"saved tuning results to {tuning_path}")

        print(f"\n=== {name}: final fit on full 80%, one evaluation on the 20% test ===")
        model, history, result = tune.train_with_best_params(
            model_class, best_params, outer_train_df, outer_test_df,
            seed=args.seed, save_as=name,
        )
        results[key] = result

        result_path = os.path.join(config.OUTPUT_DIR, f'{name}_results.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"{name}: test acc={result['accuracy']:.3f} auc={result['auc']:.3f} f1={result['f1']:.3f}")
        print(f"saved results to {result_path}")

    return results


def cmd_compare(args, dataset_df):
    """Doesn't touch dataset_df at all -- just reads whatever cv and
    train-final already wrote to disk and turns it into plots. Run
    those first if this comes up empty."""
    import compare

    cv_summary_path = f'{config.OUTPUT_DIR}/cross_validation_summary.json'
    if os.path.exists(cv_summary_path):
        print(f"loading {cv_summary_path}")
        cv_results = compare.load_cv_summary(cv_summary_path)
        table = compare.plot_all_metrics_comparison(cv_results, output_path=f'{config.FIGURE_DIR}/metrics_comparison.pdf')
        print(table.to_string(index=False))
        table.to_csv(f'{config.OUTPUT_DIR}/model_comparison_table.csv', index=False)
        print(f"\nwrote {config.OUTPUT_DIR}/model_comparison_table.csv")
    else:
        print(f"no {cv_summary_path} found -- run `cv` first")

    print("\nplotting ROC curves from train-final results (if any)")
    compare.plot_roc_curves(results_dir=config.OUTPUT_DIR, save_path=f'{config.FIGURE_DIR}/roc_curves.png')

    print("\nplotting confusion matrices as one labeled PDF grid (primary + balanced supplementary)")
    compare.plot_all_confusion_matrices(results_dir=config.OUTPUT_DIR,
                                         output_path=f'{config.FIGURE_DIR}/confusion_matrices.pdf')
    compare.plot_all_confusion_matrices(results_dir=config.OUTPUT_DIR,
                                         output_path=f'{config.FIGURE_DIR}/confusion_matrices_balanced.pdf', balanced=True)

    print("\nplotting training curves as one labeled PDF grid")
    compare.save_training_curves_pdf(history_dir=config.HISTORY_DIR,
                                      output_path=f'{config.FIGURE_DIR}/training_curves.pdf')

    print("\nplotting hyperparameter tuning results as one labeled PDF grid")
    compare.save_tuning_results_pdf(tuning_results=None, results_dir=config.OUTPUT_DIR,
                                     output_path=f'{config.FIGURE_DIR}/tuning_results.pdf')

    print("\nplotting explainability results (atomic vs. molecular, one PDF per method, if saved)")
    _plot_explainability_pdfs()


def _plot_explainability_pdfs():
    """Builds an atomic-vs-molecular PDF for each explainability method
    that has both halves saved to disk -- exactly the pairing the
    notebook's explainability section produces, just reloaded here from
    the CSVs `explain` already wrote instead of from in-memory
    dataframes. Skips a method quietly if either half is missing."""
    import compare

    method_files = {
        'GNNExplainer': ('gnnexplainer_importance.csv', 'descriptor_gnnexplainer_importance.csv'),
        'Integrated Gradients': ('integrated_gradients_importance.csv', 'descriptor_integrated_gradients_importance.csv'),
        'Feature Ablation': ('feature_ablation_importance.csv', 'descriptor_feature_ablation_importance.csv'),
        'Saliency': ('saliency_importance.csv', 'descriptor_saliency_importance.csv'),
        'SHAP': ('shap_importance.csv', 'descriptor_shap_importance.csv'),
    }

    for method_name, (atomic_file, molecular_file) in method_files.items():
        atomic_path = os.path.join(config.OUTPUT_DIR, atomic_file)
        molecular_path = os.path.join(config.OUTPUT_DIR, molecular_file)
        if not (os.path.exists(atomic_path) and os.path.exists(molecular_path)):
            continue

        atomic_df = pd.read_csv(atomic_path).rename(columns={'feature': 'name'})
        molecular_df = pd.read_csv(molecular_path).rename(columns={'descriptor': 'name'})

        slug = method_name.lower().replace(' ', '_')
        compare.save_feature_importance_pdf(
            {'Atomic': atomic_df, 'Molecular': molecular_df},
            output_path=f'{config.FIGURE_DIR}/{slug}_atomic_vs_molecular.pdf',
            feature_col='name',
        )


def cmd_eda(args, dataset_df):
    import eda
    descriptor_cols = [c for c in dataset_df.columns if c not in ('Smiles', 'Labels')]
    eda.run_full_eda(dataset_df, descriptor_cols, output_dir=config.FIGURE_DIR)


def cmd_visualize(args, dataset_df):
    import visualize
    n = getattr(args, 'n_molecules', None) or config.VISUALIZE_N_MOLECULES

    print(f"drawing {n} sample molecules")
    visualize.draw_sample_molecules(dataset_df, n=n, save_path=f'{config.FIGURE_DIR}/sample_molecules.png')

    print("\ndrawing extreme molecules (highest/lowest MW, highest LogP)")
    visualize.draw_extreme_molecules(dataset_df, save_path=f'{config.FIGURE_DIR}/extreme_molecules.png')

    print(f"\nindividual view of {n} molecules")
    visualize.view_individual_molecules(dataset_df, n=n)


def cmd_stats(args, dataset_df):
    descriptor_cols = [c for c in dataset_df.columns if c not in ('Smiles', 'Labels')]
    stats_df = compare_descriptors_by_activity(dataset_df, descriptor_cols)
    out_path = f'{config.OUTPUT_DIR}/descriptor_statistics.csv'
    stats_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def cmd_explain(args, dataset_df):
    """Reuses the gat_transformer_model checkpoint saved by train-final
    
    Runs all five explainability methods -- GNNExplainer, Integrated
    Gradients, Feature Ablation, Saliency, SHAP -- on both branches the
    model has: the atom-level graph structure and the molecular
    descriptor branch."""
    import torch
    from train import load_checkpoint

    explain_n = getattr(args, 'explain_n', 20)
    checkpoint_path = os.path.join(config.MODEL_DIR, 'gat_transformer_model_best.pth')

    if os.path.exists(checkpoint_path):
        import tune
        outer_train_df, outer_test_df = data_prep.outer_split(dataset_df, test_size=config.TUNE_TEST_SIZE, seed=args.seed)
        train_df, val_df, test_df, descriptor_cols, _ = tune.prepare_final_split(outer_train_df, outer_test_df, args.seed)
    else:
        train_df, val_df, test_df, descriptor_cols, _ = make_holdout_split(dataset_df, args.seed)

    train_graphs = build_graph_dataset(train_df, descriptor_cols)
    val_graphs = build_graph_dataset(val_df, descriptor_cols)
    test_graphs = build_graph_dataset(test_df, descriptor_cols)

    model = GATTransformerNet()
    checkpoint_path = os.path.join(config.MODEL_DIR, 'gat_transformer_model_best.pth')
    params_path = os.path.join(config.MODEL_DIR, 'gat_transformer_model_best_params.json')

    if os.path.exists(checkpoint_path):
        # checkpoint's actual layer shapes depend on whatever
        # hyperparameters tuning picked -- reconstruct with those, not
        # defaults, or load_state_dict fails with a shape mismatch
        if os.path.exists(params_path):
            with open(params_path) as f:
                saved_params = json.load(f)
            print(f"reconstructing model with tuned params from {params_path}: {saved_params}")
            model = GATTransformerNet(**saved_params)
        else:
            print(f"no {params_path} found (older checkpoint?) -- using default architecture, "
                  f"this may fail to load if tuning picked non-default hyperparameters")

        print(f"loading saved model from {checkpoint_path} (from train-final)")
        # descriptor-encoder layer builds itself on first forward pass,
        # so it needs one before load_state_dict has shapes to match against
        dummy_n = min(4, len(test_graphs))
        dummy_batch = next(iter(DataLoader(test_graphs[:dummy_n], batch_size=dummy_n))).to(config.DEVICE)
        model = model.to(config.DEVICE)
        with torch.no_grad():
            _ = model(dummy_batch)
        model = load_checkpoint(model, 'gat_transformer_model')
    else:
        print(f"no {checkpoint_path} found -- run train-final first for the actual tuned model. "
              f"Training a quick one now just so explain has something to work with.")
        model, _, _ = train_model(model, train_graphs, val_graphs, verbose=True)

    model_device = next(model.parameters()).device  # wherever it actually landed (may have fallen back from mps to cpu)

    test_metrics = evaluate_model(model, test_graphs, device=model_device)
    print(f"\nholdout test: acc={test_metrics['accuracy']:.3f} auc={test_metrics['auc']:.3f} f1={test_metrics['f1']:.3f}")

    print("\nGNNExplainer, aggregated over a sample of test molecules:")
    importance_df = explain_molecule_set(model, test_graphs, model_device, max_molecules=explain_n)
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv(f'{config.OUTPUT_DIR}/gnnexplainer_importance.csv', index=False)

    print("\nIntegrated Gradients on the same molecules, real graph structure:")
    batch = next(iter(DataLoader(test_graphs[:explain_n], batch_size=explain_n)))
    batch = batch.to(model_device)
    ig_df = integrated_gradients_on_batch(model, batch)
    print(ig_df.head(10).to_string(index=False))
    ig_df.to_csv(f'{config.OUTPUT_DIR}/integrated_gradients_importance.csv', index=False)

    print("\nFeature Ablation on the same molecules, real graph structure:")
    fa_df = feature_ablation_on_batch(model, batch)
    print(fa_df.head(10).to_string(index=False))
    fa_df.to_csv(f'{config.OUTPUT_DIR}/feature_ablation_importance.csv', index=False)

    print("\nSaliency on the same molecules, real graph structure:")
    sal_df = saliency_on_batch(model, batch)
    print(sal_df.head(10).to_string(index=False))
    sal_df.to_csv(f'{config.OUTPUT_DIR}/saliency_importance.csv', index=False)

    print("\nSHAP over atom feature dimensions, real graph structure (slower -- real per-molecule perturbation sampling):")
    shap_df = shap_on_batch(model, batch)
    print(shap_df.head(10).to_string(index=False))
    shap_df.to_csv(f'{config.OUTPUT_DIR}/shap_importance.csv', index=False)

    print("\n--- descriptor-level explainability (the RDKit descriptor branch, not the graph branch) ---")

    print("\nIntegrated Gradients on molecular descriptors:")
    desc_ig_df = integrated_gradients_descriptors(model, batch, descriptor_cols)
    print(desc_ig_df.head(10).to_string(index=False))
    desc_ig_df.to_csv(f'{config.OUTPUT_DIR}/descriptor_integrated_gradients_importance.csv', index=False)

    print("\nFeature Ablation on molecular descriptors:")
    desc_fa_df = feature_ablation_descriptors(model, batch, descriptor_cols)
    print(desc_fa_df.head(10).to_string(index=False))
    desc_fa_df.to_csv(f'{config.OUTPUT_DIR}/descriptor_feature_ablation_importance.csv', index=False)

    print("\nSaliency on molecular descriptors:")
    desc_sal_df = saliency_descriptors(model, batch, descriptor_cols)
    print(desc_sal_df.head(10).to_string(index=False))
    desc_sal_df.to_csv(f'{config.OUTPUT_DIR}/descriptor_saliency_importance.csv', index=False)

    print("\nGNNExplainer-style mask learning on molecular descriptors (real per-molecule optimization, slower than the gradient-based methods):")
    desc_gnnexplainer_df = gnnexplainer_descriptors(model, batch, descriptor_cols)
    print(desc_gnnexplainer_df.head(10).to_string(index=False))
    desc_gnnexplainer_df.to_csv(f'{config.OUTPUT_DIR}/descriptor_gnnexplainer_importance.csv', index=False)

    print("\nSHAP on molecular descriptors (slower than the others -- real per-molecule perturbation sampling):")
    desc_shap_df = shap_descriptors(model, batch, descriptor_cols)
    print(desc_shap_df.head(10).to_string(index=False))
    desc_shap_df.to_csv(f'{config.OUTPUT_DIR}/descriptor_shap_importance.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=config.RAW_DATA_PATH)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--recompute-descriptors', action='store_true',
                         help=f'ignore any cached {config.CACHE_PATH} and recompute from --data')

    subparsers = parser.add_subparsers(dest='command', required=True)

    p_cv = subparsers.add_parser('cv', help='cross-validate models')
    p_cv.add_argument('--folds', type=int, default=config.N_FOLDS)
    p_cv.add_argument('--models', nargs='*', choices=list(MODEL_REGISTRY.keys()),
                       help='subset of models to run, default is all four')
    p_cv.add_argument('--baselines', nargs='*', choices=list(BASELINE_REGISTRY.keys()),
                       help='subset of classical baselines to run, default is all five')
    p_cv.add_argument('--skip-baseline', action='store_true')

    subparsers.add_parser('eda', help='dataset overview, distributions, correlation heatmaps')

    p_viz = subparsers.add_parser('visualize', help='draw sample/extreme molecules')
    p_viz.add_argument('--n-molecules', type=int, default=None, help='overrides config.toml [visualize] n_molecules')

    subparsers.add_parser('stats', help='descriptor statistics, FDR corrected')

    p_train_final = subparsers.add_parser(
        'train-final', help='80/20 split, 5-fold CV tuning inside the 80%% for GNNs, one final eval on the 20%%'
    )
    p_train_final.add_argument('--tune-folds', type=int, default=config.TUNE_FOLDS, help='CV folds used for tuning, inside the 80%% train portion')
    p_train_final.add_argument('--tune-epochs', type=int, default=config.TUNE_EPOCHS, help='epoch cap during tuning (kept lower than the final training run for speed)')
    p_train_final.add_argument('--tune-patience', type=int, default=config.TUNE_PATIENCE, help='early-stop patience during tuning')

    subparsers.add_parser('compare', help='plots comparing all models from saved cv/train-final/explain results')

    p_explain = subparsers.add_parser('explain', help='explain the tuned model with all five methods, both branches')
    p_explain.add_argument('--explain-n', type=int, default=20, help='how many molecules to explain')

    p_all = subparsers.add_parser('all', help='run eda, visualize, cv, stats, train-final, compare, and explain in sequence')
    p_all.add_argument('--folds', type=int, default=config.N_FOLDS)
    p_all.add_argument('--explain-n', type=int, default=20)
    p_all.add_argument('--tune-folds', type=int, default=config.TUNE_FOLDS)
    p_all.add_argument('--tune-epochs', type=int, default=config.TUNE_EPOCHS)
    p_all.add_argument('--tune-patience', type=int, default=config.TUNE_PATIENCE)

    p_balance = subparsers.add_parser(
        'balance-data',
        help='save a class-balanced copy of the whole dataset -- see README for the tradeoff this creates '
             '(everything downstream trains AND tests on the balanced distribution, not the real one)'
    )
    p_balance.add_argument('--output', default='main_data_2/Drug_activity_balanced.csv',
                            help='where to save the balanced CSV')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.command == 'balance-data':
        save_balanced_dataset(args.data, args.output, seed=args.seed)
        return

    data_path = args.data
    cache_path = config.CACHE_PATH
    graph_cache_path = config.GRAPH_CACHE_PATH

    if config.BALANCE_FULL_DATASET:
        # balance the whole dataset once, then use that as the actual
        # source for everything below -- every split downstream (train,
        # val, test, CV folds) ends up drawn from this already-balanced
        # data, so models train AND test on the balanced distribution,
        # not the real one. Distinct cache filenames so this never
        # silently reuses a cache built from the real, imbalanced data.
        if args.recompute_descriptors or not os.path.exists(config.BALANCED_PATH):
            save_balanced_dataset(data_path, config.BALANCED_PATH, seed=args.seed)
        data_path = config.BALANCED_PATH
        cache_path = config.BALANCED_CACHE_PATH
        graph_cache_path = config.BALANCED_GRAPH_CACHE_PATH
        config.GRAPH_CACHE_PATH = graph_cache_path  # graphs.py reads this directly, not passed as a param

    print(f"loading dataset (source: {data_path}, cache: {cache_path})")
    dataset_df = build_full_dataset(
        data_path,
        cache_path=cache_path,
        use_cache=config.USE_CACHE,
        force_recompute=args.recompute_descriptors,
    )

    if args.command == 'cv':
        cmd_cv(args, dataset_df)
    elif args.command == 'eda':
        cmd_eda(args, dataset_df)
    elif args.command == 'visualize':
        cmd_visualize(args, dataset_df)
    elif args.command == 'stats':
        cmd_stats(args, dataset_df)
    elif args.command == 'train-final':
        cmd_train_final(args, dataset_df)
    elif args.command == 'compare':
        cmd_compare(args, dataset_df)
    elif args.command == 'explain':
        cmd_explain(args, dataset_df)
    elif args.command == 'all':
        cmd_eda(args, dataset_df)
        cmd_visualize(args, dataset_df)
        cmd_cv(args, dataset_df)
        cmd_stats(args, dataset_df)
        cmd_train_final(args, dataset_df)
        cmd_compare(args, dataset_df)
        cmd_explain(args, dataset_df)


if __name__ == '__main__':
    main()