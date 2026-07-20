import os

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from descriptors import compute_descriptors

# common drug-likeness descriptors -- preferred when choosing between
# two correlated features
PRIORITY_DESCRIPTORS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds',
    'NumAromaticRings', 'HeavyAtomCount', 'LogS', 'FractionCsp3', 'NumHBA', 'NumHBD',
    'RingCount', 'NumHeteroatoms', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi1', 'LabuteASA',
]


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Labels']).copy()
    df['Labels'] = df['Labels'].astype(int)

    def parses(smiles):
        return isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None

    df = df[df['Smiles'].apply(parses)]
    df = df.drop_duplicates().reset_index(drop=True)

    n_active = (df['Labels'] == 1).sum()
    print(f"{len(df)} compounds after cleaning ({n_active} active, {len(df) - n_active} inactive)")
    return df


def select_non_redundant_features(train_df, descriptor_names, threshold=0.85):
    # operates on train_df only -- passing the full dataset here
    # reintroduces the leak this was built to avoid
    desc = train_df[descriptor_names].copy()
    desc = desc.replace([np.inf, -np.inf], np.nan)  # inf inflates var(), masking a broken descriptor as informative
    desc = desc[desc.columns[desc.isnull().mean() <= 0.5]]
    desc = desc[desc.columns[desc.var() > 1e-10]]
    desc = desc.fillna(desc.median())

    corr = desc.corr().fillna(0)
    selected = []

    def is_redundant(feature):
        for kept in selected:
            if abs(corr.loc[feature, kept]) >= threshold:
                return True
        return False

    for feat in PRIORITY_DESCRIPTORS:
        if feat in desc.columns and not is_redundant(feat):
            selected.append(feat)

    for feat in desc.columns:
        if feat not in selected and not is_redundant(feat):
            selected.append(feat)

    print(f"kept {len(selected)}/{len(descriptor_names)} descriptors (train split only, r >= {threshold} dropped)")
    return selected


def _sanitize(df, cols):
    """Replaces inf/-inf with NaN, then fills missing values with 0.
    StandardScaler cannot handle infinities, which some RDKit
    descriptors (notably Ipc) produce on large or complex molecules."""
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def balance_classes(df, seed):
    X, y = df.drop(columns=['Labels']), df['Labels']
    X_res, y_res = RandomUnderSampler(random_state=seed).fit_resample(X, y)
    return pd.concat([X_res, y_res], axis=1)


def prepare_split(train_df, val_df, test_df, seed, correlation_threshold=0.85):
    all_cols = [c for c in train_df.columns if c not in ('Smiles', 'Labels')]
    descriptor_cols = select_non_redundant_features(train_df, all_cols, correlation_threshold)

    cols = ['Smiles', 'Labels'] + descriptor_cols
    train_df, val_df, test_df = train_df[cols].copy(), val_df[cols].copy(), test_df[cols].copy()
    train_df = balance_classes(train_df, seed)

    scaler = StandardScaler()
    train_df[descriptor_cols] = scaler.fit_transform(_sanitize(train_df, descriptor_cols))
    val_df[descriptor_cols] = scaler.transform(_sanitize(val_df, descriptor_cols))
    test_df[descriptor_cols] = scaler.transform(_sanitize(test_df, descriptor_cols))

    return train_df, val_df, test_df, descriptor_cols, scaler


def prepare_two_way_split(train_df, test_df, seed, correlation_threshold=0.85):
    """Same as prepare_split without a validation set, since the
    classical baselines fit in one call and need no early-stopping
    split."""
    all_cols = [c for c in train_df.columns if c not in ('Smiles', 'Labels')]
    descriptor_cols = select_non_redundant_features(train_df, all_cols, correlation_threshold)

    cols = ['Smiles', 'Labels'] + descriptor_cols
    train_df, test_df = train_df[cols].copy(), test_df[cols].copy()
    train_df = balance_classes(train_df, seed)

    scaler = StandardScaler()
    train_df[descriptor_cols] = scaler.fit_transform(_sanitize(train_df, descriptor_cols))
    test_df[descriptor_cols] = scaler.transform(_sanitize(test_df, descriptor_cols))

    return train_df, test_df, descriptor_cols, scaler


def save_balanced_dataset(input_path, output_path, seed=42):
    """Balances the entire dataset once, by undersampling the majority
    class to match the minority count, and saves the result as a new
    Smiles/Labels CSV at output_path.

    Point config.toml's [data] path at output_path (with new cache
    filenames, since the existing caches are keyed to the original
    data) to have every downstream split -- cv, train-final,
    baselines -- drawn from this balanced source. Test-set metrics
    then describe performance under an artificial 50/50 class balance
    rather than the true distribution; this should be stated
    explicitly if reported."""
    df = load_and_clean(input_path)

    n_active = (df['Labels'] == 1).sum()
    n_inactive = (df['Labels'] == 0).sum()
    n = min(n_active, n_inactive)

    if n == 0:
        raise ValueError("one class has zero compounds after cleaning, can't balance")

    active_df = df[df['Labels'] == 1].sample(n=n, random_state=seed)
    inactive_df = df[df['Labels'] == 0].sample(n=n, random_state=seed)
    balanced_df = pd.concat([active_df, inactive_df]).sample(frac=1, random_state=seed).reset_index(drop=True)

    balanced_df[['Smiles', 'Labels']].to_csv(output_path, index=False)
    print(f"balanced dataset: {len(balanced_df)} compounds ({n} active, {n} inactive) "
          f"— dropped {len(df) - len(balanced_df)} excess majority-class compounds")
    print(f"saved to {output_path}")
    return balanced_df


def outer_split(df, test_size=0.2, seed=42):
    """80/20 outer split. outer_test_df is set aside here and used only
    for the final evaluation, never during tuning or training."""
    outer_train_df, outer_test_df = train_test_split(
        df, test_size=test_size, stratify=df['Labels'], random_state=seed
    )
    return outer_train_df.reset_index(drop=True), outer_test_df.reset_index(drop=True)


def make_holdout_split(df, seed, test_size=0.15, val_size=0.15):
    """Single quick split for development use. Not intended as a
    reported result -- see cross_validation_splits for that."""
    rest_size = test_size + val_size
    train_df, rest_df = train_test_split(df, test_size=rest_size, stratify=df['Labels'], random_state=seed)
    val_df, test_df = train_test_split(
        rest_df, test_size=test_size / rest_size, stratify=rest_df['Labels'], random_state=seed
    )
    return prepare_split(train_df, val_df, test_df, seed)


def cross_validation_splits(df, n_folds=5, seed=42):
    """Stratified k-fold; each fold performs its own feature selection,
    scaling, and undersampling on its training rows only. The
    validation fold serves as the test set for that fold -- report
    mean +/- std across folds, not a single fold's value."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['Labels'])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        all_cols = [c for c in df.columns if c not in ('Smiles', 'Labels')]
        descriptor_cols = select_non_redundant_features(train_df, all_cols)

        cols = ['Smiles', 'Labels'] + descriptor_cols
        train_df, val_df = train_df[cols].copy(), val_df[cols].copy()
        train_df = balance_classes(train_df, seed)

        scaler = StandardScaler()
        train_df[descriptor_cols] = scaler.fit_transform(_sanitize(train_df, descriptor_cols))
        val_df[descriptor_cols] = scaler.transform(_sanitize(val_df, descriptor_cols))

        yield fold_idx, train_df, val_df, descriptor_cols, scaler


def build_full_dataset(raw_path, cache_path='enhanced_drug_data_with_descriptors.csv',
                        use_cache=True, force_recompute=False):
    """Descriptor calculation is the slow step for a large dataset
    (100k+ compounds x 104 descriptors), so results are cached to
    cache_path and reloaded on later runs.

    Set force_recompute=True, or delete cache_path, if the source data
    has changed -- the cache does not detect this automatically."""
    if use_cache and not force_recompute and os.path.exists(cache_path):
        print(f"loading cached descriptors from {cache_path} (pass force_recompute=True, "
              f"or delete this file, to rebuild from {raw_path})")
        df = pd.read_csv(cache_path)
        n_active = (df['Labels'] == 1).sum()
        print(f"{len(df)} compounds loaded ({n_active} active, {len(df) - n_active} inactive)")
        return df

    df = load_and_clean(raw_path)
    df = compute_descriptors(df)

    if use_cache:
        df.to_csv(cache_path, index=False)
        print(f"cached computed descriptors to {cache_path} for next time")

    return df