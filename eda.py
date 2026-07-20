import os

import matplotlib
# Only force a non-interactive backend when we're not inside a notebook
# kernel. Forcing Agg unconditionally (what this used to do) breaks
# inline plot display in Jupyter/VS Code, since it overrides the
# kernel's own inline backend before any plot gets drawn — the figure
# still saves to disk fine, it just never shows up in the notebook
# output, which is a real bug, not just a warning to ignore.
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
import seaborn as sns

KEY_DESCRIPTORS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount',
]


def dataset_overview(df):
    n_active = (df['Labels'] == 1).sum()
    n_inactive = (df['Labels'] == 0).sum()
    n_descriptors = len([c for c in df.columns if c not in ('Smiles', 'Labels')])

    print(f"total compounds: {len(df)}")
    print(f"descriptors: {n_descriptors}")
    print(f"active: {n_active} ({n_active/len(df)*100:.1f}%)")
    print(f"inactive: {n_inactive} ({n_inactive/len(df)*100:.1f}%)")
    if n_inactive > 0:
        print(f"class balance ratio (active/inactive): {n_active/n_inactive:.3f}")

    return {
        'total': len(df),
        'active': int(n_active),
        'inactive': int(n_inactive),
        'n_descriptors': n_descriptors,
    }


def plot_label_distribution(df, save_path=None):
    counts = df['Labels'].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    plt.title('Active vs inactive')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Inactive (0)', 'Active (1)'], rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def plot_descriptor_distributions(df, descriptors=None, save_path=None):
    """Grid of histograms, split by activity, for a handful of the more
    commonly-discussed descriptors. Skips anything not present in df
    rather than erroring, since which descriptors survived feature
    selection varies run to run."""
    descriptors = [d for d in (descriptors or KEY_DESCRIPTORS) if d in df.columns]
    if not descriptors:
        print("none of the requested descriptors are in this dataframe, nothing to plot")
        return

    n_cols = 4
    n_rows = (len(descriptors) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    active = df[df['Labels'] == 1]
    inactive = df[df['Labels'] == 0]

    for ax, desc in zip(axes, descriptors):
        ax.hist(inactive[desc].dropna(), bins=25, alpha=0.6, label='Inactive', color='red')
        ax.hist(active[desc].dropna(), bins=25, alpha=0.6, label='Active', color='blue')
        ax.set_title(desc)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[len(descriptors):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def lipinski_analysis(df):
    """Rule of Five compliance — allowing one violation, which is the
    usual convention rather than requiring zero."""
    violations = pd.Series(0, index=df.index)
    violations += (df['MolWt'] > 500).astype(int)
    violations += (df['LogP'] > 5).astype(int)
    violations += (df['NumHDonors'] > 5).astype(int)
    violations += (df['NumHAcceptors'] > 10).astype(int)

    compliant = (violations <= 1).sum()
    non_compliant = len(df) - compliant

    print(f"Lipinski compliant (<=1 violation): {compliant} ({compliant/len(df)*100:.1f}%)")
    print(f"non-compliant: {non_compliant} ({non_compliant/len(df)*100:.1f}%)")

    return {'compliant': int(compliant), 'non_compliant': int(non_compliant), 'violations': violations}


def activity_comparison_table(df, descriptors=None):
    """Simple active-vs-inactive mean comparison. This is a quick look,
    not a statistical claim — use stats_analysis.compare_descriptors_by_activity
    if you want the FDR-corrected significance test."""
    descriptors = [d for d in (descriptors or KEY_DESCRIPTORS) if d in df.columns]
    active = df[df['Labels'] == 1]
    inactive = df[df['Labels'] == 0]

    rows = []
    for desc in descriptors:
        a_mean, i_mean = active[desc].mean(), inactive[desc].mean()
        rows.append({'descriptor': desc, 'active_mean': a_mean, 'inactive_mean': i_mean, 'difference': a_mean - i_mean})

    return pd.DataFrame(rows)


def correlation_heatmap(df, descriptor_cols, save_path=None, annotate=False):
    corr = df[descriptor_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=annotate, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Descriptor correlation matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    return corr


def high_correlation_heatmap(df, descriptor_cols, threshold=0.85, save_path=None):
    """Only the pairs above threshold, self-correlations dropped. If
    nothing clears the bar, says so instead of showing an empty plot."""
    corr = df[descriptor_cols].corr()
    strong = corr.where((corr.abs() > threshold) & (corr != 1.0))
    strong = strong.dropna(how='all', axis=0).dropna(how='all', axis=1)

    if strong.empty:
        print(f"no descriptor pairs above |r| > {threshold}")
        return None

    plt.figure(figsize=(min(1.2 * len(strong.columns) + 4, 24), min(1.2 * len(strong.columns) + 4, 24)))
    sns.heatmap(strong, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.7})
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    return strong


def top_correlated_with_activity(df, descriptor_cols, n=10):
    corr = df[descriptor_cols + ['Labels']].corr()['Labels'].abs().sort_values(ascending=False)
    corr = corr.drop('Labels', errors='ignore')
    top = corr.head(n)
    print(f"top {n} descriptors correlated with activity:")
    for name, value in top.items():
        print(f"  {name:<25}: {value:.4f}")
    return top


def run_full_eda(df, descriptor_cols, output_dir='figures'):
    """Runs everything above and saves the plots. Meant for the CLI
    (`run.py eda`) — if you're in a notebook, call the individual
    functions instead so you can see each plot inline as you go."""
    os.makedirs(output_dir, exist_ok=True)

    print("=== dataset overview ===")
    dataset_overview(df)

    print("\n=== label distribution ===")
    plot_label_distribution(df, save_path=f'{output_dir}/label_distribution.png')

    print("\n=== descriptor distributions ===")
    plot_descriptor_distributions(df, save_path=f'{output_dir}/descriptor_distributions.png')

    print("\n=== Lipinski rule of five ===")
    lipinski_analysis(df)

    print("\n=== active vs inactive, key descriptors ===")
    comparison = activity_comparison_table(df)
    print(comparison.to_string(index=False))

    print("\n=== correlation heatmap ===")
    correlation_heatmap(df, descriptor_cols, save_path=f'{output_dir}/correlation_heatmap.png')

    print("\n=== highly correlated descriptor pairs ===")
    high_correlation_heatmap(df, descriptor_cols, save_path=f'{output_dir}/high_correlation_heatmap.png')

    print("\n=== top descriptors correlated with activity ===")
    top_correlated_with_activity(df, descriptor_cols)
