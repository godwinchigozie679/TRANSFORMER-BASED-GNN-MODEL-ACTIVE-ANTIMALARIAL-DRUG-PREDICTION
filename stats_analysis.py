import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compare_descriptors_by_activity(df, descriptor_cols, fdr_alpha=0.05):
    """One t-test per descriptor, active vs inactive, then FDR-corrected
    since testing 100 descriptors at p<0.05 uncorrected will turn up
    several false positives just by chance."""
    rows = []
    for col in descriptor_cols:
        active = df.loc[df['Labels'] == 1, col].dropna()
        inactive = df.loc[df['Labels'] == 0, col].dropna()
        if len(active) < 2 or len(inactive) < 2:
            continue

        _, p_value = stats.ttest_ind(active, inactive, equal_var=False)
        rows.append({
            'descriptor': col,
            'active_mean': active.mean(),
            'inactive_mean': inactive.mean(),
            'difference': active.mean() - inactive.mean(),
            'p_raw': p_value,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    rejected, p_corrected, _, _ = multipletests(result_df['p_raw'], alpha=fdr_alpha, method='fdr_bh')
    result_df['p_fdr'] = p_corrected
    result_df['significant_fdr'] = rejected
    result_df = result_df.sort_values('p_fdr').reset_index(drop=True)

    n_sig = result_df['significant_fdr'].sum()
    print(f"{n_sig}/{len(result_df)} descriptors significant after FDR correction (alpha={fdr_alpha})")

    return result_df
