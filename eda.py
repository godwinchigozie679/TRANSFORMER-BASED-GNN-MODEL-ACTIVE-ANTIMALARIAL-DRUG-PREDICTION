"""Stage 1b (eda): look at the data BEFORE any training, as in the original analysis. Run after prepare().
Each function is one notebook cell and writes into the eda/ folder (PNG at 150 dpi, the split figure as PDF):

    class_balance, descriptor_distributions, correlation_heatmap, sample_molecules, extreme_molecules,
    descriptor_statistics (Mann-Whitney U, FDR-corrected), split_overview (protocol, partition sizes, folds, scaffold overlap)
"""
import string

import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw

from config import CFG, logged, path
from data_prep import load_prepared

INACTIVE, ACTIVE = "#4C72B0", "#DD8452"


def _show():
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def _frame(data):
    """Descriptor table with SMILES and label, from the prepared data."""
    df = pd.DataFrame(data["desc"], columns=list(data["desc_names"]))
    df.insert(0, "Labels", data["y"])
    df.insert(0, "Smiles", data["smiles"])
    return df


def load_eda_data():
    """Loads the prepared data (run prepare first). Returns (data dict, dataframe, descriptor column names)."""
    data = load_prepared()
    df = _frame(data)
    return data, df, list(data["desc_names"])


@logged
def class_balance(df):
    counts = df["Labels"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["Inactive", "Active"], counts.values, color=[INACTIVE, ACTIVE])
    plt.ylabel("count")
    plt.title(f"Class balance (n={len(df)}, {counts.get(1, 0) / len(df) * 100:.1f}% active)")
    plt.tight_layout()
    plt.savefig(path("eda", "class_balance.png"), dpi=150)
    _show()
    plt.close()


@logged
def descriptor_distributions(df, descriptor_cols, n=6):
    cols = descriptor_cols[:n]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, col in zip(np.atleast_1d(axes), cols):
        for label, color in ((0, INACTIVE), (1, ACTIVE)):
            vals = df[df["Labels"] == label][col].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=color, label="Active" if label else "Inactive")
        ax.set_title(col)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path("eda", "descriptor_distributions.png"), dpi=150)
    _show()
    plt.close()


@logged
def correlation_heatmap(df, descriptor_cols, n=30):
    cols = descriptor_cols[:n]
    corr = df[cols].replace([np.inf, -np.inf], np.nan).corr()
    plt.figure(figsize=(min(20, len(cols) * 0.4), min(20, len(cols) * 0.4)))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, xticklabels=True, yticklabels=True)
    plt.title("Descriptor correlation heatmap")
    plt.tight_layout()
    plt.savefig(path("eda", "correlation_heatmap.png"), dpi=150)
    _show()
    plt.close()


def structural_label(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid_structure"
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    nC, nN, nO = sym.count("C"), sym.count("N"), sym.count("O")
    ri = mol.GetRingInfo()
    n_arom = sum(1 for r in ri.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    if n_arom >= 2 and nN >= 2:
        base = "quinazoline-like"
    elif n_arom >= 2 and nN == 1:
        base = "quinoline-like"
    elif n_arom == 1 and nN >= 1:
        base = "pyridine-like"
    elif n_arom >= 1:
        base = "aromatic"
    else:
        base = "aliphatic"
    het = ([f"N{nN}" if nN > 1 else "N"] if nN else []) + ([f"O{nO}" if nO > 1 else "O"] if nO else [])
    hal = [f"{s}{sym.count(s)}" if sym.count(s) > 1 else s for s in ("F", "Cl", "Br", "I") if sym.count(s)]
    formula = f"C{nC}" + ("_" + "_".join(het) if het else "") + ("_" + "_".join(hal) if hal else "")
    rings = ri.NumRings()
    return f"{base}_{formula}_{f'{rings}ring' + ('s' if rings != 1 else '') if rings else 'acyclic'}"


def _save_grid(img, file):
    if hasattr(img, "save"):
        img.save(file)
    elif hasattr(img, "data"):
        with open(file, "wb") as f:
            f.write(img.data)


def _draw(rows, legends, file):
    mols = [Chem.MolFromSmiles(s) for s in rows]
    keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not keep:
        print("no valid molecules to draw")
        return None
    img = Draw.MolsToGridImage([m for m, _ in keep], molsPerRow=len(keep), subImgSize=(400, 400),
                               legends=[l for _, l in keep])
    _save_grid(img, file)
    try:
        from IPython.display import display
        display(img)
    except Exception:
        pass
    print("saved", file)
    return img


def _col(df, *names):
    return next((n for n in names if n in df.columns), None)


@logged
def sample_molecules(df, n=3, seed=None):
    """A few random actives and inactives (half and half)."""
    rng = np.random.default_rng(CFG["seed"] if seed is None else seed)
    act, ina = df[df["Labels"] == 1], df[df["Labels"] == 0]
    n_act = min(n // 2 + n % 2, len(act))
    pick = pd.concat([act.iloc[rng.choice(len(act), n_act, replace=False)],
                      ina.iloc[rng.choice(len(ina), min(n - n_act, len(ina)), replace=False)]]).head(n)
    mw, lp = _col(df, "MolWt"), _col(df, "LogP", "MolLogP")
    legends = [f"{structural_label(r.Smiles)}\n{'Active' if r.Labels == 1 else 'Inactive'}"
               + (f"\nMW {r[mw]:.1f}  LogP {r[lp]:.2f}" if mw and lp else "") for _, r in pick.iterrows()]
    return _draw(pick["Smiles"].tolist(), legends, path("eda", "sample_molecules.png"))


@logged
def extreme_molecules(df):
    """Highest and lowest molecular weight, highest LogP."""
    mw, lp = _col(df, "MolWt"), _col(df, "LogP", "MolLogP")
    if not (mw and lp):
        print("MolWt / LogP not among the descriptors, cannot pick extremes")
        return None
    cases = pd.concat([df.nlargest(1, mw).assign(case="highest MW"), df.nsmallest(1, mw).assign(case="lowest MW"),
                       df.nlargest(1, lp).assign(case="highest LogP")]).drop_duplicates(subset="Smiles")
    legends = [f"{structural_label(r.Smiles)}\n{r.case}\n{'Active' if r.Labels == 1 else 'Inactive'}\n"
               f"MW {r[mw]:.1f}  LogP {r[lp]:.2f}" for _, r in cases.iterrows()]
    return _draw(cases["Smiles"].tolist(), legends, path("eda", "extreme_molecules.png"))


def _fdr_bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(len(p))
    out[order] = adj
    return out


@logged
def descriptor_statistics(df, descriptor_cols, alpha=0.05):
    """Mann-Whitney U test of every descriptor between actives and inactives, Benjamini-Hochberg FDR corrected."""
    from scipy import stats
    act, ina = df[df["Labels"] == 1], df[df["Labels"] == 0]
    rows = []
    for col in descriptor_cols:
        a = act[col].replace([np.inf, -np.inf], np.nan).dropna()
        i = ina[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 2 or len(i) < 2:
            continue
        rows.append(dict(descriptor=col, active_mean=a.mean(), active_std=a.std(), inactive_mean=i.mean(),
                         inactive_std=i.std(), p_value=stats.mannwhitneyu(a, i, alternative="two-sided")[1]))
    out = pd.DataFrame(rows)
    out["p_value_fdr"] = _fdr_bh(out["p_value"].values)
    out["significant"] = out["p_value_fdr"] < alpha
    out = out.sort_values("p_value_fdr")
    out.to_csv(path("descriptor_statistics.csv"), index=False)
    print(f"{int(out['significant'].sum())} of {len(out)} descriptors differ between actives and inactives (FDR < {alpha})")
    return out.head(15)


def _box(ax, x, y, w, h, text, color="#E8EEF7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=color,
                                edgecolor="black", linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))


@logged
def split_overview(data):
    """How the data are split: protocol, partition sizes and class mix, cross-validation folds, scaffold overlap.
    Drawn from the splits the pipeline actually uses."""
    y, sc = np.asarray(data["y"]), np.asarray(data["scaffolds"], dtype=object)
    dev, test, cv = data["dev_idx"], data["test_idx"], data["cv"]
    n, nf = len(y), len(cv)
    counts = lambda lab: (int((np.asarray(lab) == 0).sum()), int((np.asarray(lab) == 1).sum()))
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.16, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32, left=0.07, right=0.98, top=0.93, bottom=0.08)
    L = string.ascii_uppercase

    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.05, 0.84, 0.9, 0.13, f"Full dataset (n = {n:,})\nBemis-Murcko scaffold split\n(whole scaffold groups)")
    _box(ax, 0.02, 0.56, 0.58, 0.16, f"Development set {100 * len(dev) / n:.0f}%\n(n = {len(dev):,})", "#DCEBDD")
    _box(ax, 0.66, 0.56, 0.32, 0.16, f"External test {100 * len(test) / n:.0f}%\n(n = {len(test):,})\nnatural ratio", "#F6E3D4")
    _box(ax, 0.02, 0.28, 0.58, 0.18, f"{nf}-fold scaffold CV\ntrain on {nf - 1} folds,\nvalidate on 1 fold", "#DCEBDD")
    _box(ax, 0.02, 0.04, 0.58, 0.16, "Undersample inactives in the\ntraining folds ONLY", "#FFF2C7")
    _box(ax, 0.66, 0.28, 0.32, 0.18, f"Retrain best model\non all {100 * len(dev) / n:.0f}%", "#DCEBDD")
    _box(ax, 0.66, 0.04, 0.32, 0.16, "Evaluate ONCE\non external test", "#F6E3D4")
    for a in [(0.30, 0.84, 0.30, 0.72), (0.82, 0.84, 0.82, 0.72), (0.30, 0.56, 0.30, 0.46), (0.30, 0.28, 0.30, 0.20),
              (0.60, 0.37, 0.66, 0.37), (0.82, 0.28, 0.82, 0.20)]:
        _arrow(ax, *a)
    ax.set_title(f"{L[0]}. Protocol", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    parts = [("Full", counts(y)), (f"Development\n({100 * len(dev) / n:.0f}%)", counts(y[dev])),
             (f"External test\n({100 * len(test) / n:.0f}%)", counts(y[test]))]
    for i, (name, (neg, pos)) in enumerate(parts):
        ax.bar(i, neg, color=INACTIVE, label="Inactive" if i == 0 else None)
        ax.bar(i, pos, bottom=neg, color=ACTIVE, label="Active" if i == 0 else None)
        ax.text(i, neg + pos, f"{pos / (neg + pos):.1%} active", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([p[0] for p in parts])
    ax.set_ylabel("Compounds"); ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(sum(p[1]) for p in parts) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{L[1]}. Partition size and class mix", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    w = 0.27
    for f, (tr, va) in enumerate(cv):
        npos, nneg = int(y[tr].sum()), int((y[tr] == 0).sum())
        ax.bar(f - w, len(tr), w, color="#9AA5B1", label="Training (before undersampling)" if f == 0 else None)
        ax.bar(f, 2 * min(npos, nneg), w, color="#55A868", label="Training (after undersampling)" if f == 0 else None)
        ax.bar(f + w, len(va), w, color="#C44E52", label="Validation (natural ratio)" if f == 0 else None)
        ax.text(f + w, len(va), f"{y[va].mean():.0%}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(nf)); ax.set_xticklabels([f"Fold {i + 1}" for i in range(nf)])
    ax.set_ylabel("Compounds"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_ylim(0, len(dev) * 1.32)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{L[2]}. Cross-validation folds (% = active in validation)", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    s_dev, s_test = set(sc[dev]), set(sc[test])
    fold_overlap = [len(set(sc[tr]) & set(sc[va])) for tr, va in cv]
    vals = [len(s_dev), len(s_test), len(s_dev & s_test), max(fold_overlap)]
    ax.bar(range(4), vals, color=["#9AA5B1", "#9AA5B1", "#C44E52", "#C44E52"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Scaffolds\nin development", "Scaffolds\nin test", "Shared\ndev / test",
                        "Shared train/val\n(max over folds)"], fontsize=6)
    ax.set_ylabel("Unique scaffolds"); ax.set_ylim(0, max(vals) * 1.15 if max(vals) else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{L[3]}. Scaffold overlap (0 = fully disjoint)", loc="left")
    fig.savefig(path("eda", "data_split_overview.pdf"), dpi=300)
    _show()
    plt.close(fig)
    print("saved", path("eda", "data_split_overview.pdf"))
    return {"n_dev": len(dev), "n_test": len(test), "shared_scaffolds_dev_test": len(s_dev & s_test)}


@logged
def run_all():
    """Command-line version: every data-visualisation step."""
    data, df, cols = load_eda_data()
    class_balance(df)
    descriptor_distributions(df, cols)
    correlation_heatmap(df, cols)
    sample_molecules(df)
    extreme_molecules(df)
    descriptor_statistics(df, cols)
    split_overview(data)
