"""Stage 7 (plots): the figures for the paper, in the original publication style (PDF, double-column width).

figures/ gets: confusion_matrices, roc_curves, pr_curves, training_curves, tuning_results, metrics_comparison,
learning_curves (all PDF, same layout as the original analysis) and, new for the revision, prauc_ci and
delong_heatmap (PDF + PNG). Explainability figures are written by explain.py."""
import math
import os
import pickle
import string
import sys

import matplotlib
import numpy as np
import pandas as pd

if "ipykernel" not in sys.modules and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from config import BASELINES, DISPLAY, KINDS, logged, path, pkl_key

TARGET_WIDTH_SINGLE_COL = 3.3          # adjust to your journal's column width
TARGET_WIDTH_DOUBLE_COL = 7.16
ROMAN_NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
GNN_NAMES = [pkl_key(k) for k in KINDS]                               # gnn_model, gat_model, ...
BASE_NAMES = ["baseline_random_forest", "baseline_logistic_regression", "baseline_linear_svm",
              "baseline_gradient_boosting", "baseline_knn"]
ALL_NAMES = BASE_NAMES + GNN_NAMES
DISPLAY_NAMES = {**{pkl_key(k): DISPLAY[k] for k in KINDS}, **{b: DISPLAY[b] for b in BASELINES}}
BASELINE_COLOUR, GNN_COLOUR = "#1f77b4", "#d62728"
TRAIN_COLOUR, VALIDATION_COLOUR, RANDOM_COLOUR, TUNING_COLOUR = "#1f77b4", "#ff7f0e", "#5f5f5f", "#2ca02c"
AUC_YLIM = (0.70, 0.97)
LC_GNNS = ("gcn", "gat", "gcn_transformer", "gat_transformer")
LC_BASELINES = ("random_forest", "logistic_regression", "linear_svm", "gradient_boosting", "knn")
LC_METRICS = ("auc", "accuracy", "pr_auc", "f1", "mcc")
LC_LABELS = {"auc": "AUROC", "accuracy": "Accuracy", "pr_auc": "AUPRC", "f1": "F1", "mcc": "MCC"}
LC_COLOURS = {"gcn": "#0072B2", "gat": "#009E73", "gcn_transformer": "#E69F00", "gat_transformer": "#D55E00",
              "random_forest": "#7f7f7f", "logistic_regression": "#999999", "linear_svm": "#555555",
              "gradient_boosting": "#8c564b", "knn": "#bcbd22"}
LC_MARKERS = {"gcn": "o", "gat": "s", "gcn_transformer": "^", "gat_transformer": "D", "random_forest": "v",
              "logistic_regression": "P", "linear_svm": "X", "gradient_boosting": "<", "knn": ">"}
PALETTE = {"gcn": "#0072B2", "gat": "#56B4E9", "gcn_transformer": "#009E73", "gat_transformer": "#D55E00",
           "baseline_gradient_boosting": "#666666", "baseline_random_forest": "#999999",
           "baseline_logistic_regression": "#CC79A7", "baseline_linear_svm": "#E69F00", "baseline_knn": "#8C6D31"}


def set_publication_style():
    """Font sizes are the real printed sizes, so use figure widths close to the journal column widths."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "axes.labelweight": "bold", "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "black",
        "lines.linewidth": 1.6, "axes.linewidth": 0.9, "grid.linewidth": 0.5, "grid.alpha": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
        "axes.spines.top": True, "axes.spines.right": True})


def add_panel_border(ax, linewidth=0.9):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="black",
                               linewidth=linewidth, clip_on=False))


def display_name(name):
    return DISPLAY_NAMES.get(name, name)


def _finish(fig, name, png=False):
    fig.savefig(path("figures", f"{name}.pdf"), format="pdf")
    if png:
        fig.savefig(path("figures", f"{name}.png"), dpi=300)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print("  saved figures/" + name + (".pdf (+ .png)" if png else ".pdf"))


def _load_results():
    res = {}
    for name in ALL_NAMES:
        p = path("test_results", f"{name}_results.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res[name] = pickle.load(f)
    return res


# ----------------------------------------------------------------------------- original figures
def plot_all_confusion_matrices(res, ncols=3):
    """Confusion matrices at the 0.5 cutoff (the same cutoff as the test-set table)."""
    names = [n for n in ALL_NAMES if n in res]
    nrows = math.ceil(len(names) / ncols)
    pw = TARGET_WIDTH_DOUBLE_COL / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, pw * 0.95 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, name in enumerate(names):
        r = res[name]
        matrix = confusion_matrix(np.asarray(r["true_labels"]), np.asarray(r["predictions"]))
        ax = axes[i]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Inactive", "Active"],
                    yticklabels=["Inactive", "Active"], linewidths=0.7, linecolor="white", square=True,
                    cbar=False, annot_kws={"fontsize": 9, "fontweight": "bold"}, ax=ax)
        ax.set_title(f"{string.ascii_uppercase[i]}. {display_name(name)}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        add_panel_border(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.subplots_adjust(top=0.94, bottom=0.09, left=0.08, right=0.98, hspace=0.55, wspace=0.35)
    _finish(fig, "confusion_matrices")


def plot_roc_curves(res, ncols=3):
    names = [n for n in ALL_NAMES if n in res]
    nrows = math.ceil(len(names) / ncols)
    pw = TARGET_WIDTH_DOUBLE_COL / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, pw * nrows), squeeze=False)
    axes = axes.flatten()
    for i, name in enumerate(names):
        r = res[name]
        fpr, tpr, _ = roc_curve(r["true_labels"], r["probabilities"])
        ax = axes[i]
        ax.plot(fpr, tpr, color=BASELINE_COLOUR if name.startswith("baseline_") else GNN_COLOUR,
                label=f"AUC = {auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color=RANDOM_COLOUR, linewidth=1.0, label="Random")
        ax.set_title(f"{string.ascii_uppercase[i]}. {display_name(name)}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
        add_panel_border(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.98, hspace=0.55, wspace=0.32)
    _finish(fig, "roc_curves")


def plot_pr_curves(res):
    """Precision-recall curves, one panel. Dotted line = no-skill precision (fraction of actives in the test set)."""
    curves, prevalence = [], None
    for name in ALL_NAMES:
        if name not in res:
            continue
        y, p = np.asarray(res[name]["true_labels"]).astype(int), np.asarray(res[name]["probabilities"])
        precision, recall, _ = precision_recall_curve(y, p)
        curves.append((name, recall, precision, average_precision_score(y, p)))
        prevalence = float(y.mean())
    curves.sort(key=lambda c: -c[3])
    gnn_colours = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    base_colours = ["#1f77b4", "#2ca02c", "#17becf", "#7f7f7f", "#bcbd22"]
    gi = bi = 0
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, recall, precision, ap in curves:
        if name.startswith("baseline_"):
            colour, style = base_colours[bi % len(base_colours)], "--"
            bi += 1
        else:
            colour, style = gnn_colours[gi % len(gnn_colours)], "-"
            gi += 1
        ax.plot(recall, precision, color=colour, linestyle=style, linewidth=1.3,
                label=f"{display_name(name)} (AP = {ap:.3f})")
    ax.axhline(prevalence, color=RANDOM_COLOUR, linestyle=":", linewidth=1.0, label=f"No skill ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    add_panel_border(ax)
    fig.subplots_adjust(top=0.97, bottom=0.14, left=0.10, right=0.68)
    _finish(fig, "pr_curves")


def save_training_curves_pdf():
    """Loss and AUC curves per epoch for each GNN (final training run, first seed); rows = models."""
    names, hist = [], []
    for name in GNN_NAMES:
        p = path("training_history", f"{name}_history.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                hist.append(pickle.load(f))
            names.append(name)
    if not names:
        print("  no training history found - training_curves skipped")
        return
    lo = min(min(min(h["train_auc"]), min(h["val_auc"])) for h in hist)
    ylim = (min(AUC_YLIM[0], np.floor((lo - 0.02) * 20) / 20), 1.0 if lo < AUC_YLIM[0] else AUC_YLIM[1])
    panels = [("Loss", "train_loss", "val_loss"), ("AUC", "train_auc", "val_auc")]
    pw = TARGET_WIDTH_DOUBLE_COL / 2
    fig, axes = plt.subplots(len(names), 2, figsize=(TARGET_WIDTH_DOUBLE_COL, pw * 0.82 * len(names)), squeeze=False)
    for row, (name, h) in enumerate(zip(names, hist)):
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        for col, (label, tk, vk) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(epochs, h[tk], label="Training", color=TRAIN_COLOUR)
            ax.plot(epochs, h[vk], label="Validation", color=VALIDATION_COLOUR)
            ax.set_title(ROMAN_NUMERALS[col])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            if label == "AUC":
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="best")
            add_panel_border(ax)
    fig.subplots_adjust(top=0.95, bottom=0.07, left=0.08, right=0.98, hspace=0.85, wspace=0.30)
    for row, name in enumerate(names):          # row labels are placed after the layout is fixed
        l, r = axes[row, 0].get_position(), axes[row, 1].get_position()
        fig.text((l.x0 + r.x1) / 2, l.y1 + 0.028, f"{string.ascii_uppercase[row]}. {display_name(name)}",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    _finish(fig, "training_curves")


def save_tuning_results_pdf(ncols=2):
    """Mean validation AUC (+/- SD over folds) of the configurations that reached full cross-validation."""
    tuning = {}
    for name in GNN_NAMES:
        p = path(f"{name}_tuning_results.csv")
        if os.path.exists(p):
            tuning[name] = pd.read_csv(p)
    if not tuning:
        print("  no tuning results found - tuning_results skipped")
        return
    nrows = math.ceil(len(tuning) / ncols)
    pw = TARGET_WIDTH_DOUBLE_COL / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, pw * 0.75 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, (name, df) in enumerate(tuning.items()):
        ax = axes[i]
        cols = [c for c in df.columns if c not in ("mean_val_auc", "std_val_auc")]
        labels = ["\n".join(f"{c}={row[c]}" for c in cols) for _, row in df.iterrows()]
        x = np.arange(len(df))
        ax.bar(x, df["mean_val_auc"], yerr=df["std_val_auc"], capsize=3,
               error_kw={"elinewidth": 1.0, "capthick": 1.0}, color=TUNING_COLOUR, alpha=0.9)
        ax.set_title(f"{string.ascii_uppercase[i]}. {display_name(name)}")
        ax.set_ylabel("Mean validation AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        add_panel_border(ax)
    for ax in axes[len(tuning):]:
        ax.axis("off")
    fig.subplots_adjust(top=0.93, bottom=0.20, left=0.08, right=0.98, hspace=0.75, wspace=0.30)
    _finish(fig, "tuning_results")


def plot_all_metrics_comparison(metric_list=("accuracy", "auc", "precision", "recall", "f1"), ncols=2):
    """Cross-validation bars (mean +/- SD over folds) for each metric, GNNs and baselines."""
    p = path("cv_folds.csv")
    if not os.path.exists(p):
        print("  cv_folds.csv not found - metrics_comparison skipped")
        return
    cv = pd.read_csv(p)
    g = cv.groupby("model")
    table = pd.DataFrame({"internal": list(g.groups)})
    table["model"] = table["internal"].map(DISPLAY)
    table["type"] = np.where(table["internal"].str.startswith("baseline_"), "baseline", "GNN")
    for m in metric_list:
        table[f"{m}_mean"] = table["internal"].map(g[m].mean())
        table[f"{m}_std"] = table["internal"].map(g[m].std(ddof=0))
    table = table.sort_values("auc_mean", ascending=False).reset_index(drop=True)
    nrows = math.ceil(len(metric_list) / ncols)
    pw = TARGET_WIDTH_DOUBLE_COL / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, pw * 0.72 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, m in enumerate(metric_list):
        ax = axes[i]
        colours = [BASELINE_COLOUR if t == "baseline" else GNN_COLOUR for t in table["type"]]
        x = np.arange(len(table))
        ax.bar(x, table[f"{m}_mean"], yerr=table[f"{m}_std"], color=colours, capsize=3,
               error_kw={"elinewidth": 1.0, "capthick": 1.0}, alpha=0.9)
        label = m.upper() if m.lower() == "auc" else m.capitalize()
        ax.set_title(f"{string.ascii_uppercase[i]}. {label}")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(table["model"], rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        add_panel_border(ax)
    for ax in axes[len(metric_list):]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=BASELINE_COLOUR, label="Classical baseline"),
                        Patch(color=GNN_COLOUR, label="Graph neural network")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.08, right=0.98, hspace=0.65, wspace=0.30)
    _finish(fig, "metrics_comparison")


def save_learning_curves_pdf():
    """One panel per metric: mean +/- SD over scaffold CV folds against training-set size."""
    p = path("learning_curve_summary.csv")
    if not os.path.exists(p):
        print("  learning_curve_summary.csv not found (run the curves stage) - learning_curves skipped")
        return
    s = pd.read_csv(p)
    ncols = 3
    nrows = math.ceil((len(LC_METRICS) + 1) / ncols)            # one spare cell holds the legend
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(TARGET_WIDTH_DOUBLE_COL, TARGET_WIDTH_DOUBLE_COL * 0.36 * nrows))
    flat = axes.flatten()
    for i, metric in enumerate(LC_METRICS):
        ax = flat[i]
        order = [m for m in LC_GNNS + LC_BASELINES if m in set(s["model"])]
        for name in order:
            d = s[s["model"] == name].sort_values("n_train")
            is_gnn = name in LC_GNNS
            x, y, sd = d["n_train"].values, d[f"{metric}_mean"].values, d[f"{metric}_sd"].values
            ax.plot(x, y, color=LC_COLOURS[name], marker=LC_MARKERS[name], markersize=5 if is_gnn else 4,
                    markeredgecolor="white", markeredgewidth=0.6, linewidth=1.9 if is_gnn else 1.1,
                    linestyle="-" if is_gnn else "--", zorder=3 if is_gnn else 2,
                    label=display_name(pkl_key(name) if name in LC_GNNS else f"baseline_{name}"))
            if is_gnn:
                ax.fill_between(x, y - sd, y + sd, color=LC_COLOURS[name], alpha=0.14, linewidth=0, zorder=1)
        ax.set_xlabel("Training compounds")
        ax.set_ylabel(LC_LABELS[metric])
        ax.grid(True, linestyle="--", alpha=0.3)
        add_panel_border(ax)
        ax.set_title(f"{string.ascii_uppercase[i]}. {LC_LABELS[metric]}")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[len(LC_METRICS)].axis("off")
    flat[len(LC_METRICS)].legend(handles, labels, loc="center", ncol=1, fontsize=7.5)
    for ax in flat[len(LC_METRICS) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    _finish(fig, "learning_curves")


# ----------------------------------------------------------------------------- figures added for the revision
def plot_prauc_ci_and_delong(res):
    from matplotlib.ticker import MultipleLocator
    yt = res[GNN_NAMES[0]]["true_labels"]
    keys = KINDS + BASELINES
    # PR-AUC with bootstrap confidence intervals
    ci = pd.read_csv(path("bootstrap_ci_table.csv"))
    parse = lambda s: [float(v) for v in s.strip("()").split(",")]
    ci[["lo", "hi"]] = [parse(s) for s in ci["PR-AUC 95% CI"]]
    ci = ci.sort_values("PR-AUC")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    inv = {v: k for k, v in DISPLAY.items()}
    for y_pos, (_, r) in enumerate(ci.iterrows()):
        ax.errorbar(r["PR-AUC"], y_pos, xerr=[[r["PR-AUC"] - r["lo"]], [r["hi"] - r["PR-AUC"]]], fmt="o",
                    color=PALETTE[inv[r["Model"]]], ms=4, capsize=2, lw=1)
    ax.set_yticks(range(len(ci)), ci["Model"])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("PR-AUC (95% bootstrap CI)")
    _finish(fig, "prauc_ci", png=True)

    # DeLong heatmap (upper triangle): signed AUROC difference (row - column).
    # Coloured cell = significant after Holm correction (p < 0.05); grey "n.s." cell = not significant.
    dl = pd.read_csv(path("delong_pairwise.csv"))
    order = [DISPLAY[k] for k in sorted(keys, key=lambda k: -roc_auc_score(yt, res[pkl_key(k)]["probabilities"]))]
    n = len(order)
    auc_of = {**dict(zip(dl["Model A"], dl["AUC A"])), **dict(zip(dl["Model B"], dl["AUC B"]))}
    delta, sig = np.full((n, n), np.nan), np.zeros((n, n), bool)
    for _, r in dl.iterrows():
        i, j = sorted((order.index(r["Model A"]), order.index(r["Model B"])))      # i = better-ranked model (row)
        delta[i, j] = auc_of[order[i]] - auc_of[order[j]]
        sig[i, j] = r["p_holm"] < 0.05
    rows, cols = order[:-1], order[1:]                      # drop the empty last row and first column
    delta, sig = delta[:-1, 1:], sig[:-1, 1:]
    nr, nc = delta.shape
    vmax = 0.03
    cmap = plt.get_cmap("Blues")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i in range(nr):
        for j in range(nc):
            if np.isnan(delta[i, j]):
                continue
            d = delta[i, j]
            face = cmap(norm(max(d, 0))) if sig[i, j] else "#d9d9d9"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
            txt = "0.000" if abs(d) < 0.0005 else f"{d:+.3f}"
            dark = sig[i, j] and norm(max(d, 0)) > 0.55
            ax.text(j, i, txt if sig[i, j] else txt + "\nn.s.", ha="center", va="center", fontsize=7,
                    color="white" if dark else "black", fontweight="bold" if sig[i, j] else "normal")
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(nc), cols, rotation=30, ha="right")
    ax.set_yticks(range(nr), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03, extend="max")
    cb.set_label("\u0394AUROC (row \u2212 column), significant pairs", fontweight="bold")
    _finish(fig, "delong_heatmap", png=True)


IMPORTANCE_COLOURS = ["#4B83B7", "#C94A45", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]   # A blue, B red


def save_feature_importance_pdf(importance_dfs, output_path=None, ncols=2, top_n=10, feature_col="feature"):
    """Horizontal feature-importance bar charts (top_n features), double-column width, no panel border:
    only the left and bottom axes are drawn, bold titles ("A. Atomic", "B. Molecular")."""
    output_path = output_path or path("figures", "feature_importance.pdf")
    n = len(importance_dfs)
    nrows = math.ceil(n / ncols)
    pw = TARGET_WIDTH_DOUBLE_COL / ncols
    fig, axes = plt.subplots(nrows, min(n, ncols), figsize=(pw * min(n, ncols), pw * 0.9 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, (method, df) in enumerate(importance_dfs.items()):
        ax = axes[i]
        top = df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
        ax.barh(top[feature_col], top["importance"], color=IMPORTANCE_COLOURS[i % len(IMPORTANCE_COLOURS)],
                edgecolor="white", linewidth=0.6)
        ax.set_title(f"{string.ascii_uppercase[i]}. {method}", fontweight="bold")
        ax.set_xlabel("Feature importance", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both", width=1.2)
        ax.grid(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(w_pad=3.0, h_pad=2.0)                      # adapts to long descriptor names, no overlap
    fig.savefig(output_path, format="pdf")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print(f"Saved feature-importance figure to {output_path}")


@logged
def plots():
    set_publication_style()
    res = _load_results()
    plot_all_confusion_matrices(res)
    plot_roc_curves(res)
    plot_pr_curves(res)
    save_training_curves_pdf()
    save_tuning_results_pdf()
    plot_all_metrics_comparison()
    save_learning_curves_pdf()
    plot_prauc_ci_and_delong(res)