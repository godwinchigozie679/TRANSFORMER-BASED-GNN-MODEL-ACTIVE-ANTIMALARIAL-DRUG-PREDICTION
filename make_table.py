import pandas as pd

CSV_PATH = "test_results/test_set_metrics_table.csv"
OUT_PATH = "test_results/table_metrics.tex"

ORDER = ["GCN", "GAT", "GCN-Transformer", "GAT-Transformer",
         "Gradient Boosting", "Random Forest", "Logistic Regression", "Linear SVM", "k-NN"]
PCT = ["accuracy", "precision", "recall", "f1"]      # shown as percentages
DEC = ["mcc"]                                        # shown as 0.xxx

df = pd.read_csv(CSV_PATH)
N_TEST, PREV = int(df["n_test"].iloc[0]), 100 * float(df["prevalence_active"].iloc[0])   # taken from the results, not typed
tuned = df["model"].str.contains("val-tuned")
df["name"] = df["model"].str.replace(r" \(val-tuned cutoff\)", "", regex=True)
panels = {"a": df[~tuned].set_index("name"), "b": df[tuned].set_index("name")}


def fmt(v, col):
    return f"{100 * v:.1f}" if col in PCT else f"{v:.3f}"


def cell(panel, model, col, best):
    s = fmt(panel.loc[model, col], col)
    return rf"\textbf{{{s}}}" if model in best[col] else s


def best_models(panel, cols):
    out = {}
    for c in cols:
        top = panel[c].loc[ORDER].round(4 if c in DEC else 3 if c not in PCT else 3).max()
        out[c] = {m for m in ORDER if round(panel.loc[m, c], 3 if c in PCT else 3) == round(top, 3)}
    return out


cols = PCT + DEC
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Classification performance of all models on the scaffold-held-out test set "
    rf"($n={N_TEST:,}$".replace(",", "{,}") + rf"; {PREV:.1f}\% active). Accuracy, precision, recall and F1-score are percentages. "
    r"Panel (a) uses the default cutoff of 0.5; panel (b) uses a cutoff selected on the validation split only. "
    r"The best value in each column and panel is in bold.}",
    r"\label{tab:metrics}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lcccccc@{}}",
    r"\toprule",
]
for key, title, extra in (("a", r"(a) Default cutoff (0.5)", "auc"), ("b", r"(b) Validation-tuned cutoff", "cutoff")):
    p = panels[key]
    best = best_models(p, cols + ([extra] if extra == "auc" else []))
    head = "AUROC" if extra == "auc" else "Cutoff"
    lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}}\\")
    lines.append(rf"Model & {head} & Acc. (\%) & Prec. (\%) & Rec. (\%) & F1 (\%) & MCC \\")
    lines.append(r"\midrule")
    for m in ORDER:
        first = cell(p, m, "auc", best) if extra == "auc" else f"{p.loc[m, 'cutoff']:.2f}"
        rest = " & ".join(cell(p, m, c, best) for c in cols)
        lines.append(rf"{m} & {first} & {rest} \\")
        if m == "GAT-Transformer":
            lines.append(r"\addlinespace[2pt]")
    lines.append(r"\midrule" if key == "a" else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table*}"]
open(OUT_PATH, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
