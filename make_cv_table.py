import pandas as pd

CSV_PATH = "cross_validation/cv_mean_std_table.csv"          # change to the full path if needed
OUT_PATH = "test_results/table_cv.tex"

NAMES = {"gcn": "GCN", "gat": "GAT", "gcn_transformer": "GCN-Transformer", "gat_transformer": "GAT-Transformer",
         "baseline_gradient_boosting": "Gradient Boosting", "baseline_random_forest": "Random Forest",
         "baseline_logistic_regression": "Logistic Regression", "baseline_linear_svm": "Linear SVM",
         "baseline_knn": "k-NN"}
ORDER = list(NAMES)
COLS = [("accuracy", "Acc. (\\%)", True), ("balanced_accuracy", "Bal. acc. (\\%)", True),
        ("precision", "Prec. (\\%)", True), ("recall", "Rec. (\\%)", True), ("f1", "F1 (\\%)", True),
        ("auc", "AUROC", False), ("pr_auc", "PR-AUC", False), ("mcc", "MCC", False)]

df = pd.read_csv(CSV_PATH).set_index("model").loc[ORDER]


def split(s):
    m, sd = s.split("±")
    return float(m), float(sd)


best = {c: max(split(df.loc[m, c])[0] for m in ORDER) for c, _, _ in COLS}


def cell(model, col, pct):
    m, sd = split(df.loc[model, col])
    k = 100 if pct else 1
    d = 1 if pct else 3
    s = f"{k * m:.{d}f} $\\pm$ {k * sd:.{d}f}"
    return f"\\textbf{{{s}}}" if round(m, 4) == round(best[col], 4) else s


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Five-fold scaffold cross-validation on the development set (mean $\pm$ standard deviation over folds). "
    r"Models were trained with default hyperparameters on undersampled (1:1) training folds and evaluated on validation "
    r"folds with the natural class ratio, at a cutoff of 0.5. Accuracy, balanced accuracy, precision, recall and F1-score "
    r"are percentages. The best mean in each column is in bold.}",
    r"\label{tab:cv}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}l" + "c" * len(COLS) + r"@{}}",
    r"\toprule",
    "Model & " + " & ".join(h for _, h, _ in COLS) + r" \\",
    r"\midrule",
]
for m in ORDER:
    lines.append(NAMES[m] + " & " + " & ".join(cell(m, c, p) for c, _, p in COLS) + r" \\")
    if m == "gat_transformer":
        lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(OUT_PATH, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
