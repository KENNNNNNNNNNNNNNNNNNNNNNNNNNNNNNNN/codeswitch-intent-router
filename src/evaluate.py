"""Shared evaluation helpers: overall and per-density-bucket scores.

Every stage (baseline, zero-shot, few-shot) uses these, so the numbers
are always computed the same way.
"""
import math

import pandas as pd

BUCKETS = ["1 single-language (0)", "2 low-mixed (0-0.3]", "3 high-mixed (>0.3)"]

# Below this many rows, a score is too noisy to draw conclusions from.
NOISY_N = 10


def bucket(d):
    """Map a density value to one of the three density buckets."""
    if d == 0:
        return BUCKETS[0]
    if d <= 0.3:
        return BUCKETS[1]
    return BUCKETS[2]


def wilson_interval(correct, n, z=1.96):
    """95% Wilson confidence interval for an accuracy of correct/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(true, pred):
    """Accuracy, macro F1 and a 95% interval for accuracy.

    Missing predictions (None/NaN, for example an unparseable LLM reply)
    count as wrong.
    """
    # Imported here so the web app (which only needs bucket()) runs without scikit-learn.
    from sklearn.metrics import accuracy_score, f1_score

    true = pd.Series(list(true), dtype=object).reset_index(drop=True)
    pred = pd.Series(list(pred), dtype=object).reset_index(drop=True).fillna("<none>")
    n = len(true)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "noisy": True}
    correct = int((true == pred).sum())
    low, high = wilson_interval(correct, n)
    return {
        "n": n,
        "accuracy": accuracy_score(true, pred),
        # Average only over the true intents, so a "<none>" prediction lowers
        # recall without adding a fake class to the average.
        "macro_f1": f1_score(true, pred, labels=sorted(true.unique()),
                             average="macro", zero_division=0),
        "ci_low": low,
        "ci_high": high,
        "noisy": n < NOISY_N,
    }


def evaluate(df, true_col="intent", pred_col="pred", density_col="density"):
    """Return a table with one row for 'overall' and one per density bucket."""
    rows = [{"group": "overall", **score(df[true_col], df[pred_col])}]
    buckets = df[density_col].apply(bucket)
    for name in BUCKETS:
        part = df[buckets == name]
        rows.append({"group": name, **score(part[true_col], part[pred_col])})
    return pd.DataFrame(rows)


def format_table(table):
    """Pretty text version of an evaluate() table."""
    lines = [f"{'group':<24}{'n':>4}{'acc':>7}{'macro F1':>10}   95% CI for acc"]
    for _, r in table.iterrows():
        if r["n"] == 0:
            lines.append(f"{r['group']:<24}{0:>4}      -         -")
            continue
        flag = "  (too few to trust)" if r["noisy"] else ""
        lines.append(
            f"{r['group']:<24}{int(r['n']):>4}{r['accuracy']:>7.3f}{r['macro_f1']:>10.3f}"
            f"   [{r['ci_low']:.2f}, {r['ci_high']:.2f}]{flag}"
        )
    return "\n".join(lines)
