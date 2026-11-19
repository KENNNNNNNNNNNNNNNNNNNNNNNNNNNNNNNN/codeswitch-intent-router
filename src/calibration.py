"""Day 6: does confidence track accuracy, and where should handoff start?

All functions take a DataFrame of predictions with columns
intent, pred, confidence (and density for the bucket table).
A missing prediction counts as wrong. A missing confidence counts as 0,
so that message is always handed off.
"""
import numpy as np
import pandas as pd

from src.evaluate import BUCKETS, bucket

DEFAULT_THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0]
CONF_BINS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]


def _prepare(df):
    out = df.copy()
    out["correct"] = (out["pred"] == out["intent"]).astype(int)
    out["conf"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    return out


def coverage_at(df, threshold):
    """What happens if we answer only when confidence >= threshold."""
    d = _prepare(df)
    answered = d[d["conf"] >= threshold]
    handed = d[d["conf"] < threshold]
    return {
        "threshold": threshold,
        "n": len(d),
        "answered": len(answered),
        "coverage": len(answered) / len(d) if len(d) else np.nan,
        "acc_answered": answered["correct"].mean() if len(answered) else np.nan,
        "handed_off": len(handed),
        "errors_caught": int((handed["correct"] == 0).sum()),
        "errors_missed": int((answered["correct"] == 0).sum()),
    }


def coverage_table(df, thresholds=DEFAULT_THRESHOLDS):
    return pd.DataFrame([coverage_at(df, t) for t in thresholds])


def choose_threshold(df, target=0.95):
    """Lowest threshold whose answered messages are at least `target` correct.

    Candidates are 0 and every confidence value seen, so the choice is exact.
    Returns None if no threshold reaches the target (hand off everything).
    """
    d = _prepare(df)
    for t in sorted({0.0, *d["conf"].unique()}):
        row = coverage_at(df, t)
        if row["answered"] and row["acc_answered"] >= target:
            return t
    return None


def reliability_table(df, bins=CONF_BINS):
    """Accuracy vs mean confidence inside each confidence bin."""
    d = _prepare(df)
    d["bin"] = pd.cut(d["conf"], bins=bins, include_lowest=True)
    table = d.groupby("bin", observed=False).agg(
        n=("correct", "size"), mean_conf=("conf", "mean"), accuracy=("correct", "mean"))
    table["gap"] = table["mean_conf"] - table["accuracy"]
    return table.reset_index()


def expected_calibration_error(df, bins=CONF_BINS):
    """Average |confidence - accuracy| over bins, weighted by bin size."""
    t = reliability_table(df, bins)
    t = t[t["n"] > 0]
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def by_bucket(df):
    """Accuracy and mean confidence per density bucket.

    gap > 0 means the model is more confident than it is accurate.
    """
    d = _prepare(df)
    d["bucket"] = d["density"].apply(bucket)
    rows = []
    for name in BUCKETS:
        part = d[d["bucket"] == name]
        rows.append({"bucket": name, "n": len(part),
                     "accuracy": part["correct"].mean() if len(part) else np.nan,
                     "mean_conf": part["conf"].mean() if len(part) else np.nan,
                     "errors": int((part["correct"] == 0).sum())})
    out = pd.DataFrame(rows)
    out["gap"] = out["mean_conf"] - out["accuracy"]
    return out


def auroc(df):
    """Chance that a random right answer has higher confidence than a random
    wrong one (ties count half). 0.5 = confidence says nothing about
    correctness; 1.0 = it separates them perfectly. None if there are no
    wrong (or no right) answers to compare.
    """
    d = _prepare(df)
    right = d.loc[d["correct"] == 1, "conf"].to_numpy()
    wrong = d.loc[d["correct"] == 0, "conf"].to_numpy()
    if len(right) == 0 or len(wrong) == 0:
        return None
    diff = right[:, None] - wrong[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
