"""Day 6: confidence vs accuracy, and the human-handoff threshold.

Run from the repo root, after running llm_zero_shot.py on all three splits:
    python experiments/llm_zero_shot.py --split train
    python experiments/llm_zero_shot.py --split validation
    python experiments/confidence_handoff.py

Pre-registered rule (decided before looking at any of these numbers):
choose the LOWEST confidence threshold at which the messages the system
answers itself are at least 95% correct on train+validation ("dev").
Everything below the threshold is handed off to staff. The threshold is
then applied unchanged to the test split, which played no part in choosing it.

Zero-shot predictions are used because zero-shot never sees the train
split, so train sentences are fair inputs. (Few-shot draws its examples
from train, so its train predictions would not be.)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.llm_zero_shot import out_path  # noqa: E402
from src.calibration import (auroc, by_bucket, choose_threshold,  # noqa: E402
                             coverage_at, coverage_table,
                             expected_calibration_error, reliability_table)
from src.llm_client import DEFAULT_MODEL  # noqa: E402

TARGET = 0.95


def load(split, model):
    path = out_path(split, model)
    if not path.exists():
        sys.exit(f"Missing {path}. Run: python experiments/llm_zero_shot.py --split {split}")
    df = pd.read_csv(path, dtype={"parse_error": str, "raw_reply": str})
    return df.assign(split=split)


def show(title, table):
    print(f"\n=== {title} ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def plot(all_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#2b2b2a", "#8a8980"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    rel = reliability_table(all_rows)
    rel = rel[rel["n"] > 0]
    lo = min(0.5, float(rel["mean_conf"].min()) - 0.05, float(rel["accuracy"].min()) - 0.05)
    ax1.plot([lo, 1], [lo, 1], color=muted, lw=1, ls="--")
    ax1.text(lo + 0.02, lo + 0.045, "perfectly calibrated", color=muted, fontsize=8, rotation=33)
    ax1.plot(rel["mean_conf"], rel["accuracy"], color=blue, lw=2, marker="o", ms=8)
    # Bins crowd together near 1.0, so alternate the n labels above and below.
    for k, (_, r) in enumerate(rel.iterrows()):
        ax1.annotate(f"n={r['n']}", (r["mean_conf"], r["accuracy"]), ha="center",
                     textcoords="offset points", xytext=(0, 10 if k % 2 else -16),
                     fontsize=8, color=ink)
    ax1.set(xlim=(lo, 1.02), ylim=(lo, 1.06), xlabel="stated confidence (bin mean)",
            ylabel="actual accuracy", title="Reliability: all 120 sentences")

    b = by_bucket(all_rows)
    x = range(len(b))
    ax2.plot(x, b["mean_conf"], color=orange, lw=2, marker="o", ms=8, label="mean confidence")
    ax2.plot(x, b["accuracy"], color=blue, lw=2, marker="o", ms=8, label="accuracy")
    ax2.set_xticks(list(x), [f"{n[2:]}\n(n={k})" for n, k in zip(b["bucket"], b["n"])])
    ax2.set(ylim=(0.5, 1.02), ylabel="rate", title="By code-switch density: all 120 sentences")
    ax2.legend(frameon=False, loc="lower left")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    dev = pd.concat([load("train", args.model), load("validation", args.model)])
    test = load("test", args.model)
    every = pd.concat([dev, test])
    print(f"model: {args.model}")
    print(f"dev (train+validation): n={len(dev)}, test: n={len(test)}")

    # 1. Does confidence track accuracy?
    for name, part in [("dev", dev), ("test", test), ("all 120", every)]:
        show(f"Accuracy vs confidence by density bucket, {name}", by_bucket(part))
    show("Reliability by confidence bin, all 120", reliability_table(every))
    for name, part in [("dev", dev), ("test", test), ("all 120", every)]:
        a = auroc(part)
        a_txt = "n/a (no wrong answers)" if a is None else f"{a:.3f}"
        print(f"{name:>8}: calibration error {expected_calibration_error(part):.3f}, "
              f"AUROC {a_txt}")
    print("  (AUROC 0.5 = confidence says nothing about right vs wrong; 1.0 = perfect)")

    # 2. Coverage vs accuracy, and the pre-registered threshold.
    show("Coverage vs accuracy at fixed thresholds, dev", coverage_table(dev))
    show("Coverage vs accuracy at fixed thresholds, test", coverage_table(test))
    t = choose_threshold(dev, TARGET)
    print(f"\nRule: lowest threshold with >= {TARGET:.0%} accuracy on answered dev messages.")
    if t is None:
        print("No threshold reaches the target on dev: the rule says hand off everything.")
        summary = []
    else:
        d_row, t_row = coverage_at(dev, t), coverage_at(test, t)
        print(f"Chosen threshold: {t}")
        summary = [{"data": "dev (chosen here)", **d_row}, {"data": "test (held out)", **t_row}]
        show("Chosen threshold applied", pd.DataFrame(summary))

    # 3. Alternative from the PDF: hand off by density instead of confidence.
    alt = []
    for name, part in [("dev", dev), ("test", test)]:
        high = part["density"] > 0.3
        kept = part[~high]
        alt.append({"data": name, "rule": "hand off all high-mixed (density > 0.3)",
                    "handed_off": int(high.sum()), "coverage": float((~high).mean()),
                    "acc_answered": float((kept["pred"] == kept["intent"]).mean()),
                    "errors_caught": int((part[high]["pred"] != part[high]["intent"]).sum())})
    show("For comparison: density-based handoff", pd.DataFrame(alt))

    # 4. The confidence gap: every wrong answer, most confident first.
    wrong = every[every["pred"] != every["intent"]].sort_values("confidence", ascending=False)
    print(f"\n=== Every wrong answer ({len(wrong)} of {len(every)}), most confident first ===")
    for _, r in wrong.iterrows():
        print(f"  conf {r['confidence']}  [{r['split']}, density {r['density']:.2f}] "
              f"true={r['intent']}, pred={r['pred']}: {r['text']}")

    # Save tables and plot.
    out = Path("results")
    by_bucket(every).to_csv(out / "confidence_by_bucket.csv", index=False)
    coverage_table(dev).assign(data="dev").pipe(
        lambda x: pd.concat([x, coverage_table(test).assign(data="test")])
    ).to_csv(out / "handoff_coverage.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "handoff_threshold.csv", index=False)
    plot(every, out / "calibration_plot.png")
    print("\nsaved: results/confidence_by_bucket.csv, results/handoff_coverage.csv, "
          "results/handoff_threshold.csv, results/calibration_plot.png")

    n_err = len(wrong)
    print(f"\n{n_err} wrong answers in {len(every)} sentences; test alone has n={len(test)}. "
          "With this few errors, every number above is indicative, not proven.")


if __name__ == "__main__":
    main()
