"""Day 4: zero-shot LLM intent classification, compared with the baseline.

Run from the repo root:
    python experiments/llm_zero_shot.py                 # test split
    python experiments/llm_zero_shot.py --split validation
    python experiments/llm_zero_shot.py --limit 2       # quick try, 2 sentences

Each answer is saved to the results CSV as soon as it arrives. Running the
script again skips sentences that are already in the file, so an
interrupted run (or a rate limit) never repeats paid/limited calls.
Delete the CSV to start over.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import bucket, evaluate, format_table  # noqa: E402
from src.llm_client import (DEFAULT_MODEL, PROMPT_VERSION, classify,  # noqa: E402
                            make_client)

COLUMNS = ["id", "text", "intent", "density", "bucket", "pred", "confidence",
           "parse_error", "raw_reply", "model", "prompt_version"]


def load_split(split):
    data = pd.read_csv("data/dataset_with_density.csv")
    splits = pd.read_csv("data/splits.csv")[["id", "split"]]
    df = data.drop(columns=["split"]).merge(splits, on="id")
    part = df[df.split == split].copy()
    part["bucket"] = part["density"].apply(bucket)
    return part


def out_path(split, model):
    slug = model.replace("/", "_")
    return Path(f"results/llm_zero_shot_{split}_{slug}.csv")


def run(split, model, limit, pause):
    part = load_split(split)
    if limit:
        part = part.head(limit)
    path = out_path(split, model)
    path.parent.mkdir(exist_ok=True)

    done = set()
    if path.exists():
        done = set(pd.read_csv(path)["id"])
    todo = part[~part["id"].isin(done)]
    print(f"{split}: {len(part)} sentences, {len(part) - len(todo)} already done, "
          f"{len(todo)} to classify with {model}")

    client = make_client() if len(todo) else None
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        result = classify(client, row["text"], model=model)
        out = {**row[["id", "text", "intent", "density", "bucket"]].to_dict(), **result,
               "model": model, "prompt_version": PROMPT_VERSION}
        pd.DataFrame([out], columns=COLUMNS).to_csv(
            path, mode="a", header=not path.exists(), index=False)
        mark = "ok " if result["pred"] == row["intent"] else "XX "
        print(f"  [{i}/{len(todo)}] {mark}{row['intent']:<18} -> {result['pred']} "
              f"(conf {result['confidence']}) {result['parse_error']}")
        if i < len(todo):
            time.sleep(pause)
    # Read text columns as text: an all-empty column would otherwise load as numbers.
    return pd.read_csv(path, dtype={"parse_error": str, "raw_reply": str}), path


def report(results, split, path):
    print(f"\nPredictions: {path}")
    errors = results[results["parse_error"].fillna("") != ""]
    if len(errors):
        print(f"{len(errors)} replies could not be fully parsed (counted as wrong if "
              f"no intent): {sorted(errors['parse_error'].unique())}")

    print(f"\n=== LLM zero-shot, {split} ===")
    print(format_table(evaluate(results)))

    base_path = Path("results/baseline_predictions.csv")
    if base_path.exists():
        base = pd.read_csv(base_path)
        base = base[(base.split == split) & base["id"].isin(results["id"])]
        print(f"\n=== scikit-learn baseline, same {len(base)} sentences ===")
        print(format_table(evaluate(base)))

        both = results.merge(base[["id", "pred"]], on="id", suffixes=("", "_base"))
        diff = both[both["pred"] != both["pred_base"]]
        print(f"\nSentences where the two models disagree: {len(diff)}")
        for _, r in diff.iterrows():
            print(f"  [{r['bucket'][:1]}] true={r['intent']}, llm={r['pred']}, "
                  f"baseline={r['pred_base']}: {r['text']}")
    else:
        print("\n(Run experiments/baseline_sklearn.py first to see the baseline comparison.)")

    print("\nMean confidence by bucket (looked at properly in Stage 4):")
    print(results.groupby("bucket")["confidence"].agg(["count", "mean"]).round(3).to_string())
    print("\nWith n=16 overall and 3-8 per bucket, one sentence moves a bucket's "
          "accuracy by 12-33 points. Treat bucket differences as anecdotes, not findings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="only the first N sentences")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="seconds to wait between calls (free-tier rate limits)")
    args = parser.parse_args()
    results, path = run(args.split, args.model, args.limit, args.pause)
    report(results, args.split, path)


if __name__ == "__main__":
    main()
