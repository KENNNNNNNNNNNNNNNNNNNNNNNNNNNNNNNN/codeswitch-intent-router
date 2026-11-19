"""Day 5: few-shot strategies compared with zero-shot and the baseline.

Run from the repo root:
    python experiments/llm_few_shot.py                     # all 4 strategies, test split
    python experiments/llm_few_shot.py --strategy matched  # just one
    python experiments/llm_few_shot.py --limit 2           # quick try

Every prompt has 6 examples, one per intent, from the TRAIN split only.
Like the zero-shot script, answers are saved as they arrive and a rerun
skips sentences already done. The final ablation table is also saved to
results/ablation_<split>_<model>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.llm_zero_shot import load_split  # noqa: E402
from experiments.llm_zero_shot import out_path as zero_shot_path  # noqa: E402
from src.evaluate import BUCKETS, evaluate  # noqa: E402
from src.fewshot import STRATEGIES, as_examples, select, train_pool  # noqa: E402
from src.llm_client import DEFAULT_MODEL, classify, make_client  # noqa: E402

PROMPT_VERSION = "few-shot-v1"
COLUMNS = ["id", "text", "intent", "density", "bucket", "pred", "confidence",
           "parse_error", "raw_reply", "example_ids", "strategy", "model",
           "prompt_version"]
TEXT_COLUMNS = {"parse_error": str, "raw_reply": str, "example_ids": str}


def out_path(strategy, split, model):
    return Path(f"results/llm_fewshot_{strategy}_{split}_{model.replace('/', '_')}.csv")


def run_strategy(strategy, part, pool, model, pause, client_box):
    path = out_path(strategy, part["split"].iloc[0], model)
    path.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(path)["id"]) if path.exists() else set()
    todo = part[~part["id"].isin(done)]
    print(f"\n--- {strategy}: {len(part) - len(todo)} already done, {len(todo)} to classify")

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        if client_box[0] is None:
            client_box[0] = make_client()
        chosen = select(strategy, pool, row["id"], row["density"])
        result = classify(client_box[0], row["text"], model=model,
                          examples=as_examples(chosen))
        out = {**row[["id", "text", "intent", "density", "bucket"]].to_dict(), **result,
               "example_ids": " ".join(str(x) for x in sorted(chosen["id"])),
               "strategy": strategy, "model": model, "prompt_version": PROMPT_VERSION}
        pd.DataFrame([out], columns=COLUMNS).to_csv(
            path, mode="a", header=not path.exists(), index=False)
        mark = "ok " if result["pred"] == row["intent"] else "XX "
        print(f"  [{i}/{len(todo)}] {mark}{row['intent']:<18} -> {result['pred']} "
              f"(conf {result['confidence']}) {result['parse_error']}")
        time.sleep(pause)
    return pd.read_csv(path, dtype=TEXT_COLUMNS)


def summary_row(name, results, ids):
    results = results[results["id"].isin(ids)]
    table = evaluate(results).set_index("group")
    row = {"strategy": name,
           "correct": f"{int(round(table.loc['overall', 'accuracy'] * len(results)))}/{len(results)}",
           "accuracy": table.loc["overall", "accuracy"],
           "macro_f1": table.loc["overall", "macro_f1"],
           "ci_95": f"[{table.loc['overall', 'ci_low']:.2f}, {table.loc['overall', 'ci_high']:.2f}]"}
    for b in BUCKETS:
        row[f"acc {b[:1]} (n={int(table.loc[b, 'n'])})"] = table.loc[b, "accuracy"]
    if "confidence" in results:
        row["mean_conf"] = results["confidence"].mean()
        errors = results["parse_error"].fillna("").astype(str)
        row["bad_replies"] = int((errors != "").sum())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="all", choices=["all"] + STRATEGIES)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="only the first N sentences")
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv("data/dataset_with_density.csv").drop(columns=["split"])
    data = data.merge(pd.read_csv("data/splits.csv")[["id", "split"]], on="id")
    pool = train_pool(data)
    part = load_split(args.split)
    if args.limit:
        part = part.head(args.limit)
    ids = set(part["id"])

    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]
    client_box = [None]  # created only if a call is actually needed
    results = {s: run_strategy(s, part, pool, args.model, args.pause, client_box)
               for s in strategies}

    rows = []
    base_path = Path("results/baseline_predictions.csv")
    if base_path.exists():
        base = pd.read_csv(base_path)
        rows.append(summary_row("baseline (sklearn)", base[base.split == args.split], ids))
    zs_path = zero_shot_path(args.split, args.model)
    zero = None
    if zs_path.exists():
        zero = pd.read_csv(zs_path, dtype={"parse_error": str, "raw_reply": str})
        if ids <= set(zero["id"]):
            rows.append(summary_row("zero-shot", zero, ids))
        else:
            zero = None
            print("\n(zero-shot results don't cover these sentences; run llm_zero_shot.py)")
    for s, res in results.items():
        rows.append(summary_row(f"few-shot {s}", res, ids))

    table = pd.DataFrame(rows)
    print(f"\n=== Ablation, {args.split} split, {args.model} ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    save = Path(f"results/ablation_{args.split}_{args.model.replace('/', '_')}.csv")
    if not args.limit and args.strategy == "all":
        table.to_csv(save, index=False)
        print(f"saved to {save}")

    if zero is not None:
        print("\nPaired with zero-shot (same sentences): fixed = zero-shot wrong, few-shot right;"
              " broke = the reverse")
        z = zero.set_index("id")
        for s, res in results.items():
            r = res.set_index("id").loc[sorted(ids)]
            z_ok = z.loc[r.index, "pred"] == r["intent"]
            f_ok = r["pred"] == r["intent"]
            fixed = r[f_ok & ~z_ok]
            broke = r[~f_ok & z_ok]
            print(f"  {s:<8} fixed {len(fixed)}, broke {len(broke)}")
            for _, x in broke.iterrows():
                print(f"           broke: true={x['intent']}, now={x['pred']}: {x['text']}")

    print("\nAll wrong answers by strategy:")
    for s, res in results.items():
        wrong = res[(res["id"].isin(ids)) & (res["pred"] != res["intent"])]
        for _, x in wrong.iterrows():
            print(f"  {s:<8} [{x['bucket'][:1]}] true={x['intent']}, pred={x['pred']} "
                  f"(conf {x['confidence']}): {x['text']}")

    print(f"\nn={len(ids)}. A difference of 1-2 sentences between strategies is noise, not a result.")


if __name__ == "__main__":
    main()
