"""Compare tagger v1 and v2 against hand-labelled gold tokens.

    python experiments/check_tagger_v2.py

Gold sets:
  data/gold_tokens.csv          10 registrar sentences (Day 3, v1 tokenisation)
  data/gold_tokens_general.csv  everyday Taglish sentences, v2 tokenisation
                                (contractions split, guideline v0.2)
  data/gold_tokens_hard.csv     verb forms, Taglish verbs and slang
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tagger as v1  # noqa: E402
from src import tagger_v2 as v2  # noqa: E402


def v1_tags(text):
    return [(t, v1.tag_token(t)) for t in v1.tokenize(text)]


def v2_tags(text):
    return [(t.text, t.tag) for t in v2.tag_sentence(text)]


def score(name, texts, gold, tagger):
    right = total = skipped = unk = 0
    errors = []
    for sid, rows in gold.groupby("id"):
        expected = list(rows.sort_values("token_index")[["token", "tag"]].itertuples(index=False))
        got = tagger(texts[sid])
        if [g[0] for g in got] != [e[0] for e in expected]:
            skipped += 1
            errors.append(f"  id {sid}: tokens differ: {[g[0] for g in got]}")
            continue
        for (tok, tag), (_, gold_tag) in zip(got, expected):
            total += 1
            unk += tag == "UNK"
            if tag == gold_tag:
                right += 1
            else:
                errors.append(f"  id {sid}: {tok!r} gold={gold_tag} tagger={tag}")
    acc = right / total if total else float("nan")
    print(f"{name:<6} {right}/{total} tokens correct ({acc:.3f}), {unk} UNK, "
          f"{skipped} sentences skipped (tokenisation differs)")
    for e in errors:
        print(e)


def main():
    data = pd.read_csv("data/taglish_dataset.csv").set_index("id")["text"]
    gold = pd.read_csv("data/gold_tokens.csv")
    print("=== Registrar gold (10 sentences, Day 3) ===")
    score("v1", data, gold, v1_tags)
    score("v2", data, gold, v2_tags)

    general = Path("data/gold_tokens_general.csv")
    if general.exists():
        texts = pd.read_csv("data/gold_sentences_general.csv").set_index("id")["text"]
        g = pd.read_csv(general)
        print(f"\n=== Everyday Taglish gold ({g['id'].nunique()} sentences) ===")
        print("v1 cannot split contractions, so it is only scored on sentences "
              "where its tokens line up.")
        score("v1", texts, g, v1_tags)
        score("v2", texts, g, v2_tags)

    hard = Path("data/gold_tokens_hard.csv")
    if hard.exists():
        texts = pd.read_csv("data/gold_sentences_hard.csv").set_index("id")["text"]
        g = pd.read_csv(hard)
        print(f"\n=== Harder set: verbs, Taglish verbs and slang ({g['id'].nunique()} sentences) ===")
        score("v2", texts, g, v2_tags)


if __name__ == "__main__":
    main()
