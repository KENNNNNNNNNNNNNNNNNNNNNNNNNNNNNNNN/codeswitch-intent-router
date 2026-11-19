"""Day 5: choose few-shot examples for the prompt.

Every strategy returns exactly one example per intent (6 in total), taken
from the TRAIN split only. The strategies differ only in WHICH sentence
they pick for each intent, so any accuracy difference comes from the
language mix of the examples, not from which intents were shown.

Strategies
  random       a random train sentence per intent (seeded per input)
  english      a random pure-English train sentence per intent (seeded per input)
  matched      the train sentence whose density is closest to the input's
  diverse      one fixed set for every input, spread from pure Tagalog to
               pure English (see diverse_set)
"""
from itertools import permutations

import numpy as np
import pandas as pd

STRATEGIES = ["random", "english", "matched", "diverse"]

# English share targets for the "diverse" set: pure Tagalog ... pure English.
DIVERSE_TARGETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def train_pool(df):
    """Only train rows may ever be used as examples."""
    pool = df[df["split"] == "train"].copy()
    if pool.empty:
        raise ValueError("no train rows to choose examples from")
    pool["en_share"] = pool["n_en"] / (pool["n_en"] + pool["n_tl"]).replace(0, np.nan)
    return pool.sort_values("id")


def _rng(seed, input_id):
    return np.random.default_rng([seed, int(input_id)])


def pick_random(pool, input_id, seed=42, english_only=False):
    rng = _rng(seed, input_id)
    rows = []
    for intent, group in pool.groupby("intent", sort=True):
        if english_only:
            group = group[(group["n_tl"] == 0) & (group["n_en"] > 0)]
            if group.empty:
                raise ValueError(f"no pure-English train example for {intent}")
        rows.append(group.iloc[rng.integers(len(group))])
    return pd.DataFrame(rows)


def pick_matched(pool, input_density):
    """Per intent, the example with density closest to the input's.

    Ties go to the lower id, so the choice is deterministic.
    """
    rows = []
    for _, group in pool.groupby("intent", sort=True):
        gap = (group["density"] - input_density).abs()
        best = group.assign(gap=gap).sort_values(["gap", "id"]).iloc[0]
        rows.append(best.drop("gap"))
    return pd.DataFrame(rows)


def diverse_set(pool, targets=DIVERSE_TARGETS):
    """One fixed set spanning the language mix, one example per intent.

    Each intent is given one target English share (0 = pure Tagalog,
    1 = pure English) and picks its closest example. The target-to-intent
    assignment is the one with the smallest total distance, found by trying
    all 720 assignments.
    """
    intents = sorted(pool["intent"].unique())
    if len(intents) != len(targets):
        raise ValueError("need one target per intent")
    # Closest example (and its distance) for every intent/target pair.
    closest = {}
    for intent in intents:
        group = pool[(pool["intent"] == intent) & pool["en_share"].notna()]
        for target in targets:
            gap = (group["en_share"] - target).abs()
            row = group.assign(gap=gap).sort_values(["gap", "id"]).iloc[0]
            closest[intent, target] = row
    best = min(permutations(targets),
               key=lambda order: sum(closest[i, t]["gap"] for i, t in zip(intents, order)))
    return pd.DataFrame([closest[i, t].drop("gap") for i, t in zip(intents, best)])


_DIVERSE_CACHE = {}


def select(strategy, pool, input_id, input_density, seed=42):
    """Return the 6 example rows (one per intent) for one input sentence."""
    if strategy == "random":
        chosen = pick_random(pool, input_id, seed)
    elif strategy == "english":
        chosen = pick_random(pool, input_id, seed, english_only=True)
    elif strategy == "matched":
        chosen = pick_matched(pool, input_density)
    elif strategy == "diverse":
        key = tuple(pool["id"])
        if key not in _DIVERSE_CACHE:
            _DIVERSE_CACHE[key] = diverse_set(pool)
        chosen = _DIVERSE_CACHE[key]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    assert (chosen["split"] == "train").all()
    assert chosen["intent"].nunique() == len(chosen)
    return chosen


def as_examples(chosen):
    """(text, intent) pairs in a fixed intent order, for build_messages."""
    return list(chosen.sort_values("intent")[["text", "intent"]].itertuples(index=False, name=None))
