import pandas as pd
import pytest

from src.evaluate import BUCKETS, bucket, evaluate, score, wilson_interval


@pytest.mark.parametrize("d, expected", [(0, BUCKETS[0]), (0.1, BUCKETS[1]),
                                         (0.3, BUCKETS[1]), (0.31, BUCKETS[2]),
                                         (0.5, BUCKETS[2])])
def test_bucket_edges_match_baseline(d, expected):
    assert bucket(d) == expected


def test_perfect_score():
    s = score(["a", "b"], ["a", "b"])
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0 and s["n"] == 2


def test_missing_prediction_counts_as_wrong():
    s = score(["a", "b"], ["a", None])
    assert s["accuracy"] == 0.5


def test_small_groups_are_flagged_noisy():
    assert score(["a"] * 5, ["a"] * 5)["noisy"]
    assert not score(["a"] * 10, ["a"] * 10)["noisy"]


def test_wilson_interval_is_wide_for_small_n():
    low, high = wilson_interval(4, 5)  # 0.80 on 5 sentences
    assert low < 0.4 and high > 0.95


def test_evaluate_has_overall_and_every_bucket():
    df = pd.DataFrame({
        "intent": ["a", "b", "a", "b"],
        "pred":   ["a", "b", "b", "b"],
        "density": [0.0, 0.2, 0.5, 0.5],
    })
    table = evaluate(df).set_index("group")
    assert table.loc["overall", "accuracy"] == 0.75
    assert table.loc[BUCKETS[0], "n"] == 1
    assert table.loc[BUCKETS[2], "accuracy"] == 0.5
