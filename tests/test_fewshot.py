import pandas as pd
import pytest

from src.fewshot import (STRATEGIES, as_examples, diverse_set, pick_matched,
                         select, train_pool)


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv("data/dataset_with_density.csv").drop(columns=["split"])
    return d.merge(pd.read_csv("data/splits.csv")[["id", "split"]], on="id")


@pytest.fixture(scope="module")
def pool(data):
    return train_pool(data)


def test_pool_contains_only_train(pool):
    assert set(pool["split"]) == {"train"}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_never_uses_validation_or_test_sentences(strategy, data, pool):
    held_out = set(data[data["split"] != "train"]["id"])
    for _, row in data[data["split"] == "test"].iterrows():
        chosen = select(strategy, pool, row["id"], row["density"])
        assert not set(chosen["id"]) & held_out


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_one_example_per_intent(strategy, data, pool):
    chosen = select(strategy, pool, input_id=1, input_density=0.4)
    assert sorted(chosen["intent"]) == sorted(data["intent"].unique())


def test_matched_picks_the_closest_density(pool):
    chosen = pick_matched(pool, 0.45)
    for _, row in chosen.iterrows():
        same_intent = pool[pool["intent"] == row["intent"]]
        assert abs(row["density"] - 0.45) == (same_intent["density"] - 0.45).abs().min()


def test_english_examples_have_no_tagalog_tokens(pool):
    chosen = select("english", pool, input_id=7, input_density=0.3)
    assert (chosen["n_tl"] == 0).all()


def test_random_is_reproducible_but_varies_by_input(pool):
    a = select("random", pool, 3, 0.0)["id"].tolist()
    assert a == select("random", pool, 3, 0.0)["id"].tolist()
    others = {tuple(select("random", pool, i, 0.0)["id"]) for i in range(1, 30)}
    assert len(others) > 1


def test_diverse_spans_pure_tagalog_to_pure_english(pool):
    shares = sorted(diverse_set(pool)["en_share"])
    assert shares[0] == 0.0 and shares[-1] == 1.0


def test_diverse_is_the_same_for_every_input(pool):
    assert (select("diverse", pool, 1, 0.0)["id"].tolist()
            == select("diverse", pool, 99, 0.5)["id"].tolist())


def test_as_examples_gives_text_intent_pairs(pool):
    pairs = as_examples(pick_matched(pool, 0.0))
    assert len(pairs) == 6 and all(isinstance(t, str) for t, _ in pairs)
    assert [i for _, i in pairs] == sorted(i for _, i in pairs)


def test_unknown_strategy_raises(pool):
    with pytest.raises(ValueError):
        select("nearest-neighbour", pool, 1, 0.0)
