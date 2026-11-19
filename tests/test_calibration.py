import pandas as pd
import pytest

from src.calibration import (auroc, by_bucket, choose_threshold, coverage_at,
                             expected_calibration_error, reliability_table)


def make(rows):
    """rows: (correct?, confidence, density)"""
    return pd.DataFrame([{"intent": "a", "pred": "a" if ok else "b",
                          "confidence": c, "density": d} for ok, c, d in rows])


def test_coverage_counts_handoffs_and_caught_errors():
    df = make([(True, 0.99, 0), (False, 0.6, 0), (True, 0.9, 0), (False, 0.95, 0)])
    row = coverage_at(df, 0.9)
    assert row["answered"] == 3 and row["handed_off"] == 1
    assert row["errors_caught"] == 1 and row["errors_missed"] == 1
    assert row["acc_answered"] == pytest.approx(2 / 3)


def test_threshold_zero_answers_everything():
    df = make([(True, 0.2, 0), (False, 0.1, 0)])
    assert coverage_at(df, 0.0)["coverage"] == 1.0


def test_missing_confidence_is_always_handed_off():
    df = make([(True, None, 0), (True, 0.9, 0)])
    assert coverage_at(df, 0.01)["handed_off"] == 1


def test_choose_lowest_threshold_meeting_target():
    df = make([(True, 0.99, 0)] * 9 + [(False, 0.7, 0), (True, 0.6, 0)])
    # Answered accuracy: at 0 or 0.6 -> 10/11 (0.909), at 0.7 -> 9/10, at 0.99 -> 9/9.
    assert choose_threshold(df, 0.95) == 0.99
    assert choose_threshold(df, 0.90) == 0.0


def test_choose_returns_zero_when_already_accurate():
    assert choose_threshold(make([(True, 0.5, 0)] * 20), 0.95) == 0.0


def test_choose_returns_none_when_target_unreachable():
    assert choose_threshold(make([(False, 0.9, 0), (False, 0.8, 0)]), 0.95) is None


def test_auroc_perfect_and_useless():
    perfect = make([(True, 0.9, 0), (True, 0.8, 0), (False, 0.3, 0)])
    useless = make([(True, 0.99, 0), (False, 0.99, 0)])
    assert auroc(perfect) == 1.0
    assert auroc(useless) == 0.5
    assert auroc(make([(True, 0.9, 0)])) is None


def test_overconfident_model_has_positive_gap_and_calibration_error():
    df = make([(True, 0.99, 0.5), (False, 0.99, 0.5)])
    b = by_bucket(df).set_index("bucket")
    assert b.loc["3 high-mixed (>0.3)", "gap"] == pytest.approx(0.49)
    assert expected_calibration_error(df) == pytest.approx(0.49)


def test_reliability_bins_cover_all_rows():
    df = make([(True, 0.1, 0), (True, 0.85, 0), (False, 0.99, 0), (True, 1.0, 0)])
    assert reliability_table(df)["n"].sum() == 4
