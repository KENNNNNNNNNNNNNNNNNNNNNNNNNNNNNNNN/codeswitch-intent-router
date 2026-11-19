import pandas as pd

from src.router import decide, preregistered_threshold, route
from tests.test_llm_client import FakeClient


def test_route_combines_tagging_density_and_llm():
    client = FakeClient(['{"intent": "tuition_payment", "confidence": 0.9}'])
    r = route("Pwede ba magbayad ng tuition using GCash?", client, threshold=0.5)
    assert r["intent"] == "tuition_payment" and r["confidence"] == 0.9
    assert 0 < r["density"] <= 0.5
    assert any(tag == "TL" for _, tag in r["tokens"])
    assert r["handoff"] is False


def test_low_confidence_is_handed_off():
    client = FakeClient(['{"intent": "clearance", "confidence": 0.6}'])
    r = route("clearance?", client, threshold=0.8)
    assert r["handoff"] is True and "below" in r["reason"]


def test_unusable_reply_is_always_handed_off_even_at_zero():
    client = FakeClient(["sorry, I cannot help"])
    r = route("hello", client, threshold=0.0)
    assert r["intent"] is None and r["handoff"] is True


def test_threshold_is_inclusive():
    analysis = {"intent": "clearance", "confidence": 0.97, "parse_error": ""}
    assert decide(analysis, 0.97)[0] is False
    assert decide(analysis, 0.98)[0] is True


def test_preregistered_threshold_reads_dev_row(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"data": ["dev (chosen here)", "test (held out)"],
                  "threshold": [0.97, 0.97]}).to_csv(path, index=False)
    assert preregistered_threshold(path) == 0.97


def test_preregistered_threshold_defaults_to_zero(tmp_path):
    assert preregistered_threshold(tmp_path / "missing.csv") == 0.0
