"""Tests for src/llm_client.py. The API is always mocked: no key, no network."""
from types import SimpleNamespace

import pytest

from src.llm_client import (INTENTS, build_messages, call_model, classify,
                            parse_reply)


# ---- a fake Groq client -------------------------------------------------

class FakeClient:
    """Mimics client.chat.completions.create(...) and records each call."""

    def __init__(self, replies):
        self.replies = list(replies)  # strings to return, or exceptions to raise
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RateLimitError(Exception):
    """Same class name as groq.RateLimitError, which is what the retry checks."""


# ---- parse_reply ---------------------------------------------------------

def test_parse_clean_json():
    assert parse_reply('{"intent": "clearance", "confidence": 0.9}') == ("clearance", 0.9, "")


def test_parse_json_inside_extra_text_and_code_fence():
    reply = 'Sure!\n```json\n{"intent": "tuition_payment", "confidence": 0.75}\n```'
    assert parse_reply(reply) == ("tuition_payment", 0.75, "")


def test_parse_normalises_case_and_spaces():
    intent, _, error = parse_reply('{"intent": " Class Schedule ", "confidence": 1}')
    assert intent == "class_schedule" and error == ""


def test_parse_unknown_intent_gives_none():
    intent, confidence, error = parse_reply('{"intent": "parking", "confidence": 0.8}')
    assert intent is None and confidence == 0.8 and "unknown intent" in error


@pytest.mark.parametrize("value, expected", [(1.7, 1.0), (-0.2, 0.0), ("0.6", 0.6)])
def test_parse_confidence_is_clipped_and_converted(value, expected):
    _, confidence, _ = parse_reply(f'{{"intent": "clearance", "confidence": {value!r}}}'
                                   .replace("'", '"'))
    assert confidence == expected


def test_parse_missing_confidence_keeps_intent():
    assert parse_reply('{"intent": "clearance"}') == ("clearance", None,
                                                      "missing or invalid confidence")


@pytest.mark.parametrize("reply", ["", "   ", "I think it is clearance.", "{not json}"])
def test_parse_bad_replies(reply):
    intent, confidence, error = parse_reply(reply)
    assert intent is None and confidence is None and error


# ---- build_messages ------------------------------------------------------

def test_zero_shot_prompt_lists_every_intent_and_ends_with_the_text():
    messages = build_messages("Paano magbayad ng tuition?")
    assert messages[0]["role"] == "system"
    for name in INTENTS:
        assert name in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Paano magbayad ng tuition?"}
    assert len(messages) == 2


def test_examples_become_earlier_turns():
    messages = build_messages("new text", examples=[("ex one", "clearance")])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert '"clearance"' in messages[2]["content"]


# ---- call_model / classify with the fake client --------------------------

def test_classify_returns_parsed_result_and_uses_temperature_zero():
    client = FakeClient(['{"intent": "enrollment_status", "confidence": 0.95}'])
    result = classify(client, "Enrolled na ba ako?", model="some-model")
    assert result["pred"] == "enrollment_status"
    assert result["confidence"] == 0.95
    assert client.calls[0]["model"] == "some-model"
    assert client.calls[0]["temperature"] == 0


def test_rate_limit_is_retried_then_succeeds():
    waits = []
    client = FakeClient([RateLimitError("slow down"), '{"intent": "clearance", "confidence": 1}'])
    reply = call_model(client, [], sleep=waits.append, wait_seconds=1)
    assert "clearance" in reply
    assert waits == [1]


def test_gives_up_after_max_retries():
    client = FakeClient([RateLimitError("x")] * 3)
    with pytest.raises(RateLimitError):
        call_model(client, [], max_retries=2, sleep=lambda s: None)
    assert len(client.calls) == 3


def test_other_errors_are_not_retried():
    client = FakeClient([ValueError("bad request")])
    with pytest.raises(ValueError):
        call_model(client, [], sleep=lambda s: None)
    assert len(client.calls) == 1
