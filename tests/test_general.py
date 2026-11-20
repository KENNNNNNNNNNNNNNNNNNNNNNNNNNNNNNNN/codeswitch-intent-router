"""Stage B: general intents, LLM tagging of unknown words, translator. API mocked."""
import json

import pytest

from src.llm_client import (build_general_messages, build_messages,
                            normalise_intents, parse_general_reply)
from src.router import PRESETS, analyze_general, decide
from src.translate import MAX_CHARS
from src.translate import build_messages as translate_messages
from src.translate import parse_reply as parse_translation
from src.translate import translate
from tests.test_llm_client import FakeClient


# ---- intents --------------------------------------------------------------

def test_intent_names_are_normalised_and_get_default_descriptions():
    assert normalise_intents(["Order Status", " refund "]) == {
        "order_status": "order status", "refund": "refund"}


def test_duplicate_or_too_many_intents_are_rejected():
    with pytest.raises(ValueError):
        normalise_intents(["a", "A"])
    with pytest.raises(ValueError):
        normalise_intents([f"i{n}" for n in range(21)])


def test_presets_are_valid_intent_lists():
    for intents in PRESETS.values():
        assert 2 <= len(normalise_intents(intents)) <= 20


def test_research_prompt_is_unchanged():
    # Days 4-6 results depend on this prompt staying the same.
    system = build_messages("x")[0]["content"]
    assert system.startswith("You classify messages sent to a university registrar's office")


# ---- general prompt and parsing ----------------------------------------------

def test_general_prompt_has_both_tasks_when_needed():
    system = build_general_messages("Sana all may jowa", ["love_life", "other"], ["jowa"])[0]["content"]
    assert "love_life" in system and '"jowa"' in system and "Task 2" in system


def test_general_prompt_without_intents_only_asks_for_words():
    system = build_general_messages("jowa ko", None, ["jowa"])[0]["content"]
    assert "Task 1. For each" in system and "intent" not in system.split("Reply")[0].lower()


def test_general_prompt_needs_something_to_ask():
    with pytest.raises(ValueError):
        build_general_messages("hello", None, [])


def test_parse_general_reply_reads_intent_and_word_languages():
    reply = json.dumps({"intent": "Love Life", "confidence": 0.8,
                        "word_languages": {"JOWA": "tl", "Kyla": "NE", "other": "TL"}})
    r = parse_general_reply(reply, ["love_life"], ["jowa", "Kyla"])
    assert r["intent"] == "love_life" and r["confidence"] == 0.8
    assert r["word_languages"] == {"jowa": "TL", "Kyla": "NE"}  # extra words ignored


def test_parse_general_reply_ignores_bad_tags():
    r = parse_general_reply('{"word_languages": {"jowa": "Spanish"}}', None, ["jowa"])
    assert r["word_languages"] == {} and r["error"]


# ---- analyze_general ----------------------------------------------------------

def test_unknown_words_are_filled_in_by_the_llm():
    client = FakeClient([json.dumps({"intent": "love_life", "confidence": 0.9,
                                     "word_languages": {"jowa": "TL"}})])
    r = analyze_general("Sana all may jowa", client, ["love_life", "other"])
    tags = {t["text"]: (t["tag"], t["how"]) for t in r["tokens"]}
    assert tags["jowa"] == ("TL", "decided by the LLM")
    assert r["n_unknown"] == 0 and r["intent"] == "love_life"
    assert len(client.calls) == 1  # one call for both jobs


def test_no_api_call_when_no_intents_and_no_unknown_words():
    client = FakeClient([])
    r = analyze_general("Kumain ka na ba?", client, intents=None)
    assert client.calls == [] and r["intent"] is None and not r["uses_intents"]


def test_preset_routing_and_handoff_decision():
    client = FakeClient(['{"intent": "tuition_payment", "confidence": 0.6}'])
    r = analyze_general("Paano magbayad ng tuition?", client, PRESETS["campus_registrar"])
    assert r["intent"] == "tuition_payment"
    assert decide(r, 0.8)[0] is True and decide(r, 0.5)[0] is False


# ---- translator ----------------------------------------------------------------

def test_translate_prompt_names_the_target_language():
    assert "Tagalog" in translate_messages("I am hungry", "tagalog")[0]["content"]
    assert "English" in translate_messages("Gutom na ako", "english")[0]["content"]


@pytest.mark.parametrize("text, target", [("hi", "spanish"), ("   ", "english"),
                                          ("a" * (MAX_CHARS + 1), "english")])
def test_translate_rejects_bad_input(text, target):
    with pytest.raises(ValueError):
        translate_messages(text, target)


def test_translate_parses_json_and_falls_back_to_raw_text():
    assert parse_translation('{"translation": "I am hungry."}') == ("I am hungry.", "")
    text, error = parse_translation("I am hungry.")
    assert text == "I am hungry." and error


def test_translate_end_to_end_with_fake_client():
    client = FakeClient(['{"translation": "Hindi na ako makapaghintay kumain dahil gutom na ako."}'])
    r = translate(client, "Ako'y hindi na makapag wait kumain 'cause I am gutom na", "tagalog")
    assert r["translation"].startswith("Hindi na ako") and r["error"] == ""
    assert client.calls[0]["temperature"] == 0


def test_parse_both_languages():
    from src.translate import parse_both
    assert parse_both('{"english": "Hi.", "tagalog": "Kumusta."}') == (
        {"english": "Hi.", "tagalog": "Kumusta."}, "")
    versions, error = parse_both('{"english": "Hi."}')
    assert versions["tagalog"] is None and "tagalog" in error
    assert parse_both("no json")[1]
