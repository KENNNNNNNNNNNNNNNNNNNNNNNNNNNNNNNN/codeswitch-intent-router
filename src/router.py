"""The router: one message in, a routing decision out.

analyze() does the expensive part (one LLM call) and decide() applies the
handoff threshold, so the demo can move its threshold slider without
calling the API again.
"""
from pathlib import Path

import pandas as pd

from src.density import density
from src.evaluate import bucket
from src.llm_client import DEFAULT_MODEL, classify
from src.tagger import switch_rate, tag_sentence

THRESHOLD_FILE = Path("results/handoff_threshold.csv")


def preregistered_threshold(path=THRESHOLD_FILE):
    """The threshold chosen on dev by the Day 6 rule (0.0 if not found)."""
    try:
        table = pd.read_csv(path)
        return float(table.loc[table["data"].str.startswith("dev"), "threshold"].iloc[0])
    except (FileNotFoundError, KeyError, IndexError, pd.errors.EmptyDataError):
        return 0.0


def analyze(text, client, model=DEFAULT_MODEL):
    """Tag the tokens, measure density, and ask the LLM for intent + confidence."""
    tokens = tag_sentence(text)
    tags = [tag for _, tag in tokens]
    d = density(tags)
    result = classify(client, text, model=model)
    return {
        "text": text,
        "tokens": tokens,
        "density": d,
        "bucket": bucket(d),
        "switch_rate": switch_rate(tags),
        "intent": result["pred"],
        "confidence": result["confidence"],
        "parse_error": result["parse_error"],
        "raw_reply": result["raw_reply"],
        "model": model,
    }


def decide(analysis, threshold):
    """Answer automatically, or hand off to staff? Returns (handoff, reason)."""
    if analysis["intent"] is None:
        return True, f"the model's reply could not be used ({analysis['parse_error']})"
    if analysis["confidence"] is None:
        return True, "the model gave no confidence"
    if analysis["confidence"] < threshold:
        return True, (f"confidence {analysis['confidence']:.2f} is below "
                      f"the threshold {threshold:.2f}")
    return False, (f"confidence {analysis['confidence']:.2f} is at or above "
                   f"the threshold {threshold:.2f}")


def route(text, client, threshold=None, model=DEFAULT_MODEL):
    """analyze() + decide() in one call. threshold=None uses the Day 6 rule."""
    if threshold is None:
        threshold = preregistered_threshold()
    analysis = analyze(text, client, model)
    handoff, reason = decide(analysis, threshold)
    return {**analysis, "threshold": threshold, "handoff": handoff, "reason": reason}


# ---------------------------------------------------------------------------
# General use (after Day 7): tagger v2, any intent list, unknown words
# tagged by the LLM in the same call. analyze() above stays as it was for
# the Day 7 Streamlit demo.
# ---------------------------------------------------------------------------
from src import tagger_v2  # noqa: E402
from src.llm_client import INTENTS as REGISTRAR_INTENTS  # noqa: E402
from src.llm_client import analyse_general  # noqa: E402

# Ready-made intent lists. Only "campus_registrar" was evaluated (Days 4-6);
# the others are examples, and the handoff threshold was not calibrated on them.
PRESETS = {
    "campus_registrar": REGISTRAR_INTENTS,
    "customer_support": {
        "order_status": "where an order or delivery is",
        "refund_or_return": "returning an item or getting money back",
        "payment_problem": "failed, double or missing payments",
        "account_access": "login, password or account problems",
        "product_question": "questions about a product or service",
        "complaint": "unhappy with the service, wants to complain",
    },
    "barangay_health": {
        "fever_or_illness": "fever, cough, flu or other sickness symptoms",
        "injury": "wounds, falls, bites or accidents",
        "medicine_question": "which medicine to take, dosage, availability",
        "vaccination": "vaccines and immunisation schedules",
        "appointment": "health centre schedule, check-ups, consultations",
        "emergency": "urgent danger to life, needs help now",
    },
}


def analyze_general(text, client, intents=None, model=DEFAULT_MODEL):
    """Tag with tagger v2, then one LLM call for intent and UNK words.

    intents: dict or list of names, or None/empty to skip intent routing.
    If there are no intents and no unknown words, no API call is made.
    """
    tagged = tagger_v2.analyse(text)
    tokens = tagged["tokens"]
    unknown = [t.text for t in tokens if t.tag == "UNK"]
    llm = {"intent": None, "confidence": None, "word_languages": {}, "error": "",
           "raw_reply": ""}
    if intents or unknown:
        llm = analyse_general(client, text, intents, unknown, model=model)

    filled = []
    for t in tokens:
        if t.tag == "UNK" and t.text in llm["word_languages"]:
            filled.append(t._replace(tag=llm["word_languages"][t.text], how="decided by the LLM"))
        else:
            filled.append(t)
    tags = [t.tag for t in filled]
    d = density(tags)
    return {
        "text": text,
        "tokens": [t._asdict() for t in filled],
        "density": d,
        "bucket": bucket(d),
        "switch_rate": switch_rate(tags),
        "n_unknown": tags.count("UNK"),
        "intent": llm["intent"],
        "confidence": llm["confidence"],
        "uses_intents": bool(intents),
        "parse_error": llm["error"],
        "raw_reply": llm["raw_reply"],
        "model": model,
    }
