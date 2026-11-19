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
