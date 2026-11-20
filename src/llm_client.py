"""Talk to an LLM (through Groq) to classify a registrar inquiry.

The prompt building and reply parsing are plain functions, so they can be
tested without calling the API. Stage 3 (few-shot) reuses everything here
and only adds examples to the prompt.
"""
import json
import math
import os
import re
import time

DEFAULT_MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "zero-shot-v1"

# Short, neutral definitions of each intent. Written from the intent names
# and the general kinds of request in the dataset, not from test sentences.
INTENTS = {
    "class_schedule": "class schedules, class times, sections and schedule changes",
    "clearance": "clearance status and clearance requirements",
    "document_request": "requesting official documents or certificates from the "
                        "registrar (for example certificate of registration, good "
                        "moral, diploma copy), including processing time and fees",
    "enrollment_status": "whether the student is enrolled or their enrollment is confirmed",
    "student_records": "viewing, getting copies of, or correcting grades, transcript "
                       "of records and other academic or personal records",
    "tuition_payment": "paying tuition, payment methods and payment deadlines",
}

SYSTEM_PROMPT = """You classify messages sent to a university registrar's office \
in the Philippines. Messages may be in English, Tagalog, or a mix of both (Taglish).

Choose exactly one intent:
{intent_list}

Reply with only a JSON object, no other text:
{{"intent": "<one intent name from the list>", "confidence": <number from 0 to 1>}}

"confidence" is your probability that the chosen intent is correct."""


def intent_list_text(intents=INTENTS):
    return "\n".join(f"- {name}: {desc}" for name, desc in intents.items())


def build_messages(text, examples=None, intents=INTENTS):
    """Chat messages for one classification.

    examples: optional list of (text, intent) pairs. Zero-shot passes none,
    and then the prompt is exactly the Stage 2 prompt. Few-shot examples are
    listed in the system prompt with their intent only, no confidence, so
    the examples cannot teach the model what confidence number to give.
    """
    system = SYSTEM_PROMPT.format(intent_list=intent_list_text(intents))
    if examples:
        shown = "\n".join(f"Message: {ex_text}\nIntent: {ex_intent}"
                          for ex_text, ex_intent in examples)
        system += f"\n\nLabelled examples:\n\n{shown}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": text}]


def parse_reply(reply, intents=INTENTS):
    """Turn the model's reply into (intent, confidence, error).

    intent is None if the reply has no valid intent. confidence is None if
    missing or not a number, and clipped to [0, 1] otherwise. error is a
    short reason string, or "" when the reply was fine.
    """
    if not reply or not reply.strip():
        return None, None, "empty reply"
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None, None, "no JSON object"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, "invalid JSON"
    if not isinstance(data, dict):
        return None, None, "JSON is not an object"

    intent = str(data.get("intent", "")).strip().lower().replace(" ", "_")
    error = ""
    if intent not in intents:
        error = f"unknown intent: {data.get('intent')!r}"
        intent = None

    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
        if math.isnan(confidence):
            raise ValueError
        confidence = min(1.0, max(0.0, confidence))
    except (TypeError, ValueError):
        confidence = None
        error = error or "missing or invalid confidence"
    return intent, confidence, error


# Errors worth waiting and retrying for. Matched by class name so this file
# does not need the groq package just to be imported (tests run without it).
RETRYABLE = {"RateLimitError", "APIConnectionError", "APITimeoutError",
             "InternalServerError"}


def make_client():
    """Create a Groq client. The key comes from the GROQ_API_KEY variable."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Store it as a user environment variable "
            "and open a new terminal."
        )
    from groq import Groq
    return Groq()


def call_model(client, messages, model=DEFAULT_MODEL, max_retries=5,
               wait_seconds=5.0, sleep=time.sleep):
    """Send messages and return the reply text. Retries on rate limits."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            if type(exc).__name__ not in RETRYABLE or attempt == max_retries:
                raise
            delay = wait_seconds * (2 ** attempt)
            print(f"  {type(exc).__name__}, waiting {delay:.0f}s before retrying...")
            sleep(delay)
    raise RuntimeError("unreachable")


def classify(client, text, model=DEFAULT_MODEL, examples=None, **kwargs):
    """Classify one message. Returns a dict with pred, confidence, error, raw."""
    reply = call_model(client, build_messages(text, examples), model=model, **kwargs)
    intent, confidence, error = parse_reply(reply)
    return {"pred": intent, "confidence": confidence, "parse_error": error, "raw_reply": reply}


# ---------------------------------------------------------------------------
# General use (after Day 7): any intent list, plus tagging the words the
# offline tagger could not place. The research prompt above is unchanged.
# ---------------------------------------------------------------------------

GENERAL_PROMPT_VERSION = "general-v1"
MAX_INTENTS = 20

GENERAL_SYSTEM_PROMPT = """You analyse messages that may be in English, Tagalog, \
or a mix of both (Taglish), written the way Filipinos text and chat.
{intent_part}{unknown_part}
Reply with only a JSON object, no other text:
{json_shape}"""

INTENT_PART = """
Task 1. Choose exactly one intent for the message:
{intent_list}
"confidence" is your probability (0 to 1) that the chosen intent is correct.
"""

UNKNOWN_PART = """
Task {n}. For each of these words from the message, say which language it is in:
{words}
Use "EN" (English), "TL" (Tagalog or Filipino, including slang such as "jowa"),
or "NE" (a name, brand or other language-independent word).
"""


def normalise_intents(intents):
    """Accept a dict {name: description} or a list of names.

    Names become lowercase_with_underscores. A missing description becomes
    the name itself. Raises ValueError if there are too many or duplicates.
    """
    if isinstance(intents, dict):
        items = list(intents.items())
    else:
        items = [(name, "") for name in intents]
    out = {}
    for name, desc in items:
        key = re.sub(r"\s+", "_", str(name).strip().lower())
        if not key:
            continue
        if key in out:
            raise ValueError(f"duplicate intent: {key}")
        out[key] = (desc or "").strip() or key.replace("_", " ")
    if len(out) > MAX_INTENTS:
        raise ValueError(f"at most {MAX_INTENTS} intents")
    return out


def build_general_messages(text, intents, unknown_words):
    """Prompt for the general analysis. Either part may be empty, not both."""
    intents = normalise_intents(intents) if intents else {}
    unknown_words = list(dict.fromkeys(unknown_words))
    if not intents and not unknown_words:
        raise ValueError("nothing to ask the model")
    shape = {}
    intent_part = unknown_part = ""
    if intents:
        intent_part = INTENT_PART.format(intent_list=intent_list_text(intents))
        shape.update({"intent": "<one intent name>", "confidence": "<number 0-1>"})
    if unknown_words:
        unknown_part = UNKNOWN_PART.format(
            n=2 if intents else 1, words=", ".join(f'"{w}"' for w in unknown_words))
        shape["word_languages"] = {w: "<EN|TL|NE>" for w in unknown_words[:2]}
        if len(unknown_words) > 2:
            shape["word_languages"]["..."] = "..."
    system = GENERAL_SYSTEM_PROMPT.format(
        intent_part=intent_part, unknown_part=unknown_part,
        json_shape=json.dumps(shape))
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def parse_general_reply(reply, intents, unknown_words):
    """Returns dict: intent, confidence, word_languages, error."""
    intents = normalise_intents(intents) if intents else {}
    result = {"intent": None, "confidence": None, "word_languages": {}, "error": ""}
    if intents:
        intent, confidence, error = parse_reply(reply, intents)
        result.update(intent=intent, confidence=confidence, error=error)
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    langs = data.get("word_languages", {}) if isinstance(data, dict) else {}
    if isinstance(langs, dict):
        wanted = {w.lower(): w for w in unknown_words}
        for word, tag in langs.items():
            tag = str(tag).strip().upper()
            if str(word).lower() in wanted and tag in ("EN", "TL", "NE"):
                result["word_languages"][wanted[str(word).lower()]] = tag
    if unknown_words and not result["word_languages"] and not result["error"]:
        result["error"] = "no word languages in reply"
    return result


def analyse_general(client, text, intents, unknown_words, model=DEFAULT_MODEL, **kwargs):
    """One API call: intent + confidence (if intents) and tags for unknown words."""
    reply = call_model(client, build_general_messages(text, intents, unknown_words),
                       model=model, **kwargs)
    return {**parse_general_reply(reply, intents, unknown_words), "raw_reply": reply}
