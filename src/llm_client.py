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

    examples: optional list of (text, intent) pairs shown as earlier turns.
    Zero-shot passes none; Stage 3 passes few-shot examples.
    """
    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT.format(intent_list=intent_list_text(intents))}]
    for ex_text, ex_intent in examples or []:
        messages.append({"role": "user", "content": ex_text})
        messages.append({"role": "assistant",
                         "content": json.dumps({"intent": ex_intent, "confidence": 1.0})})
    messages.append({"role": "user", "content": text})
    return messages


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
