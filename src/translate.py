"""Translate English, Tagalog or Taglish text into pure English or pure Tagalog.

One API call per translation. The quality is not measured anywhere in this
project, so the demo labels translations as unverified.
"""
import json
import re

from src.llm_client import DEFAULT_MODEL, call_model

TARGETS = {"english": "English", "tagalog": "Tagalog (Filipino)"}
MAX_CHARS = 2000

SYSTEM_PROMPT = """You translate messages written in English, Tagalog, or a mix \
of both (Taglish) into natural, everyday {language} only.

- Translate every word that is not already in {language}, including slang and
  shortened forms (for example "Ako'y" = "Ako ay", "'cause" = "because").
- Keep names, brands, numbers and the meaning and tone of the original.
- Keep sentence and paragraph breaks.
- Do not add explanations.

Reply with only a JSON object: {{"translation": "<the translated text>"}}"""


def build_messages(text, target):
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}")
    if not text.strip():
        raise ValueError("nothing to translate")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text is longer than {MAX_CHARS} characters")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(language=TARGETS[target])},
            {"role": "user", "content": text}]


def parse_reply(reply):
    """Return (translation, error). Falls back to the raw text if it isn't JSON."""
    if not reply or not reply.strip():
        return None, "empty reply"
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            value = data.get("translation") if isinstance(data, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip(), ""
        except json.JSONDecodeError:
            pass
    return reply.strip(), "reply was not JSON; showing it as is"


def translate(client, text, target, model=DEFAULT_MODEL, **kwargs):
    reply = call_model(client, build_messages(text, target), model=model, **kwargs)
    translation, error = parse_reply(reply)
    return {"translation": translation, "target": target, "error": error, "raw_reply": reply}


BOTH_PROMPT = """You translate messages written in English, Tagalog, or a mix of \
both (Taglish). Give two versions of the message:

- "english": natural, everyday English only.
- "tagalog": natural, everyday Tagalog (Filipino) only, with no English words
  where a common Tagalog word exists.

In both versions translate slang and shortened forms (for example "Ako'y" =
"Ako ay", "'cause" = "because"), keep names, brands and numbers, keep the
meaning and tone, and keep sentence and paragraph breaks. Do not explain.

Reply with only a JSON object: {"english": "<text>", "tagalog": "<text>"}"""


def build_both_messages(text):
    if not text.strip():
        raise ValueError("nothing to translate")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text is longer than {MAX_CHARS} characters")
    return [{"role": "system", "content": BOTH_PROMPT}, {"role": "user", "content": text}]


def parse_both(reply):
    """Return ({"english": str|None, "tagalog": str|None}, error)."""
    out = {"english": None, "tagalog": None}
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return out, "reply was not JSON"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out, "reply was not valid JSON"
    if not isinstance(data, dict):
        return out, "reply was not a JSON object"
    for key in out:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    missing = [k for k, v in out.items() if v is None]
    return out, (f"missing: {', '.join(missing)}" if missing else "")


def translate_both(client, text, model=DEFAULT_MODEL, **kwargs):
    """One API call that returns the message in plain English and plain Tagalog."""
    reply = call_model(client, build_both_messages(text), model=model, **kwargs)
    versions, error = parse_both(reply)
    return {**versions, "error": error, "raw_reply": reply}
