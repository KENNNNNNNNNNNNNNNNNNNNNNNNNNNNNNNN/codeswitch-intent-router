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
