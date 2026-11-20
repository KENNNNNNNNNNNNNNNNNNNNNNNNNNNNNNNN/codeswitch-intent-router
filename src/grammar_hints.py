"""Formal-writing suggestions for raw/daw and rin/din.

KWF rule: after a word ending in a vowel or in w/y, write "raw"/"rin"; after any
other consonant, write "daw"/"din". In casual writing both forms are used
freely, so these are suggestions, never errors. See
docs/tagalog_morphology_notes.md, section 5.
"""
import re

PAIRS = {"daw": "raw", "raw": "daw", "din": "rin", "rin": "din"}
R_AFTER = set("aeiouwy")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ'’-]+")


def expected_form(word, previous):
    """The formal form of raw/daw/rin/din after `previous`, or None if unsure."""
    prev = previous.lower().replace("’", "'").rstrip("'-")
    if not prev or not prev[-1].isalpha():
        return None
    if prev[-1] == "d":
        return None  # sources differ after a final d; make no suggestion
    r_form = prev[-1] in R_AFTER
    base = word.lower()
    if base in ("daw", "raw"):
        return "raw" if r_form else "daw"
    if base in ("din", "rin"):
        return "rin" if r_form else "din"
    return None


def hints(text):
    """List of suggestions: {word, previous, suggestion, reason, position}."""
    out = []
    matches = list(WORD_RE.finditer(text))
    for i, m in enumerate(matches):
        word = m.group(0)
        if word.lower() not in PAIRS or i == 0:
            continue
        prev = matches[i - 1]
        between = text[prev.end():m.start()]
        if any(ch in between for ch in ".!?;:"):
            continue  # a new sentence or clause: no previous word to follow
        want = expected_form(word, prev.group(0))
        if want and want != word.lower():
            ending = "a vowel" if prev.group(0)[-1].lower() in "aeiou" else (
                f"'{prev.group(0)[-1].lower()}'" if prev.group(0)[-1].lower() in "wy" else "a consonant")
            suggestion = want.capitalize() if word[0].isupper() else want
            out.append({
                "word": word, "previous": prev.group(0), "suggestion": suggestion,
                "position": m.start(),
                "reason": f"“{prev.group(0)}” ends in {ending}, so formal Filipino uses “{want}”.",
            })
    return out
