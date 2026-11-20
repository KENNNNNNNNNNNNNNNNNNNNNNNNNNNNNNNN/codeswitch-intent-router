"""Version 2 of the token language tagger, for general (non-registrar) text.

What changed from src/tagger.py (v1, kept unchanged so the Day 1-7 results
stay reproducible):

* Contractions are split into their parts (guideline Rule 7):
  "Ako'y" -> "Ako" + "'y" (= ay), "sa'yo" -> "sa" + "'yo" (= iyo),
  "'to" = ito, "'di" = hindi, "'cause" = because.
* English apostrophe forms stay one English token: "don't", "I'm", "registrar's".
* Word-frequency lists (Tagalog and English subtitles, see data/lexicon/SOURCE.md)
  cover everyday words such as "kumain", "gutom", "pagod".
* Tagalog affixes and infixes are stripped to find the root, and the ROOT
  decides the tag (guideline Rule 1): "nagutom" -> TL, "nag-wait" -> EN.
  A Tagalog prefix written on its own ("makapag wait") is TL.
* Words that are common in both languages ("may", "at") are decided by
  their neighbours.
* Words it still cannot place are tagged UNK instead of being guessed as
  English. UNK, like NE, is left out of density. Stage B lets the LLM fill
  them in.
* A reviewed slang list (data/lexicon/slang.csv) is checked first, and an
  optional Tagalog dictionary (data/lexicon/tl_dict.txt, built from Wiktionary)
  fills gaps: it is only used for words the frequency lists don't know, so it
  can never turn a common English word into Tagalog.
* Taglish verbs with a repeated syllable are understood: "nagsho-shopping",
  "magre-review", "magpapa-check" (see docs/tagalog_morphology_notes.md).
"""
import csv
import math
import re
import unicodedata
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

from src.density import density
from src.tagger import NE_WORDS, TL_WORDS as V1_TL_WORDS

Token = namedtuple("Token", "text norm tag how")

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"

# A word in both lists is Tagalog only if it is at least this many times more
# frequent (relative to list size) in Tagalog subtitles than in English ones;
# otherwise it is English. Taglish speech puts many English words ("class",
# "meeting") into Tagalog subtitles, but they stay far more common in English,
# while Tagalog words that leak into English subtitles ("pera", "tao") are
# hundreds of times more common in Tagalog. Words listed in AMBIGUOUS are
# decided by their neighbours instead.
STRONG_RATIO = 5.0
# The Tagalog list is small (about 89,000 words of subtitles), so a word seen
# only a few times there is weak evidence. A word that is also in the English
# list needs at least this many Tagalog occurrences to be called Tagalog.
MIN_TL_COUNT = 10

# Extra Tagalog words not reliably in the frequency list (or too short to trust).
EXTRA_TL = {
    "ako", "ikaw", "ka", "siya", "sya", "kami", "tayo", "kayo", "sila",
    "akin", "iyo", "ito", "iyan", "iyon", "yan", "yun", "yon", "dito", "diyan",
    "doon", "dun", "ng", "nang", "ang", "mga", "si", "sina", "ni", "nina",
    "kay", "kina", "po", "opo", "oo", "hindi", "di", "huwag", "wag", "ba",
    "naman", "lang", "lamang", "din", "rin", "daw", "raw", "nga", "kasi",
    "talaga", "sana", "pala", "yata", "muna", "pa", "na", "ay", "kung",
    "kapag", "pag", "pero", "ngunit", "dahil", "kaya", "para", "parang",
    "mas", "pinaka", "ano", "sino", "saan", "kailan", "bakit", "paano",
    "ilan", "magkano", "alin", "yung", "iyong", "nasa", "wala", "meron",
    "mayroon", "kain", "inom", "tulog", "gutom", "pagod", "busog", "uwi",
    # family words that also occur as names in English subtitles
    "nanay", "tatay", "inay", "itay", "ina", "ama", "kuya", "lola", "lolo",
    "tita", "tito", "bunso", "pinsan", "pera",
}

# Words common in both languages: decided by the neighbouring words.
AMBIGUOUS = {"may", "at", "ate", "pang", "bang", "sale", "pare", "dun"}

# A Tagalog prefix written apart from its root ("makapag wait") is TL.
BARE_PREFIXES = {
    "mag", "nag", "pag", "naka", "maka", "makapag", "nakapag", "magpa",
    "nagpa", "pinag", "nagre", "mache", "ipa", "ipapa", "ipag", "nakaka",
    "nakakapag", "makiki", "nakiki", "pinaka",
}

# Prefixes stripped (longest first) to find a root.
PREFIXES = sorted({
    "nakakapag", "nakikipag", "makikipag", "pinakama", "makapag", "nakapag",
    "magpaka", "nagpaka", "pinaka", "nagpa", "magpa", "pinag", "ipina",
    "nakaka", "makaka", "nagka", "magka", "naka", "maka", "naki", "maki",
    "ipag", "pag", "mag", "nag", "pa", "ma", "na", "ka", "i", "ipa",
    "nagre", "mache",
}, key=len, reverse=True)
SUFFIXES = ["han", "hin", "an", "in"]

# Whole-word contractions: lowercase form -> list of (piece, expansion, tag).
WHOLE = {
    "'cause": [("'cause", "because", "EN")],
    "'cuz": [("'cuz", "because", "EN")],
    "cuz": [("cuz", "because", "EN")],
    "'til": [("'til", "until", "EN")],
    "'em": [("'em", "them", "EN")],
    "'bout": [("'bout", "about", "EN")],
    "'to": [("'to", "ito", "TL")],
    "'yan": [("'yan", "iyan", "TL")],
    "'yun": [("'yun", "iyon", "TL")],
    "'yon": [("'yon", "iyon", "TL")],
    "'yung": [("'yung", "iyong", "TL")],
    "'di": [("'di", "hindi", "TL")],
    "'wag": [("'wag", "huwag", "TL")],
    "'pag": [("'pag", "kapag", "TL")],
    "'kaw": [("'kaw", "ikaw", "TL")],
    "'la": [("'la", "wala", "TL")],
    "di'ba": [("di", "hindi", "TL"), ("'ba", "ba", "TL")],
    "diba": [("di", "hindi", "TL"), ("ba", "ba", "TL")],
}
# "sa'yo", "sa'kin", ...: sa + shortened pronoun.
SA_FORMS = {"yo": "iyo", "kin": "akin", "king": "aking", "min": "amin",
            "tin": "atin", "nyo": "inyo", "inyo": "inyo", "kanya": "kanya", "nya": "niya"}
# Suffix contractions after a vowel: "ako'y" (ay), "ako't" (at), "ano'ng" (ang).
SUFFIX_FORMS = {"y": "ay", "t": "at", "ng": "ang"}
# English endings after an apostrophe: don't, I'm, you're, we'll, I've, I'd, registrar's.
EN_ENDINGS = {"t", "s", "m", "re", "ve", "ll", "d"}

TOKEN_RE = re.compile(r"'?[^\W_]+(?:['-][^\W_]+)*")


def normalise(text):
    return text.replace("’", "'").replace("‘", "'").replace("`", "'")


def tokenize(text):
    return TOKEN_RE.findall(normalise(text))


@lru_cache(maxsize=1)
def _lexicon():
    def load(name):
        counts = {}
        path = LEXICON_DIR / name
        for line in path.read_text(encoding="utf-8").splitlines():
            word, count = line.rsplit(" ", 1)
            counts[word] = int(count)
        return counts, sum(counts.values())
    return load("tl_freq.txt"), load("en_freq.txt")


@lru_cache(maxsize=1)
def _slang():
    """{word: tag} from the slang list. Multi-word phrases are skipped (each
    word is tagged on its own); rows whose status is "remove" are ignored."""
    path = LEXICON_DIR / "slang.csv"
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                word = row["word"].strip().lower()
                if " " in word or row.get("status", "").strip().lower() == "remove":
                    continue
                if row["tag"] in ("EN", "TL", "NE"):
                    out[word] = row["tag"]
    return out


@lru_cache(maxsize=1)
def _dictionary():
    """Tagalog words from the optional Wiktionary-based dictionary."""
    path = LEXICON_DIR / "tl_dict.txt"
    if not path.exists():
        return frozenset()
    return frozenset(w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())


def strip_accents(word):
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def word_score(word):
    """log(relative frequency in Tagalog / in English).

    Returns None if the word is in neither list, +inf if only Tagalog,
    -inf if only English.
    """
    (tl, tl_total), (en, en_total) = _lexicon()
    in_tl, in_en = word in tl, word in en
    if not in_tl and not in_en:
        return None
    if in_tl and not in_en:
        return math.inf
    if in_en and not in_tl:
        return -math.inf
    if tl[word] < MIN_TL_COUNT:
        return -math.inf  # too little Tagalog evidence against a known English word
    return math.log((tl[word] / tl_total) / (en[word] / en_total))


def _lookup(word):
    """Tag a plain word from the word lists, or None if unknown.

    Returns (tag, how). tag may be 'AMB' for words common in both languages.
    """
    slang = _slang().get(word)
    if slang:
        return slang, "slang list"
    if word in AMBIGUOUS:
        return "AMB", "common in both languages"
    if word in EXTRA_TL or word in V1_TL_WORDS:
        return "TL", "Tagalog word list"
    score = word_score(word)
    if score is None:
        plain = strip_accents(word)
        if word in _dictionary() or plain in _dictionary():
            return "TL", "Tagalog dictionary (Wiktionary)"
        return None
    if score >= math.log(STRONG_RATIO):
        return "TL", "Tagalog frequency list"
    return "EN", "English frequency list"


def _roots(word):
    """Candidate roots after removing Tagalog affixes (most likely first)."""
    seen, out = set(), []

    def add(root, how):
        if len(root) >= 3 and root not in seen:
            seen.add(root)
            out.append((root, how))

    def strip_infix(w):
        # k-um-ain -> kain, s-in-ulat -> sulat
        if len(w) > 4 and w[0] not in "aeiou" and w[1:3] in ("um", "in"):
            return w[0] + w[3:]
        return None

    def strip_redup(w):
        # kakain -> kain, mag-uusap style: first CV repeated
        if len(w) > 4 and w[:2] == w[2:4]:
            return w[2:]
        if len(w) > 3 and w[0] in "aeiou" and w[0] == w[1]:
            return w[1:]
        return None

    frontier = [(word, "")]
    for _ in range(3):
        nxt = []
        for w, how in frontier:
            for p in PREFIXES:
                if w.startswith(p) and len(w) - len(p) >= 3:
                    r = w[len(p):]
                    add(r, f"{how}{p}-")
                    nxt.append((r, f"{how}{p}-"))
            for f in (strip_infix, strip_redup):
                r = f(w)
                if r:
                    add(r, f"{how}affix ")
                    nxt.append((r, f"{how}affix "))
            for s in SUFFIXES:
                if w.endswith(s) and len(w) - len(s) >= 3:
                    add(w[: -len(s)], f"{how}-{s} ")
        frontier = nxt
    return out


def is_prefix_chunk(chunk, root):
    """True if `chunk` (the part before a hyphen) is Tagalog prefixes, plus at
    most a short repeated piece: the start of the root ("re" in magre-review,
    "s"/"sho" in nagsho-shopping) or the prefix's own last syllable repeated
    ("pa" in magpapa-check, "ka" in nakaka-stress)."""
    def strip(c, depth=0):
        if c in PREFIXES or c in BARE_PREFIXES:
            return [""]
        rests = []
        for p in PREFIXES:
            if c.startswith(p) and len(c) > len(p):
                rests.append(c[len(p):])
                if depth < 1:
                    rests.extend(r for r in strip(c[len(p):], depth + 1) if r is not None)
        return rests
    if chunk in PREFIXES or chunk in BARE_PREFIXES:
        return True
    for rest in strip(chunk):
        if rest == "":
            return True
        vowels = sum(ch in "aeiou" for ch in rest)
        if len(rest) <= 3 and vowels <= 1:
            if root.startswith(rest[0]) or chunk[:-len(rest)].endswith(rest):
                return True
    return False


def _tag_word(word):
    """(tag, how) for one lowercase word with no apostrophe."""
    if word.isdigit() or word in NE_WORDS:
        return "NE", "number or name"
    if word in BARE_PREFIXES:
        return "TL", "Tagalog prefix on its own"
    found = _lookup(word)
    if found:
        return found
    if "-" in word:
        parts = word.split("-")
        root = "-".join(parts[1:])
        if root and is_prefix_chunk(parts[0], root):
            tag, how = _tag_word(root)
            if tag in ("EN", "TL"):
                return tag, f"root '{root}' decides ({how})"
        if len(parts) == 2 and parts[0] == parts[1]:
            return _tag_word(parts[0])
        tags = {_tag_word(p)[0] for p in parts}
        if len(tags) == 1 and tags <= {"EN", "TL"}:
            return tags.pop(), "all parts agree"
    for root, how in _roots(word):
        found = _lookup(root)
        if found and found[0] in ("EN", "TL"):
            return found[0], f"root '{root}' decides ({how.strip()})"
    # Prefix + repeated first letters of an English root, written as one word:
    # "magdadrive" = mag + da + drive, "nagsheshare" = nag + she + share.
    for p in PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 5:
            rest = word[len(p):]
            for k in (1, 2, 3):
                piece, root = rest[:k], rest[k:]
                if len(root) >= 3 and root.startswith(piece[0]) and _lookup(root) and _lookup(root)[0] == "EN":
                    return "EN", f"root '{root}' decides ({p}- + repeated '{piece}')"
    return "UNK", "not in the word lists"


def _split_token(token):
    """Turn one surface token into Token pieces (contractions split)."""
    low = token.lower()
    if low in WHOLE:
        pieces, start = [], 0
        for piece, norm, tag in WHOLE[low]:
            text = token[start:start + len(piece)] if len(WHOLE[low]) > 1 else token
            start += len(piece)
            pieces.append(Token(text, norm, tag, "contraction"))
        return pieces

    if "'" in low:
        base, _, tail = low.rpartition("'")
        surface_base = token[: len(base)]
        if base == "sa" and tail in SA_FORMS:
            return [Token(token[:2], "sa", "TL", "contraction"),
                    Token(token[2:], SA_FORMS[tail], "TL", "contraction")]
        if tail in SUFFIX_FORMS and base and base[-1] in "aeiou":
            btag, bhow = _tag_word(base)
            if btag in ("TL", "UNK", "AMB"):
                return [Token(surface_base, base, "TL" if btag == "UNK" else btag, bhow),
                        Token(token[len(base):], SUFFIX_FORMS[tail], "TL", "contraction")]
        if tail in EN_ENDINGS:
            btag, _ = _tag_word(base) if base else ("UNK", "")
            if btag in ("EN", "AMB") or base in {"i", "don", "can", "won", "isn", "aren",
                                                  "doesn", "didn", "wasn", "weren",
                                                  "couldn", "wouldn", "shouldn", "ain"}:
                return [Token(token, low, "EN", "English contraction")]
        if low.startswith("'") and _tag_word(low[1:])[0] == "EN":
            return [Token(token, low[1:], "EN", "shortened English word")]
        if base == "ma" and tail == "am":
            return [Token(token, "ma'am", "EN", "English word")]

    tag, how = _tag_word(low)
    return [Token(token, low, tag, how)]


# English "may" is a modal: "May I...", "you may be...". Tagalog "may" means
# "there is / has" and is usually followed by a noun, often an English one
# ("may unpaid balance"), so neighbours alone would mislabel it.
MAY_EN_NEXT = {"i", "you", "we", "he", "she", "it", "they", "be", "not", "have",
               "also", "well", "still", "never", "need", "ask", "help", "come", "go"}


def _resolve_ambiguous(tokens):
    """AMB tokens take the majority language of up to 2 neighbours each side."""
    tags = [t.tag for t in tokens]
    out = []
    for i, t in enumerate(tokens):
        if t.tag != "AMB":
            out.append(t)
            continue
        if t.norm == "may":
            nxt = tokens[i + 1].norm if i + 1 < len(tokens) else ""
            tag = "EN" if nxt in MAY_EN_NEXT else "TL"
            out.append(t._replace(tag=tag, how="'may' decided by the next word"))
            continue
        near = [tags[j] for j in range(max(0, i - 2), min(len(tags), i + 3))
                if j != i and tags[j] in ("EN", "TL")]
        en, tl = near.count("EN"), near.count("TL")
        if en != tl:
            tag = "EN" if en > tl else "TL"
            how = "decided by neighbouring words"
        else:
            score = word_score(t.norm)
            tag = "TL" if (score is not None and score > 0) else "EN"
            how = "more frequent language (no clear neighbours)"
        out.append(t._replace(tag=tag, how=how))
    return out


def _mark_names(tokens, raw_tokens_first_index):
    """Unknown capitalised words that don't start the sentence are names (NE)."""
    out = []
    for i, t in enumerate(tokens):
        if t.tag == "UNK" and t.text[:1].isupper() and i not in raw_tokens_first_index:
            out.append(t._replace(tag="NE", how="capitalised unknown word (name)"))
        else:
            out.append(t)
    return out


def tag_sentence(text):
    """Return a list of Token(text, norm, tag, how). Tags: EN, TL, NE, UNK."""
    text = normalise(text)
    tokens, first = [], set()
    for match in TOKEN_RE.finditer(text):
        before = text[: match.start()].rstrip()
        if not before or before[-1] in ".!?":
            first.add(len(tokens))
        tokens.extend(_split_token(match.group(0)))
    tokens = _mark_names(tokens, first)
    return _resolve_ambiguous(tokens)


def analyse(text):
    """Tokens plus density, ignoring NE and UNK, and how many are unknown."""
    tokens = tag_sentence(text)
    tags = [t.tag for t in tokens]
    return {"tokens": tokens, "density": density(tags),
            "n_en": tags.count("EN"), "n_tl": tags.count("TL"),
            "n_unknown": tags.count("UNK")}
