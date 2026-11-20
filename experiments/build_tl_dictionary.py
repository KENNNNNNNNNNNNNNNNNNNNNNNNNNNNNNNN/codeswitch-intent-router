"""Build data/lexicon/tl_dict.txt from the Wiktionary Tagalog data on kaikki.org.

1. In your browser, open https://kaikki.org/dictionary/Tagalog/ and download
   the JSONL file with all Tagalog words (link near the bottom of the page).
   Save it outside the repo, for example in your Files-TR folder.
2. From the repo root run:
       python experiments/build_tl_dictionary.py "C:\\path\\to\\the-downloaded-file.jsonl"

The input may also be gzip-compressed (.gz). The output is a plain list of
lowercase Tagalog word forms, one per line, which the tagger uses only for
words that are not in the frequency lists.

Licence: Wiktionary text is available under CC BY-SA. The output is a derived
work under the same licence; see data/lexicon/SOURCE.md.
"""
import argparse
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

OUT = Path("data/lexicon/tl_dict.txt")
WORD_RE = re.compile(r"^[a-z]+(?:[-'][a-z]+)*$")
# Form tags that are not ordinary spellings of the word.
SKIP_FORM_TAGS = {"Baybayin", "romanization", "table-tags", "inflection-template", "class"}


def plain(word):
    """Lowercase and remove accent marks (pantáw -> pantaw)."""
    word = unicodedata.normalize("NFD", word.strip().lower())
    return "".join(c for c in word if unicodedata.category(c) != "Mn")


def borrowed_from_english(entry):
    """True for entries whose etymology says the word came from English.

    These are left out: spelled the English way they should be tagged EN, and
    respelled forms are rare enough to leave to the model.
    """
    text = (entry.get("etymology_text") or "").lower()
    return "from english" in text


def words_from(entry):
    yield entry.get("word", "")
    for form in entry.get("forms", []) or []:
        tags = set(form.get("tags", []) or [])
        if tags & SKIP_FORM_TAGS:
            continue
        yield form.get("form", "")


def build(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    kept, seen_entries, skipped_english, skipped_names = set(), 0, 0, 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("lang_code") != "tl":
                continue
            seen_entries += 1
            if entry.get("pos") == "name":
                skipped_names += 1
                continue
            if borrowed_from_english(entry):
                skipped_english += 1
                continue
            for w in words_from(entry):
                w = plain(w)
                if len(w) >= 2 and WORD_RE.match(w):
                    kept.add(w)
    return kept, seen_entries, skipped_english, skipped_names


def main():
    parser = argparse.ArgumentParser(description="Build the Tagalog dictionary word list.")
    parser.add_argument("jsonl", help="downloaded kaikki.org Tagalog JSONL file (.jsonl or .jsonl.gz)")
    args = parser.parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    words, entries, english, names = build(path)
    if not words:
        sys.exit("No Tagalog entries found. Is this the Tagalog JSONL file from kaikki.org?")
    OUT.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(f"read {entries} Tagalog entries; left out {english} borrowed from English "
          f"and {names} proper names")
    print(f"wrote {len(words)} word forms to {OUT}")


if __name__ == "__main__":
    main()
