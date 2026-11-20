"""Build the word-frequency lists used by the v2 tagger.

Source: FrequencyWords by Hermit Dave (https://github.com/hermitdave/FrequencyWords),
word counts from OpenSubtitles 2018. Content licence: CC BY-SA 4.0.

Run once from the repo root (needs internet). The output files are committed,
so you normally never need to run this:
    python experiments/build_lexicon.py
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"
SOURCES = {
    "tl_freq.txt": (f"{BASE}/tl/tl_full.txt", 2),   # keep words seen at least twice
    "en_freq.txt": (f"{BASE}/en/en_50k.txt", 1),
}
OUT = Path("data/lexicon")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (url, min_count) in SOURCES.items():
        lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
        kept = []
        for line in lines:
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            word, count = parts[0].lower(), int(parts[1])
            if count >= min_count and word.replace("'", "").replace("-", "").isalpha():
                kept.append(f"{word} {count}")
        (OUT / name).write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{name}: {len(kept)} words")


if __name__ == "__main__":
    main()
