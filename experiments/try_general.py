"""Try the general analysis and the translator from the terminal (real API calls).

    python experiments/try_general.py "Ako'y gutom na, let's eat"
    python experiments/try_general.py "Sana all may jowa" --intents love_life work other
    python experiments/try_general.py "Nasaan na order ko?" --preset customer_support
    python experiments/try_general.py "Ako'y gutom na" --translate english

Without --intents or --preset no intent is predicted, and the API is only
called if the tagger finds unknown words.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import make_client  # noqa: E402
from src.router import PRESETS, analyze_general  # noqa: E402
from src.translate import translate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=sorted(PRESETS))
    group.add_argument("--intents", nargs="+", metavar="NAME")
    parser.add_argument("--translate", choices=["english", "tagalog"])
    args = parser.parse_args()

    client = make_client()
    intents = PRESETS[args.preset] if args.preset else args.intents
    r = analyze_general(args.text, client, intents)
    print(" ".join(f"{t['text']}/{t['tag']}" for t in r["tokens"]))
    for t in r["tokens"]:
        if t["how"] == "decided by the LLM":
            print(f"  LLM tagged {t['text']!r} as {t['tag']}")
    print(f"density {r['density']:.2f} ({r['bucket'][2:]}), still unknown: {r['n_unknown']}")
    if r["uses_intents"]:
        print(f"intent: {r['intent']}  confidence: {r['confidence']}  {r['parse_error']}")
    if args.translate:
        t = translate(client, args.text, args.translate)
        print(f"{args.translate}: {t['translation']}  {t['error']}")


if __name__ == "__main__":
    main()
