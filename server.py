"""Web app for the Taglish Router. Run from the repo root:

    uvicorn server:app --port 8000

then open http://localhost:8000. The API key stays on this server; the
browser never sees it. The server only listens on this computer.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.llm_client import DEFAULT_MODEL, make_client, normalise_intents
from src.router import PRESETS, analyze_general, decide, preregistered_threshold
from src.translate import MAX_CHARS, translate_both

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
RESULTS = ROOT / "results"
MODEL_SLUG = DEFAULT_MODEL.replace("/", "_")

app = FastAPI(title="Taglish Router")
app.mount("/static", StaticFiles(directory=WEB), name="static")


PAGES = {
    "home": ("/", "Taglish Router"),
    "how": ("/how", "How it works | Taglish Router"),
    "simulation": ("/simulation", "Simulation | Taglish Router"),
    "evidence": ("/evidence", "Evidence | Taglish Router"),
}


def render_page(name):
    """Fill the shared layout with one page. Read from disk each time, so
    edits to the HTML files show up on refresh."""
    html = (WEB / "layout.html").read_text(encoding="utf-8")
    title = PAGES[name][1]
    html = html.replace("{{title}}", title).replace("{{page}}", name)
    for other in ("how", "simulation", "evidence"):
        html = html.replace(f"{{{{current_{other}}}}}", ' aria-current="page"' if other == name else "")
    html = html.replace("{{backdrop}}", (WEB / "backdrop.html").read_text(encoding="utf-8"))
    return html.replace("{{content}}", (WEB / "pages" / f"{name}.html").read_text(encoding="utf-8"))


def _page_route(name):
    def handler():
        return HTMLResponse(render_page(name))
    return handler


for _name, (_path, _title) in PAGES.items():
    app.get(_path, response_class=HTMLResponse, name=_name)(_page_route(_name))


# ---- model calls --------------------------------------------------------------

@lru_cache(maxsize=1)
def client():
    return make_client()


def get_client():
    try:
        return client()
    except RuntimeError as err:  # key missing
        raise HTTPException(503, str(err)) from err


def api_error(err):
    name = type(err).__name__
    if name == "RateLimitError":
        return HTTPException(429, "The model's free-tier limit was reached. Wait a minute and try again.")
    if name == "AuthenticationError":
        return HTTPException(503, "The Groq API key was rejected. Store a valid key and restart the server.")
    return HTTPException(502, f"The model call failed ({name}).")


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    preset: str | None = None          # a key of PRESETS, "custom" or "none"
    intents: list[str] = []            # used when preset == "custom"
    threshold: float | None = Field(default=None, ge=0, le=1)


@lru_cache(maxsize=256)
def _analyze_cached(text, intents_key):
    intents = dict(intents_key) if intents_key else None
    return analyze_general(text, get_client(), intents)


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Type a message first.")
    if body.preset in (None, "none"):
        intents = None
    elif body.preset == "custom":
        try:
            intents = normalise_intents(body.intents) if body.intents else None
        except ValueError as err:
            raise HTTPException(422, str(err)) from err
        if intents and len(intents) < 2:
            raise HTTPException(422, "Give at least two intents, separated by commas.")
    elif body.preset in PRESETS:
        intents = normalise_intents(PRESETS[body.preset])
    else:
        raise HTTPException(422, f"Unknown preset {body.preset!r}.")

    key = tuple(sorted(intents.items())) if intents else None
    try:
        result = dict(_analyze_cached(text, key))
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        raise api_error(err) from err

    rule = preregistered_threshold(RESULTS / "handoff_threshold.csv")
    threshold = rule if body.threshold is None else body.threshold
    result["threshold"] = threshold
    result["calibrated"] = body.preset == "campus_registrar"
    if result["uses_intents"]:
        result["handoff"], result["reason"] = decide(result, threshold)
    else:
        result["handoff"], result["reason"] = None, "No intents were given, so there is nothing to route."
    return result


class TranslateIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


@lru_cache(maxsize=256)
def _translate_cached(text):
    return translate_both(get_client(), text)


@app.post("/api/translate")
def translate_endpoint(body: TranslateIn):
    """Plain English and plain Tagalog versions, from one model call."""
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Type a message first.")
    try:
        return _translate_cached(text)
    except HTTPException:
        raise
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    except Exception as err:  # noqa: BLE001
        raise api_error(err) from err


@app.get("/api/presets")
def presets():
    return {
        "presets": {name: normalise_intents(i) for name, i in PRESETS.items()},
        "threshold": preregistered_threshold(RESULTS / "handoff_threshold.csv"),
        "model": DEFAULT_MODEL,
        "max_chars": MAX_CHARS,
    }


# ---- evidence -------------------------------------------------------------------

def _records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _read(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@lru_cache(maxsize=1)
def tagger_scores():
    """v1 vs v2 on both gold sets, computed live from the gold files."""
    from src import tagger as v1
    from src import tagger_v2 as v2

    def run(texts, gold, tag):
        right = total = 0
        for sid, rows in gold.groupby("id"):
            expected = list(rows.sort_values("token_index")[["token", "tag"]].itertuples(index=False))
            got = tag(texts[sid])
            if [g[0] for g in got] != [e[0] for e in expected]:
                continue
            for (_, t), (_, g) in zip(got, expected):
                total += 1
                right += t == g
        return {"correct": right, "total": total}

    def t1(text):
        return [(t, v1.tag_token(t)) for t in v1.tokenize(text)]

    def t2(text):
        return [(t.text, t.tag) for t in v2.tag_sentence(text)]

    out = []
    data = pd.read_csv(ROOT / "data/taglish_dataset.csv").set_index("id")["text"]
    gold = pd.read_csv(ROOT / "data/gold_tokens.csv")
    out.append({"set": "Registrar messages", "sentences": int(gold["id"].nunique()),
                "v1": run(data, gold, t1), "v2": run(data, gold, t2)})
    gpath = ROOT / "data/gold_tokens_general.csv"
    if gpath.exists():
        texts = pd.read_csv(ROOT / "data/gold_sentences_general.csv").set_index("id")["text"]
        g = pd.read_csv(gpath)
        out.append({"set": "Everyday Taglish", "sentences": int(g["id"].nunique()),
                    "v1": run(texts, g, t1), "v2": run(texts, g, t2)})
    return out


@app.get("/api/evidence")
def evidence():
    out = {"model": DEFAULT_MODEL, "tagger": tagger_scores()}
    ablation = _read(f"ablation_test_{MODEL_SLUG}.csv")
    out["ablation"] = _records(ablation) if ablation is not None else []
    buckets = _read("confidence_by_bucket.csv")
    out["buckets"] = _records(buckets) if buckets is not None else []
    threshold = _read("handoff_threshold.csv")
    out["threshold"] = _records(threshold) if threshold is not None else []
    coverage = _read("handoff_coverage.csv")
    out["coverage"] = _records(coverage[coverage["data"] == "dev"]) if coverage is not None else []

    frames = []
    for split in ("train", "validation", "test"):
        df = _read(f"llm_zero_shot_{split}_{MODEL_SLUG}.csv")
        if df is not None:
            frames.append(df.assign(split=split))
    wrong = []
    if frames:
        allp = pd.concat(frames)
        out["n_sentences"] = len(allp)
        w = allp[allp["pred"] != allp["intent"]].sort_values("confidence", ascending=False)
        wrong = _records(w[["text", "intent", "pred", "confidence", "density", "split"]])
    out["wrong"] = wrong
    return out
