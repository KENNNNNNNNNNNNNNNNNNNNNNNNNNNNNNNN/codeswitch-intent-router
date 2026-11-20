"""Web server tests. The model is replaced by a fake, so no key or network is used."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server


def fake_client(reply):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def api(monkeypatch):
    def use(reply):
        server._recent.clear()
        server.client.cache_clear()
        server._analyze_cached.cache_clear()
        server._translate_cached.cache_clear()
        monkeypatch.setattr(server, "make_client", lambda: fake_client(reply))
        return TestClient(server.app)
    return use


@pytest.mark.parametrize("path, marker", [
    ("/", 'data-page="home"'), ("/how", 'id="how"'),
    ("/simulation", 'id="sim-form"'), ("/evidence", 'id="ablation-table"')])
def test_each_page_is_served_with_layout_and_backdrop(api, path, marker):
    html = api("{}").get(path).text
    assert marker in html and 'class="site-header"' in html and 'class="backdrop"' in html
    assert "{{" not in html  # every placeholder was filled


def test_current_page_is_marked_in_the_nav(api):
    html = api("{}").get("/evidence").text
    assert 'href="/evidence" aria-current="page"' in html
    assert 'href="/how" aria-current' not in html


def test_assets_are_served(api):
    c = api("{}")
    for path in ("/static/style.css", "/static/app.js"):
        assert c.get(path).status_code == 200


def test_analyze_with_preset(api):
    c = api(json.dumps({"intent": "tuition_payment", "confidence": 0.9}))
    r = c.post("/api/analyze", json={"text": "Paano magbayad ng tuition?",
                                     "preset": "campus_registrar", "threshold": 0.95})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "tuition_payment" and body["handoff"] is True
    assert body["calibrated"] is True


def test_analyze_with_custom_intents_and_unknown_word(api):
    c = api(json.dumps({"intent": "love_life", "confidence": 0.8, "word_languages": {"chorva": "TL"}}))
    r = c.post("/api/analyze", json={"text": "Sana all may chorva", "preset": "custom",
                                     "intents": ["love life", "work"]}).json()
    assert r["intent"] == "love_life" and r["n_unknown"] == 0 and r["calibrated"] is False


def test_analyze_returns_formal_writing_hints(api):
    c = api(json.dumps({"intent": "clearance", "confidence": 0.9}))
    r = c.post("/api/analyze", json={"text": "Pupunta daw ako sa registrar", "preset": "campus_registrar"}).json()
    assert r["hints"][0]["suggestion"] == "raw"


def test_custom_needs_two_intents(api):
    c = api("{}")
    r = c.post("/api/analyze", json={"text": "hi", "preset": "custom", "intents": ["only"]})
    assert r.status_code == 422


def test_language_only_mode_makes_no_handoff_decision(api):
    c = api("{}")
    r = c.post("/api/analyze", json={"text": "Kumain ka na ba?", "preset": "none"}).json()
    assert r["uses_intents"] is False and r["handoff"] is None


def test_unknown_preset_and_empty_text_are_rejected(api):
    c = api("{}")
    assert c.post("/api/analyze", json={"text": "hi", "preset": "nope"}).status_code == 422
    assert c.post("/api/analyze", json={"text": "   ", "preset": "none"}).status_code == 422


def test_translate_returns_both_languages_from_one_call(api):
    c = api('{"english": "I am hungry.", "tagalog": "Gutom na ako."}')
    r = c.post("/api/translate", json={"text": "I am gutom na"}).json()
    assert r["english"] == "I am hungry." and r["tagalog"] == "Gutom na ako." and r["error"] == ""
    assert c.post("/api/translate", json={"text": "   "}).status_code == 422


def test_missing_key_gives_a_clear_error(monkeypatch):
    server._recent.clear()
    server.client.cache_clear()
    server._analyze_cached.cache_clear()

    def no_key():
        raise RuntimeError("GROQ_API_KEY is not set.")
    monkeypatch.setattr(server, "make_client", no_key)
    r = TestClient(server.app).post("/api/analyze", json={"text": "hi", "preset": "campus_registrar"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_presets_and_evidence(api):
    c = api("{}")
    p = c.get("/api/presets").json()
    assert "campus_registrar" in p["presets"] and p["max_chars"] == 2000
    ev = c.get("/api/evidence").json()
    assert ev["tagger"][0]["v2"]["total"] > 0
    assert len(ev["buckets"]) == 3 and len(ev["wrong"]) >= 1


def test_visitor_speed_limit(api, monkeypatch):
    c = api('{"english": "Hi.", "tagalog": "Kumusta."}')
    monkeypatch.setattr(server, "RATE_LIMIT", 3)
    codes = [c.post("/api/translate", json={"text": f"hi {i}"}).status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_website_does_not_need_scikit_learn():
    """The slim requirements.txt for Vercel has no scikit-learn; make sure the
    modules the server imports don't load it."""
    import subprocess
    import sys
    code = ("import sys, server; "
            "print('sklearn' in sys.modules, 'matplotlib' in sys.modules, 'streamlit' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False False"
