"""Day 7 demo. Run from the repo root:  streamlit run app.py"""
import html

import streamlit as st

from src.llm_client import DEFAULT_MODEL, make_client
from src.router import analyze, decide, preregistered_threshold

st.set_page_config(page_title="Taglish Router", page_icon="🔀")

TAG_COLOURS = {"EN": "#2a78d6", "TL": "#eb6834", "NE": "#8a8980"}
TAG_NAMES = {"EN": "English", "TL": "Tagalog", "NE": "name / neutral"}


@st.cache_resource
def get_client():
    return make_client()


@st.cache_data(show_spinner=False)
def cached_analyze(text, model):
    # Cached so moving the slider, or re-submitting the same text, costs no API call.
    return analyze(text, get_client(), model)


st.title("Taglish Router")
st.caption("Campus registrar intent routing for English, Tagalog and Taglish messages. "
           "Research demo on a small synthetic dataset; not for real use.")

rule = preregistered_threshold()
with st.sidebar:
    st.header("Handoff threshold")
    threshold = st.slider("Hand off to staff when confidence is below", 0.0, 1.0,
                          value=rule, step=0.01)
    st.caption(f"Pre-registered rule (Day 6): **{rule:.2f}**. That is the lowest threshold "
               "giving ≥95% accuracy on train+validation. At 0.00 the system never hands "
               "off on confidence; move the slider to explore other values.")
    if abs(threshold - rule) > 1e-9:
        st.warning("You are not using the pre-registered threshold.")
    st.divider()
    st.caption(f"Model: `{DEFAULT_MODEL}` via Groq")

text = st.text_area("Message to the registrar",
                    placeholder="e.g. Pwede ba magbayad ng tuition using GCash?")
if st.button("Route message", type="primary") and not text.strip():
    st.info("Type a message first.")

if text.strip():
    try:
        with st.spinner("Asking the model..."):
            a = cached_analyze(text.strip(), DEFAULT_MODEL)
    except RuntimeError as err:  # missing API key
        st.error(str(err))
        st.stop()
    except Exception as err:  # noqa: BLE001 - show any API problem to the user
        st.error(f"The model call failed: {type(err).__name__}: {err}")
        st.stop()

    handoff, reason = decide(a, threshold)
    if handoff:
        st.error(f"**Handoff to staff.** Reason: {reason}.")
    else:
        st.success(f"**Answered automatically.** Reason: {reason}.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted intent", a["intent"] or "none")
    c2.metric("Confidence", "n/a" if a["confidence"] is None else f"{a['confidence']:.2f}")
    c3.metric("Code-switch density", f"{a['density']:.2f}", a["bucket"][2:], delta_color="off")

    st.subheader("Token languages")
    chips = " ".join(
        f"<span style='border:1.5px solid {TAG_COLOURS[tag]};border-radius:6px;"
        f"padding:2px 6px;margin:2px;display:inline-block'>{html.escape(tok)} "
        f"<small style='color:{TAG_COLOURS[tag]}'>{tag}</small></span>"
        for tok, tag in a["tokens"])
    st.markdown(chips, unsafe_allow_html=True)
    st.caption(" · ".join(f"{k} = {v}" for k, v in TAG_NAMES.items())
               + f" · switch rate {a['switch_rate']:.2f}")

    with st.expander("Raw model reply"):
        st.code(a["raw_reply"] or "(empty)")

st.divider()
st.caption("On the 120-sentence dataset the model's stated confidence was ≥0.95 for 116 "
           "answers, including wrong ones, so a high confidence here is not a guarantee. "
           "See the README for results and limitations.")
