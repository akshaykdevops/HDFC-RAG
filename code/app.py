"""Streamlit UI — Groww HDFC fund FAQ (PRD §9 runtime surface).

Facts-only. No investment advice. Chat state lives only in memory (no persisted
user identity), exactly per the PRD non-goals.
"""

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from code.config import DISCLAIMER, EXAMPLE_QUESTIONS, FUNDS, GROWW_DARK, GROWW_GREEN, VECTOR_DB_DIR, WELCOME
from code.ingest import ingest
from code.retrieval import answer
from code.vector_store import get_collection


def _linkify(text: str) -> str:
    """Make the single Source: <url> citation clickable for display only."""
    return re.sub(r"(Source: )((?:https?)://\S+)", r"\1[\2](\2)", text)


st.set_page_config(page_title="Groww HDFC Fund FAQ", page_icon="🌱", layout="centered")

st.markdown(
    f"""
    <style>
      .groww-banner {{
        background: {GROWW_GREEN};
        color: white;
        padding: 1.1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
      }}
      .groww-banner p {{ margin: 0.35rem 0 0 0; opacity: 0.95; }}
      .disclaimer {{
        opacity: 0.8;
        font-size: 0.9rem;
        margin: 0.5rem 0 1rem 0;
      }}
      .stChatMessage {{ border-radius: 10px; }}
      .stButton>button {{ border-radius: 20px; }}
      .stButton>button:hover {{ border-color: {GROWW_GREEN}; color: {GROWW_GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="groww-banner">
      <strong>Groww · HDFC fund FAQ</strong>
      <p>{WELCOME}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Corpus (5 Groww pages)**")
    for fund in FUNDS:
        st.markdown(f"- [{fund['category']}: {fund['name']}]({fund['url']})")
    st.caption("Public sources only. No app screenshots, no third-party blogs.")
    if st.button("Rebuild index"):
        with st.spinner("Rebuilding index (load → chunk → embed → ChromaDB)…"):
            n = ingest(refresh_from_web=False)
        st.success(f"Indexed {n} chunks.")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def ensure_index() -> int:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    col = get_collection()
    if col.count() == 0:
        return ingest(refresh_from_web=False)
    return col.count()


with st.spinner("Loading fund index…"):
    ensure_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("Try an example:")
cols = st.columns(3)
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i].button(q, key=f"ex_{i}"):
        st.session_state.pending = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(_linkify(msg["content"]))
        if msg.get("caption"):
            st.caption(msg["caption"])

prompt = st.chat_input("Ask a factual fund question")
if pending := st.session_state.pop("pending", None):
    prompt = pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the five Groww pages…"):
            result = answer(prompt)
        st.markdown(_linkify(result.text))
        caption = None
        if result.blocked:
            caption = f"Guardrail: {result.kind} — no factual retrieval attempted."
        elif not result.used_llm:
            caption = "Extractive fallback — add a MISTRAL_API_KEY for LLM-written answers."
        if caption:
            st.caption(caption)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "caption": caption}
    )