"""Phase 6 — Streamlit UI smoke tests (PRD §9) via streamlit.testing.AppTest.

Offline and deterministic: the Mistral path is disabled so answers come from the
grounded extractive fallback (FR-5). Requires the local Chroma index.
"""

import pytest
from streamlit.testing.v1 import AppTest

from code.config import VECTOR_DB_DIR


pytestmark = pytest.mark.skipif(
    not (VECTOR_DB_DIR / "chroma.sqlite3").exists(),
    reason="Chroma index not built; run python -m code.ingest",
)


@pytest.fixture(autouse=True)
def no_mistral(monkeypatch):
    monkeypatch.setattr(
        "code.retrieval._call_mistral",
        lambda _q, _c: (_ for _ in ()).throw(RuntimeError("missing_key")),
        raising=True,
    )


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file("code/app.py", default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _last_assistant_text(app) -> str:
    return app.chat_message[-1].markdown[0].value


def test_app_loads_with_welcome_and_corpus(app):
    joined = " ".join(m.value for m in app.markdown)
    assert "Facts-only. No investment advice." in joined
    assert "Ask factual questions about five HDFC Direct Growth funds" in joined
    assert len(app.chat_input) == 1
    assert any(("HDFC ELSS Tax Saver" in b.label) for b in app.button)


def test_example_chip_asks_and_grounds_answer(app):
    assert len(app.chat_message) == 0
    app.button[0].click().run()
    assert len(app.chat_message) == 2
    assert app.chat_message[0].name == "user"
    assert "expense ratio" in _last_assistant_text(app).lower()
    assert "1.02%" in _last_assistant_text(app)


def test_typed_question_returns_grounded_fact(app):
    app.chat_input[0].set_value("What is the minimum SIP for HDFC ELSS Tax Saver?").run()
    assert len(app.chat_message) == 4
    text = _last_assistant_text(app)
    assert "₹500" in text
    assert "hdfc-elss-tax-saver-fund-direct-plan-growth" in text
    assert "Last updated from sources:" in text


def test_citation_rendered_clickable(app):
    rendered = _last_assistant_text(app)
    assert (
        "[https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth]"
        "(https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth)" in rendered
    )


def test_guardrail_question_is_blocked_with_caption(app):
    app.chat_input[0].set_value("Should I buy HDFC Small Cap Fund?").run()
    msg = app.chat_message[-1]
    assert "cannot give buy, sell, or portfolio advice" in msg.markdown[0].value
    assert any("Guardrail: advice" in c.value for c in msg.caption)


def test_capital_gains_question_abstains(app):
    app.chat_input[0].set_value("How to download capital-gains statement?").run()
    text = _last_assistant_text(app)
    assert "not described in the five allowed" in text


def test_clear_conversation_resets_chat(app):
    messages = app.session_state["messages"]
    assert len(messages) > 0
    clear = next(b for b in app.button if b.label == "Clear conversation")
    clear.click().run()
    assert len(app.session_state["messages"]) == 0
    assert len(app.chat_message) == 0