"""Phase 6 — Retrieval testing against the local Chroma index (no Mistral)."""

import pytest

from code.config import VECTOR_DB_DIR
from code.retrieval import answer


pytestmark = pytest.mark.skipif(
    not (VECTOR_DB_DIR / "chroma.sqlite3").exists(),
    reason="Chroma index not built; run python -m code.ingest",
)


@pytest.fixture(autouse=True)
def no_mistral(monkeypatch):
    """Keep Phase 6 deterministic and offline (FR-5): force the extractive fallback."""

    def _fail_mistral(_question, _context):
        raise RuntimeError("missing_key")

    monkeypatch.setattr("code.retrieval._call_mistral", _fail_mistral)


def test_large_cap_expense_ratio():
    result = answer("What is the expense ratio of HDFC Large Cap Fund Direct Growth?")
    assert not result.blocked
    assert "1.02%" in result.text
    assert "hdfc-large-cap-fund-direct-growth" in result.citation
    assert "Last updated from sources:" in result.text


def test_elss_min_sip():
    result = answer("What is the minimum SIP for HDFC ELSS Tax Saver?")
    assert "₹500" in result.text
    assert "elss-tax-saver" in result.citation


def test_small_cap_exit_load():
    result = answer("What is the exit load of HDFC Small Cap Fund Direct Growth?")
    assert "1%" in result.text
    assert "1 year" in result.text.lower()
    assert "small-cap-fund-direct-growth" in result.citation


def test_flexi_nav_quoted_with_as_of():
    result = answer("What is the NAV of HDFC Flexi Cap Fund?")
    assert not result.blocked
    assert "₹" in result.text
    assert "NAV as of" in result.text or "as of 28 Aug 2026" in result.text
    assert "equity-fund-direct-growth" in result.citation


def test_capital_gains_statement_abstains():
    result = answer("How to download capital-gains statement?")
    assert not result.blocked
    assert "does not describe" in result.text.lower()
    assert result.text.lower().count("source:") == 1
    assert "Last updated from sources:" in result.text
    for bad in ("step 1", "go to settings", "tap on", "click "):
        assert bad not in result.text.lower()


def test_elss_lock_in_not_invented():
    result = answer("Does HDFC ELSS have a lock-in period?")
    assert not result.blocked
    assert "lock-in" in result.text.lower()
    assert "3 year" not in result.text.lower()
    assert "3-year" not in result.text.lower()
    assert "three year" not in result.text.lower()


def test_ambiguous_expense_ratio_still_grounded_or_abstains():
    result = answer("What is the expense ratio?")
    assert not result.blocked
    assert result.text.lower().count("source:") == 1
    assert "Last updated from sources:" in result.text
    grounded = "expense ratio" in result.text.lower() and result.kind is None
    weakened = result.kind == "no_context"
    assert grounded or weakened


def test_out_of_corpus_fund_does_not_fabricate():
    result = answer("What is the expense ratio of HDFC Multi Cap Fund?")
    assert not result.blocked
    lower = result.text.lower()
    assert "not in the five allowed groww fund pages" in lower or "does not" in lower
    assert "multi cap" not in lower
    assert result.text.count("Source:") == 1