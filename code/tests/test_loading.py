"""Phase 1 — Data loading tests (closed corpus, as-of, known gaps)."""

import json

import pytest

from code.config import FUNDS, RAW_DIR
from code.loading import (
    ALLOWED_URLS,
    MANIFEST_FILE,
    assert_allowed_url,
    extract_as_of,
    load_raw_documents,
    run_data_loading,
)


def test_allow_list_is_exactly_five_groww_urls():
    assert len(ALLOWED_URLS) == 5
    assert len(FUNDS) == 5
    assert ALLOWED_URLS == {f["url"] for f in FUNDS}
    assert all(u.startswith("https://groww.in/mutual-funds/") for u in ALLOWED_URLS)
    assert "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth" in ALLOWED_URLS


def test_rejects_url_outside_closed_corpus():
    with pytest.raises(ValueError, match="not in the closed corpus"):
        assert_allowed_url("https://groww.in/mutual-funds/some-other-fund")


def test_load_returns_five_cleaned_documents_with_url_and_as_of():
    docs = run_data_loading(refresh_from_web=False)
    assert len(docs) == 5
    urls = {d.url for d in docs}
    assert urls == ALLOWED_URLS
    for doc in docs:
        assert doc.url in doc.text
        assert doc.source_as_of
        assert "Source URL:" in doc.text
        assert "Page data as of NAV date:" in doc.text
        path = RAW_DIR / doc.corpus_file
        assert path.exists()


def test_persist_writes_manifest():
    run_data_loading(refresh_from_web=False)
    payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    assert payload["phase"] == 1
    assert payload["count"] == 5
    assert set(payload["allowed_urls"]) == ALLOWED_URLS


def test_as_of_from_nav_line():
    assert extract_as_of("NAV as of 28 Aug 2026: ₹100") == "28 Aug 2026"


def test_elss_does_not_invent_lock_in():
    elss = next(d for d in run_data_loading() if d.id == "hdfc-elss")
    assert "3 year" not in elss.text.lower()
    assert "3-year" not in elss.text.lower()
    assert "lock-in duration" in elss.text.lower() or "does not state a statutory lock-in" in elss.text.lower()


def test_capital_gains_statement_gap_survives():
    docs = run_data_loading()
    joined = " ".join(d.text.lower() for d in docs)
    assert "how to download" not in joined or "does not describe how to download" in joined
    assert "capital-gains statement" in joined


def test_load_raw_documents_dict_shape():
    records = load_raw_documents()
    assert len(records) == 5
    for rec in records:
        assert set(rec) >= {"id", "name", "category", "url", "source_as_of", "text", "corpus_file"}
