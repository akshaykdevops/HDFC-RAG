"""Phase 2 — Chunking tests."""

import json

import pytest

from code.chunking import CHUNKS_FILE, build_chunks, run_chunking, split_text
from code.config import CHUNK_SIZE
from code.loading import ALLOWED_URLS, load_raw_documents


def _docs():
    return load_raw_documents(refresh_from_web=False)


def test_heading_split_keeps_snapshot_apart_from_glossary():
    large = next(d for d in _docs() if d["id"] == "hdfc-large-cap")
    bodies = split_text(large["text"])
    snapshot = next(b for b in bodies if "## Snapshot" in b)
    glossary = next(b for b in bodies if "## Terms defined" in b)
    assert "Expense ratio: 1.02%" in snapshot
    assert "A fee payable" not in snapshot
    assert "A fee payable" in glossary
    assert "1.02%" not in glossary


def test_elss_sip_and_nil_exit_load_survive_chunking():
    elss = next(d for d in _docs() if d["id"] == "hdfc-elss")
    joined = " ".join(split_text(elss["text"]))
    assert "₹500" in joined
    assert "Nil" in joined


def test_chunk_metadata_and_closed_corpus():
    chunks = build_chunks(_docs())
    assert chunks
    urls = {c.source_url for c in chunks}
    assert urls == ALLOWED_URLS
    for c in chunks:
        assert c.fund_name
        assert c.source_url
        assert c.source_as_of
        assert c.category
        assert c.fund_name in c.text
        assert c.source_url in c.text
        assert c.corpus == "groww-hdfc-five"


def test_rejects_document_outside_allow_list():
    docs = _docs()
    docs[0] = {**docs[0], "url": "https://example.com/blog"}
    with pytest.raises(ValueError, match="closed corpus"):
        build_chunks(docs)


def test_windows_stay_near_target_size():
    large = next(d for d in _docs() if d["id"] == "hdfc-large-cap")
    for body in split_text(large["text"]):
        # Heading replay can add a short prefix; allow a small slack over 700.
        assert len(body) <= CHUNK_SIZE + 80


def test_run_chunking_writes_jsonl():
    chunks = run_chunking(_docs())
    assert CHUNKS_FILE.exists()
    lines = [json.loads(ln) for ln in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == len(chunks)
    assert {"fund_name", "source_url", "source_as_of", "category", "text"} <= set(lines[0])
