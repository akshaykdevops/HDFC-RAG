"""Phase 3 — Embedding tests: 384-dim, one model for chunks and queries, 1:1 artifact."""

import json

import pytest

from code.chunking import CHUNKS_FILE
from code.config import EMBEDDING_MODEL
from code.embedding import (
    EMBEDDINGS_FILE,
    embed_question,
    embed_texts,
    persist_embeddings,
    run_embedding,
)


def _chunk_records() -> list[dict]:
    assert CHUNKS_FILE.exists(), "no Phase 2 chunks; run python -m code.chunking"
    return [json.loads(ln) for ln in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_embed_texts_returns_384_dim_unit_vectors():
    vectors = embed_texts(["Expense ratio of HDFC Large Cap Fund is 1.02%."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_query_embedding_shares_same_vector_space_as_chunks():
    chunk = "Minimum SIP for HDFC ELSS Tax Saver Fund is ₹500."
    question = "What is the minimum SIP for HDFC ELSS Tax Saver?"
    chunk_vec, question_vec = embed_texts([chunk])[0], embed_question(question)
    assert len(question_vec) == 384
    assert _dot(chunk_vec, question_vec) > 0.3


def test_run_embedding_materializes_aligned_artifact():
    run_embedding()
    assert EMBEDDINGS_FILE.exists()
    payload = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    assert payload["model"] == EMBEDDING_MODEL
    records = _chunk_records()
    assert payload["dimensions"] == 384
    assert len(payload["items"]) == len(records)
    assert [item["id"] for item in payload["items"]] == [rec["id"] for rec in records]
    for rec, item in zip(records, payload["items"]):
        assert len(item["embedding"]) == 384
        assert item["text"] == rec["text"]
        assert item["source_url"] == rec["source_url"]


def test_persist_requires_chunks_and_vectors_1_to_1():
    with pytest.raises(ValueError, match="no chunks"):
        persist_embeddings([], [])
    with pytest.raises(ValueError, match="empty"):
        embed_texts([])
    with pytest.raises(ValueError, match="empty"):
        embed_question("   ")
    with pytest.raises(ValueError, match="alignment"):
        persist_embeddings([{"id": "x"}], [[0.0] * 3, [0.1] * 3])


def test_run_embedding_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        run_embedding([])