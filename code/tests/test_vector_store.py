"""Phase 4 — Vector store tests: persistent Chroma collection over Phase 3 vectors."""

import pytest

from code.chunking import load_chunk_records
from code.config import COLLECTION_NAME, VECTOR_DB_DIR
from code.embedding import embed_texts, run_embedding
from code.loading import ALLOWED_URLS
from code.vector_store import get_collection, load_embedded_records, replace_collection


@pytest.fixture(scope="module")
def corpus() -> tuple[list[dict], list[list[float]]]:
    records = load_chunk_records()
    vectors = embed_texts([r["text"] for r in records])
    return records, vectors


def _iter_meta_and_docs():
    got = get_collection().get(include=["documents", "metadatas"])
    return (
        {ident: meta for ident, meta in zip(got["ids"], got["metadatas"])},
        {ident: doc for ident, doc in zip(got["ids"], got["documents"])},
        got["ids"],
    )


def test_replace_collection_persists_every_chunk(corpus):
    records, vectors = corpus
    n = replace_collection(records, vectors)
    assert n == len(records)
    col = get_collection()
    assert col.count() == len(records)
    assert col.name == COLLECTION_NAME


def test_payload_carries_document_text_and_citation_metadata(corpus):
    records, vectors = corpus
    replace_collection(records, vectors)
    metas, docs, ids = _iter_meta_and_docs()
    assert len(ids) == len(records)
    for rec in records[:12]:
        assert docs[rec["id"]] == rec["text"]
        meta = metas[rec["id"]]
        assert meta["source_url"] == rec["source_url"]
        assert meta["fund_name"] == rec["fund_name"]
        assert meta["category"] == rec["category"]
        assert meta["source_as_of"] == rec["source_as_of"]
        assert meta["corpus"] == "groww-hdfc-five"


def test_collection_only_holds_the_five_allow_listed_funds(corpus):
    records, vectors = corpus
    replace_collection(records, vectors)
    metas, _, ids = _iter_meta_and_docs()
    assert {metas[i]["source_url"] for i in ids} == ALLOWED_URLS


def test_query_embeds_with_phase3_model_and_returns_grounded_hits(corpus):
    records, vectors = corpus
    replace_collection(records, vectors)
    result = get_collection().query(
        query_texts=["What is the minimum SIP for HDFC ELSS Tax Saver?"],
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    assert docs and metas and distances
    assert all(m["source_url"] in ALLOWED_URLS for m in metas)
    assert all(d is not None and d >= 0 for d in distances)
    assert "elss-tax-saver" in metas[0]["source_url"]


def test_phase3_artifact_feeds_the_rebuild(corpus):
    records, _ = corpus
    run_embedding()
    loaded_records, loaded_vectors, model = load_embedded_records()
    assert [r["id"] for r in loaded_records] == [r["id"] for r in records]
    assert model == "sentence-transformers/all-MiniLM-L6-v2"
    n = replace_collection(loaded_records, loaded_vectors)
    assert n == len(records)
    assert get_collection().count() == len(records)


def test_rejects_document_outside_closed_corpus(corpus):
    records, vectors = corpus
    bad = [{**records[0], "source_url": "https://example.com/blog"}]
    with pytest.raises(ValueError, match="closed corpus"):
        replace_collection(bad, [vectors[0]])


def test_rejects_1_to_1_alignment_break(corpus):
    records, vectors = corpus
    with pytest.raises(ValueError, match="1:1 alignment"):
        replace_collection(records[:1], vectors)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no chunks"):
        replace_collection([], [])