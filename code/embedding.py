"""Phase 3 — Embedding: all-MiniLM-L6-v2 via Chroma's ONNX runtime, local/CPU.

Same model at index and query time. Runs on onnxruntime instead of
torch/sentence-transformers so the full pipeline fits Render's free-tier
(512 MB RAM) instance.
"""

from __future__ import annotations

import json
import sys

from code.chunking import load_chunk_records
from code.config import EMBEDDING_MODEL, EMBEDDINGS_DIR, EMBEDDINGS_FILE

_MODEL = None


def get_model():
    """Return the one shared ONNX MiniLM embedder (kept warm across queries)."""
    global _MODEL
    if _MODEL is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _MODEL = DefaultEmbeddingFunction()
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed an ordered list of passages into 384-dim unit vectors."""
    if not texts:
        raise ValueError("Nothing to embed: text list is empty")
    model = get_model()
    return [[float(x) for x in v] for v in model(texts)]


def embed_question(question: str) -> list[float]:
    """Query-time embedder — identical model and params as chunk embedding."""
    if not question.strip():
        raise ValueError("Nothing to embed: question is empty")
    return embed_texts([question])[0]


def persist_embeddings(records: list[dict], vectors: list[list[float]]) -> None:
    """Persist vectors aligned 1:1 with chunks into data/embeddings/embeddings.json."""
    if not records:
        raise ValueError("Nothing to persist: no chunks to embed")
    if len(records) != len(vectors):
        raise ValueError(f"1:1 alignment broken: {len(records)} chunks vs {len(vectors)} vectors")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": EMBEDDING_MODEL,
        "dimensions": len(vectors[0]),
        "items": [{**rec, "embedding": vec} for rec, vec in zip(records, vectors)],
    }
    EMBEDDINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_embedding(records: list[dict] | None = None) -> tuple[list[dict], list[list[float]]]:
    """Materialise Phase 3 artifacts into data/embeddings/embeddings.json."""
    if records is None:
        records = load_chunk_records()
    vectors = embed_texts([r["text"] for r in records])
    persist_embeddings(records, vectors)
    return records, vectors


def main() -> None:
    records, vectors = run_embedding()
    by_fund: dict[str, int] = {}
    for rec in records:
        by_fund[rec["fund_id"]] = by_fund.get(rec["fund_id"], 0) + 1
    dims = len(vectors[0]) if vectors else 0
    print(f"Phase 3 embedding: {len(records)} chunks → {EMBEDDINGS_FILE}")
    print(f"  model: {EMBEDDING_MODEL}")
    print(f"  dimensions: {dims}")
    for fund_id, n in sorted(by_fund.items()):
        print(f"  - {fund_id}: {n}")


if __name__ == "__main__":
    main()