"""Phase 4 — Vector store: persistent ChromaDB collection over Phase 3 vectors.

The collection is the single queryable index for Phase 5. Queries are embedded
at query time with Phase 3's cached all-MiniLM-L6-v2 via MiniLmEmbeddingFunction
so the index and the query never mix embedders.
"""

from __future__ import annotations

import json
import threading

from chromadb.api.types import EmbeddingFunction, Embeddings

from code.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    VECTOR_DB_DIR,
)
from code.embedding import embed_texts
from code.loading import ALLOWED_URLS

COLLECTION_METADATA = {"hnsw:space": "cosine"}


class MiniLmEmbeddingFunction(EmbeddingFunction):
    """Query-time adapter over Phase 3's cached model — no second embedder."""

    def __init__(self) -> None:
        self._model = EMBEDDING_MODEL

    def __call__(self, input) -> Embeddings:
        return embed_texts(list(input))

    @staticmethod
    def name() -> str:
        return EMBEDDING_MODEL

    def get_config(self) -> dict:
        return {"model": self._model}

    @staticmethod
    def build_from_config(config: dict) -> "MiniLmEmbeddingFunction":
        return MiniLmEmbeddingFunction()


_client_singleton = None
_client_lock = threading.Lock()


def _client():
    """One native Chroma client per process.

    Never create a fresh client per call: short-lived clients in a long-running
    process (the Streamlit app) race Chroma's async sqlite commits and can end up
    querying a collection UUID that has since been replaced, surfacing as
    NotFoundError("Collection [...] does not exist") on a healthy index.
    """
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            import chromadb

            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            _client_singleton = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        return _client_singleton


def get_collection():
    """Return the single queryable collection (Phase 5 consumes this)."""
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=MiniLmEmbeddingFunction(),
        metadata=COLLECTION_METADATA,
    )


def replace_collection(records: list[dict], vectors: list[list[float]]) -> int:
    """Full replace of the collection with the Phase 2/3 chunks (closed corpus)."""
    if not records:
        raise ValueError("Nothing to index: no chunks to store")
    if len(records) != len(vectors):
        raise ValueError(f"1:1 alignment broken: {len(records)} chunks vs {len(vectors)} vectors")
    urls = {r["source_url"] for r in records}
    extra = urls - ALLOWED_URLS
    if extra:
        raise ValueError(f"Refusing to index outside the closed corpus: {', '.join(sorted(extra))}")

    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=MiniLmEmbeddingFunction(),
        metadata=COLLECTION_METADATA,
    )
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "fund_id": r["fund_id"],
                "fund_name": r["fund_name"],
                "category": r["category"],
                "source_url": r["source_url"],
                "source_as_of": r["source_as_of"],
                "chunk_index": r["chunk_index"],
                "corpus": r["corpus"],
            }
            for r in records
        ],
        embeddings=vectors,
    )
    return len(records)


def load_embedded_records(path=EMBEDDINGS_FILE) -> tuple[list[dict], list[list[float]], str]:
    """Read Phase 3 output: records + vectors aligned 1:1, and the model name."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 embeddings; run python -m code.embedding first: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [{k: v for k, v in item.items() if k != "embedding"} for item in payload["items"]]
    vectors = [item["embedding"] for item in payload["items"]]
    return records, vectors, payload["model"]


def main() -> None:
    records, vectors, model = load_embedded_records()
    indexed = replace_collection(records, vectors)
    count = get_collection().count()
    print(f"Phase 4 vector store: {indexed} chunk vectors → {VECTOR_DB_DIR}")
    print(f"  collection: {COLLECTION_NAME} (count={count})")
    print(f"  embedding model: {model}")
    print(f"  distance: {COLLECTION_METADATA.get('hnsw:space')}")


if __name__ == "__main__":
    main()