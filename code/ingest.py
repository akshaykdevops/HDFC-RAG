"""Run phases 1–4: load raw pages, chunk, embed, write Chroma."""

from __future__ import annotations

import sys

from code.chunking import run_chunking
from code.config import FUNDS, VECTOR_DB_DIR
from code.embedding import embed_texts, persist_embeddings
from code.loading import load_raw_documents
from code.vector_store import replace_collection


def ingest(refresh_from_web: bool = False) -> int:
    docs = load_raw_documents(refresh_from_web=refresh_from_web)
    records = [c.to_record() for c in run_chunking(docs)]
    vectors = embed_texts([r["text"] for r in records])
    persist_embeddings(records, vectors)
    return replace_collection(records, vectors)


def main() -> None:
    refresh = "--refresh" in sys.argv
    n = ingest(refresh_from_web=refresh)
    print(f"Indexed {n} chunks from {len(FUNDS)} Groww pages into {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
