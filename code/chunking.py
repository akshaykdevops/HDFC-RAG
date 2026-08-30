"""Phase 2 — Chunking: heading-aware ~700-char passages with overlap."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from code.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_DIR, CHUNKS_FILE
from code.loading import ALLOWED_URLS, load_raw_documents


@dataclass(frozen=True)
class ChunkRecord:
    """Phase 2 output: one passage plus citation metadata."""

    id: str
    text: str
    fund_id: str
    fund_name: str
    category: str
    source_url: str
    source_as_of: str
    section: str
    chunk_index: int
    corpus: str = "groww-hdfc-five"

    def to_record(self) -> dict:
        return asdict(self)


def _section_name(block: str) -> str:
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped[3:].strip().lower()
    return "preamble"


def split_text(text: str) -> list[str]:
    """Split on ## headings first; window only inside a section (~700 chars + overlap)."""
    chunks: list[str] = []
    for section in _by_heading(text):
        heading = next((ln for ln in section.splitlines() if ln.startswith("## ")), "")
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
            continue
        for piece in _window(section):
            if heading and not piece.startswith(heading):
                piece = f"{heading}\n{piece}"
            chunks.append(piece)
    return [c for c in chunks if c.strip()]


def _with_fund_identity(fund_name: str, category: str, url: str, body: str) -> str:
    """Keep the fund name with the fact so a split passage still cites the right scheme."""
    return f"{fund_name} ({category}). Source: {url}\n{body.strip()}"


def build_chunk_records(docs: list[dict]) -> list[dict]:
    return [c.to_record() for c in build_chunks(docs)]


def build_chunks(docs: list[dict]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    seen_urls: set[str] = set()
    for doc in docs:
        url = doc["url"]
        if url not in ALLOWED_URLS:
            raise ValueError(f"Chunking refused a document outside the closed corpus: {url}")
        seen_urls.add(url)
        for i, body in enumerate(split_text(doc["text"])):
            records.append(
                ChunkRecord(
                    id=f"{doc['id']}:{i}",
                    text=_with_fund_identity(doc["name"], doc["category"], url, body),
                    fund_id=doc["id"],
                    fund_name=doc["name"],
                    category=doc["category"],
                    source_url=url,
                    source_as_of=doc["source_as_of"],
                    section=_section_name(body),
                    chunk_index=i,
                )
            )
    extra = seen_urls - ALLOWED_URLS
    if extra:
        raise ValueError(f"Unexpected URLs in chunk input: {extra}")
    return records


def persist_chunks(records: list[dict]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_chunk_records(path: Path = CHUNKS_FILE) -> list[dict]:
    """Read Phase 2 output back as ordered records (embedding phase input)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 chunks; run python -m code.chunking first: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_chunking(docs: list[dict] | None = None) -> list[ChunkRecord]:
    """Materialise Phase 2 artifacts into data/chunks/chunks.jsonl. Does not embed."""
    if docs is None:
        docs = load_raw_documents(refresh_from_web=False)
    chunks = build_chunks(docs)
    persist_chunks([c.to_record() for c in chunks])
    return chunks


def _by_heading(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and buf:
            parts.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


def _window(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            space = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return out


def main() -> None:
    chunks = run_chunking()
    sections = sorted({c.section for c in chunks})
    print(f"Phase 2 chunking: {len(chunks)} chunks → {CHUNKS_FILE}")
    print(f"  sections: {', '.join(sections)}")
    by_fund: dict[str, int] = {}
    for c in chunks:
        by_fund[c.fund_id] = by_fund.get(c.fund_id, 0) + 1
    for fund_id, n in by_fund.items():
        print(f"  - {fund_id}: {n}")


if __name__ == "__main__":
    main()
