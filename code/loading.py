"""Phase 1 — Data loading: closed corpus from five Groww URLs only."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from code.config import FUNDS, RAW_DIR, SOURCE_AS_OF

ALLOWED_URLS = frozenset(fund["url"] for fund in FUNDS)
MANIFEST_FILE = RAW_DIR / "manifest.json"

NAV_AS_OF_RE = re.compile(
    r"(?:NAV as of|Page data as of NAV date:)\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+'?[0-9]{2,4})",
    re.I,
)

NOISE_PREFIXES = (
    "Stocks",
    "Invest in Stocks",
    "Intraday",
    "ETF Screener",
    "IPO",
    "MTFs",
    "Stock Screener",
    "Stock Events",
    "Demat Account",
    "Share Market",
    "F&O",
    "Trade in Futures",
    "Indices",
    "Terminal",
    "Option chain",
    "Pledge",
    "Commodities",
    "API trading",
    "Mutual Funds",
    "Invest in Mutual Funds",
    "Mutual Fund Houses",
    "NFO",
    "Mutual Funds by Groww",
    "Start SIP",
    "Mutual Funds screener",
    "Track Funds",
    "SIP calculator",
    "Brokerage calculator",
    "Margin calculator",
    "SWP calculator",
    "Pricing",
    "Credit",
    "Download the App",
    "GROWW",
    "PRODUCTS",
    "Others:",
    "See All",
    "Compare similar funds",
    "Check past data",
    "Return calculator",
    "Monthly SIPOne time",
)


@dataclass(frozen=True)
class LoadedDocument:
    """Phase 1 output: one cleaned fund page plus citation metadata."""

    id: str
    name: str
    category: str
    url: str
    source_as_of: str
    text: str
    corpus_file: str

    def to_record(self) -> dict:
        return asdict(self)


def assert_allowed_url(url: str) -> str:
    """FR-1: nothing outside the five configured Groww pages may be loaded."""
    if url not in ALLOWED_URLS:
        raise ValueError(f"URL is not in the closed corpus: {url}")
    return url


def extract_as_of(text: str, fallback: str = SOURCE_AS_OF) -> str:
    match = NAV_AS_OF_RE.search(text)
    if not match:
        return fallback
    raw = match.group(1).replace("'", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    if re.search(r"\b\d{2}$", raw) and not re.search(r"\b\d{4}$", raw):
        year = raw[-2:]
        raw = raw[:-2].strip() + f" 20{year}"
    return raw


def stamp_snapshot(text: str, name: str, url: str, as_of: str) -> str:
    """Keep fund facts; always stamp Source URL and as-of (architecture output)."""
    body = text.strip()
    body = re.sub(r"^# .+\n+", "", body)
    body = re.sub(r"^Source URL: .+\n", "", body, count=1, flags=re.M)
    body = re.sub(r"^Page data as of NAV date: .+\n+", "", body, count=1, flags=re.M)
    header = (
        f"# {name}\n\n"
        f"Source URL: {url}\n"
        f"Page data as of NAV date: {as_of}\n"
    )
    return f"{header}\n{body.strip()}\n"


def strip_site_chrome(markdown: str, fund_name: str) -> str:
    """Drop Groww nav/footer; keep snapshot, exit load, about, managers, holdings."""
    lines = markdown.splitlines()
    kept: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(NOISE_PREFIXES):
            continue
        if stripped.startswith("Home>") or stripped.startswith("© 2016"):
            break
        if not started and (
            stripped.startswith("NAV:")
            or stripped.startswith("# ")
            or stripped.startswith("### Minimum")
            or stripped.startswith("### About")
            or stripped.startswith("## Understand")
            or stripped.startswith("## Snapshot")
            or stripped.startswith("### Exit")
            or stripped.startswith("Source URL:")
        ):
            started = True
        if started:
            kept.append(line)
    text = "\n".join(kept).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text and not text.lstrip().startswith("#"):
        text = f"# {fund_name}\n\n{text}"
    return text or markdown


def fetch_page(url: str) -> str | None:
    assert_allowed_url(url)
    try:
        import requests
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md
    except ImportError:
        return None

    try:
        resp = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": "GrowwHDFCFactsBot/0.1 (hobby RAG prototype)"},
        )
        resp.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return md(str(soup), heading_style="ATX")


def persist_documents(docs: list[LoadedDocument]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        path = RAW_DIR / doc.corpus_file
        path.write_text(doc.text, encoding="utf-8")
    manifest = {
        "phase": 1,
        "name": "data-loading",
        "count": len(docs),
        "allowed_urls": sorted(ALLOWED_URLS),
        "documents": [
            {
                "id": d.id,
                "name": d.name,
                "category": d.category,
                "url": d.url,
                "source_as_of": d.source_as_of,
                "corpus_file": d.corpus_file,
            }
            for d in docs
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return MANIFEST_FILE


def _document_from_disk(fund: dict) -> LoadedDocument:
    path = RAW_DIR / fund["corpus_file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw corpus file: {path}")
    assert_allowed_url(fund["url"])
    text = path.read_text(encoding="utf-8")
    as_of = extract_as_of(text, SOURCE_AS_OF)
    stamped = stamp_snapshot(text, fund["name"], fund["url"], as_of)
    return LoadedDocument(
        id=fund["id"],
        name=fund["name"],
        category=fund["category"],
        url=fund["url"],
        source_as_of=as_of,
        text=stamped,
        corpus_file=fund["corpus_file"],
    )


def _document_from_web(fund: dict) -> LoadedDocument:
    fetched = fetch_page(fund["url"])
    if not fetched:
        return _document_from_disk(fund)
    cleaned = strip_site_chrome(fetched, fund["name"])
    as_of = extract_as_of(cleaned, date.today().strftime("%d %b %Y"))
    stamped = stamp_snapshot(cleaned, fund["name"], fund["url"], as_of)
    return LoadedDocument(
        id=fund["id"],
        name=fund["name"],
        category=fund["category"],
        url=fund["url"],
        source_as_of=as_of,
        text=stamped,
        corpus_file=fund["corpus_file"],
    )


def run_data_loading(refresh_from_web: bool = False) -> list[LoadedDocument]:
    """Materialise the closed corpus into data/raw/. Does not chunk or embed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for fund in FUNDS:
        if refresh_from_web:
            docs.append(_document_from_web(fund))
        else:
            docs.append(_document_from_disk(fund))
    if len(docs) != 5:
        raise RuntimeError(f"Closed corpus must contain exactly 5 funds, got {len(docs)}")
    persist_documents(docs)
    return docs


def load_raw_documents(refresh_from_web: bool = False) -> list[dict]:
    """Backward-compatible records for later ingest phases."""
    return [doc.to_record() for doc in run_data_loading(refresh_from_web=refresh_from_web)]


def main() -> None:
    refresh = "--refresh" in sys.argv
    docs = run_data_loading(refresh_from_web=refresh)
    mode = "web refresh" if refresh else "local snapshots"
    print(f"Phase 1 data loading ({mode}): {len(docs)} documents → {RAW_DIR}")
    for doc in docs:
        print(f"  - {doc.category}: {doc.name} ({doc.source_as_of})")
        print(f"    {doc.url}")


if __name__ == "__main__":
    main()
