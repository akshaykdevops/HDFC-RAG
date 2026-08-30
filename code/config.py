from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.json"

COLLECTION_NAME = "groww_hdfc_funds"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Allowed RAG sources only — do not ingest anything else.
FUNDS = [
    {
        "id": "hdfc-large-cap",
        "name": "HDFC Large Cap Fund Direct Growth",
        "category": "Large-cap",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "corpus_file": "hdfc-large-cap.md",
    },
    {
        "id": "hdfc-flexi-cap",
        "name": "HDFC Flexi Cap Fund Direct Growth",
        "category": "Flexi-cap",
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "corpus_file": "hdfc-flexi-cap.md",
        "slug_note": "Groww still uses the legacy equity-fund slug.",
    },
    {
        "id": "hdfc-elss",
        "name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "category": "ELSS",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "corpus_file": "hdfc-elss.md",
    },
    {
        "id": "hdfc-small-cap",
        "name": "HDFC Small Cap Fund Direct Growth",
        "category": "Small-cap",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "corpus_file": "hdfc-small-cap.md",
    },
    {
        "id": "hdfc-baf",
        "name": "HDFC Balanced Advantage Fund Direct Growth",
        "category": "Hybrid",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        "corpus_file": "hdfc-baf.md",
    },
]

DEFAULT_CITATION = FUNDS[0]["url"]
SOURCE_AS_OF = "28 Aug 2026"

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the minimum SIP for HDFC ELSS Tax Saver?",
    "What is the exit load of HDFC Small Cap Fund Direct Growth?",
]

WELCOME = (
    "Ask factual questions about five HDFC Direct Growth funds listed on Groww. "
    "I only use those five public pages."
)
DISCLAIMER = "Facts-only. No investment advice."

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4
MIN_RELEVANCE = 0.34

GROWW_GREEN = "#00B386"
GROWW_DARK = "#1B1B1B"
GROWW_BG = "#F6F6F6"
GROWW_MUTED = "#6A6A6A"
