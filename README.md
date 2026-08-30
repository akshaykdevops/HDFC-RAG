# Groww HDFC Fund FAQ (RAG prototype)

Facts-only chatbot over **five** public [Groww](https://groww.in/) pages for HDFC Direct Growth funds. Not investment advice.

Product spec: [docs/PRD.md](docs/PRD.md) · Architecture: [architecture.md](architecture.md) · Brief: [docs/groww ai chatbot.txt](docs/groww%20ai%20chatbot.txt)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add MISTRAL_API_KEY from https://console.mistral.ai/
```

Index the corpus (first run also happens when you open the app):

```bash
python -m code.loading          # Phase 1 only → data/raw/
python -m code.chunking         # Phase 2 only → data/chunks/chunks.jsonl
python -m code.embedding        # Phase 3 only → data/embeddings/embeddings.json
python -m code.vector_store     # Phase 4 only → ChromaDB at data/vector_db/
python -m code.ingest           # Phases 1–4 → chunks, embeddings, Chroma
```

Optional: re-fetch the same five Groww URLs only:

```bash
python -m code.ingest --refresh
```

## Run

```bash
streamlit run code/app.py
```

Without a Mistral key, answers fall back to a short extract from the top chunk so retrieval still works.

## Test retrieval (backend only)

```bash
python -m code.retrieval "What is the exit load of HDFC Small Cap Fund Direct Growth?"
python -m code.retrieval "..." --debug   # also show retrieved chunks + cosine scores
python -m code.retrieval                 # interactive REPL (q to quit)
```

## Tests

```bash
pytest
```

## Stack

| Piece | Choice |
| --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| LLM | Mistral API |
| UI | Streamlit (Groww green `#00B386`) |
