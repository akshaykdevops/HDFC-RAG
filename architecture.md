# Architecture — Groww HDFC Fund FAQ (RAG prototype)

**Scope:** [docs/PRD.md](docs/PRD.md) only. Hobby retrieve-then-generate over **five** public Groww HDFC Direct Growth pages. Not a production Groww system.

**Non-goals (from PRD):** any other URL; advice; return computation/comparison; PII storage; account workflows; multi-AMC search; persisted chat identity.

## Repository layout

```
data/                          # pipeline artifacts (not source)
  raw/                         # Phase 1 — five Groww page snapshots
  chunks/                      # Phase 2 — chunks.jsonl
  embeddings/                  # Phase 3 — MiniLM vectors (embeddings.json)
  vector_db/                   # Phase 4 — ChromaDB

code/                          # application source
  loading.py                   # Phase 1
  chunking.py                  # Phase 2
  embedding.py                 # Phase 3
  vector_store.py              # Phase 4
  retrieval.py                 # Phase 5
  policies.py                  # guardrails (before retrieve)
  ingest.py                    # runs phases 1–4
  app.py                       # Streamlit UI
  tests/                       # Phase 6
```

```
[Five Groww URLs]
        │
        ▼
 Phase 1  Data loading     → snapshots (source of truth)
        │
        ▼
 Phase 2  Chunking         → ~700-char overlapping passages
        │
        ▼
 Phase 3  Embedding        → all-MiniLM-L6-v2 (same model at index + query)
        │
        ▼
 Phase 4  Vector store     → ChromaDB (persistent)
        │
        ▼
 Phase 5  Retrieval logic  → guardrails → top-k=4 → Mistral (or extractive fallback)
        │
        ▼
 Phase 6  Retrieval testing → pytest + PRD example/edge cases
        │
        ▼
 Streamlit UI (PRD §9)
```

---

## Phase 1 — Data loading

**Purpose:** Materialise a closed corpus. Nothing enters the index except the five configured fund pages.

**Inputs (closed list)**

| Category | Scheme | URL |
| --- | --- | --- |
| Large-cap | HDFC Large Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| Flexi-cap | HDFC Flexi Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` (legacy slug) |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| Small-cap | HDFC Small Cap Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| Hybrid | HDFC Balanced Advantage Fund Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

**Design**

- Default path: curated snapshots under `data/raw/` (PRD mitigation: Groww HTML is noisy/JS-heavy).
- Optional refresh: re-fetch **those five URLs only**; do not widen the allow-list.
- Strip site chrome; keep fund facts (snapshot, exit load, about, managers, listed holdings).
- Stamp each document with **as-of** (prototype default **28 Aug 2026** / NAV date on the page).
- **FR-1:** index only these five funds.

**Known gaps (must survive loading unchanged)**

- Capital-gains statement download is not on these pages.
- ELSS extract does not state a statutory lock-in; do not inject “3 years.”

**Output:** five cleaned documents + URL + as-of metadata.

---

## Phase 2 — Chunking

**Purpose:** Split each page into passages small enough for MiniLM and large enough to hold one fact (expense ratio, SIP min, exit load).

**Design (PRD §10)**

- Target chunk size **~700 characters** with overlap so a fact is not split from its fund name.
- Prefer heading/section boundaries (snapshot vs glossary vs holdings) so “expense ratio: 1.02%” is not dominated by the glossary definition of expense ratio.
- Attach metadata on every chunk: `fund_name`, `source_url`, `source_as_of`, category.

**Constraint:** no extra documents, blogs, or app screenshots mixed into chunks.

**Output:** ordered list of text chunks + metadata, persisted to `data/chunks/chunks.jsonl`.

---

## Phase 3 — Embedding

**Purpose:** Map chunks and later questions into the same vector space.

**Design (PRD §10)**

- Model: Hugging Face **`sentence-transformers/all-MiniLM-L6-v2`**, local, CPU-ok (success metric: index rebuild on a laptop without GPU).
- Embed **chunks at ingest** and **the user question at query time** with the identical model. Do not mix embedders.
- No third-party embedding API in this prototype.

**Output:** 384-dim vectors aligned 1:1 with chunks, persisted to `data/embeddings/embeddings.json`.

---

## Phase 4 — Vector store

**Purpose:** Persist embeddings for repeated Streamlit sessions without re-crawling.

**Design (PRD §10)**

- Store: **ChromaDB**, persistent directory `data/vector_db/`.
- Collection holds only Phase 2 chunks for the five funds.
- Payload per vector: document text + metadata (`source_url` is the citation candidate).
- Rebuild is a full replace of this collection (closed corpus; no incremental multi-tenant index).

**Output:** queryable collection used exclusively by Phase 5.

---

## Phase 5 — Retrieval logic

**Purpose:** Turn a question into grounded context, then an answer that meets the PRD answer contract — or a refusal/abstention.

**Order of operations (FR-2, FR-3)**

1. **Guardrails (before any retrieve)**  
   - Empty query → prompt for a factual question + citation footer.  
   - PII (PAN, Aadhaar, phone, email, OTP, account/folio numbers) → refuse; do not store identifiers.  
   - Advice / portfolio / “which is better” / “how much SIP” → refuse; one relevant educational link among the five pages.  
   - Compare / project / calculate returns → refuse; point at the Groww page as published factsheet.

2. **Retrieve**  
   - Embed the question with Phase 3’s MiniLM.  
   - Chroma **top-k = 4**.  
   - Drop hits below the relevance threshold (PRD grounding).  
   - Citation URL = retrieved fund’s Groww URL (one URL only).

3. **Abstain** if no hit or the fact is not in context (other AMCs, statement download, invented lock-in). Still emit **one** citation + as-of.

4. **Generate**  
   - **Mistral** (`MISTRAL_API_KEY`, default `mistral-small-latest`) with system rules: facts from context only; ≤3 sentences; no advice; no return math; no invented lock-in/process.  
   - If the key is missing: extractive fallback from retrieved chunks so the RAG path remains demoable (PRD §9).  
   - Post-trim body to **three sentences**; append `Source: <url>` and `Last updated from sources: <date>`.

**Runtime surface:** Streamlit (welcome, three example chips, disclaimer **Facts-only. No investment advice.**, sidebar corpus links, Groww accent `#00B386`). No persisted user identity or chat store (PRD non-goals).

---

## Phase 6 — Retrieval testing

**Purpose:** Prove closed-corpus retrieval and policy without requiring a production eval harness (PRD non-goal).

**Automated (FR-5)**

- Guardrail cases **without calling Mistral**: empty query; PII; “Should I buy HDFC Small Cap?”; “Compare returns of large vs small cap.”
- “How to download capital-gains statement?” **must pass guardrails** then abstain at retrieve/generate (not in corpus).
- “ELSS lock-in?” must not invent a 3-year lock-in.

**Retrieval / answer checks (PRD §13)**

| Query | Must retrieve / answer |
| --- | --- |
| Expense ratio of HDFC Large Cap Fund Direct Growth | **1.02%** + large-cap Groww URL |
| Minimum SIP for HDFC ELSS Tax Saver | **₹500** + ELSS Groww URL |
| Exit load of HDFC Small Cap Fund Direct Growth | **1% if redeemed within 1 year** + small-cap Groww URL |

**Also cover**

- Ambiguous “expense ratio” with no fund name: best-effort retrieve; still one citation; abstain if scores are weak.
- Advice and PII never yield a recommendation or echoed identity document.
- Index rebuild from corpus completes on CPU.

**Manual:** same three example chips in the Streamlit UI.

---

## Decisions locked by the PRD

| Concern | Decision |
| --- | --- |
| Corpus | Five Groww URLs only |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Store | ChromaDB persistent |
| Generator | Mistral API |
| UI | Streamlit |
| Citation | Exactly one Groww fund URL per answer |
| Freshness | `Last updated from sources:` (default 28 Aug 2026) |
| Policy | Guardrails before retrieval; facts-only; no PII store |
