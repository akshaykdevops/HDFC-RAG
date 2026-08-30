# Product Requirements Document

**Product:** Groww HDFC Fund FAQ (facts-only RAG prototype)  
**Owner:** Product (hobby / build-hour)  
**Status:** Prototype  
**Date:** 30 Aug 2026  
**Source brief:** `docs/groww ai chatbot.txt`  
**Surfaces:** Streamlit chat (`app.py`) over five public Groww fund pages  

---

## 1. Problem

Investors on Groww ask the same factual questions about a fund (expense ratio, SIP minimum, exit load, risk, benchmark) while also asking for advice (“should I buy?”) or sharing identity details. A small, citable FAQ assistant can answer **only what is published** on a tightly scoped set of pages, and must refuse everything else.

This is a **RAG hobby prototype**, not a production Groww feature. Success is a working retrieve-then-generate loop with guardrails, not coverage of the whole Groww catalogue.

## 2. Goals

| Priority | Goal |
| --- | --- |
| P0 | Answer factual questions from **only** the five HDFC Direct Growth pages listed below, with **one citation URL** and **Last updated from sources:** |
| P0 | Refuse opinionated / portfolio advice with a polite facts-only message and a relevant educational (fund page) link |
| P0 | Reject PII in the query; do not store PAN, Aadhaar, account numbers, OTPs, emails, or phones |
| P0 | Do not compute or compare returns; point to the Groww page as the published factsheet |
| P1 | Tiny UI: welcome line, three example questions, disclaimer **Facts-only. No investment advice.** |
| P1 | Groww-like visual language (accent `#00B386`) on Streamlit |
| P1 | Documented edge-case tests for the above |

## 3. Non-goals

- Any URL, blog, screenshot, or app-backend source beyond the five pages.
- Buy / sell / SIP amount / “best fund” recommendations.
- Computing CAGR, comparing two funds’ returns, or projecting SIP outcomes.
- Account-logged workflows (capital-gains statement download, folio lookup, OTP).
- Multi-AMC search, holdings analysis as advice, or a production LLM eval harness.
- Storing chat history or user identity.

## 4. Users

| User | Need |
| --- | --- |
| Curious investor (anonymous) | Fast facts on five HDFC schemes before opening the full Groww page |
| Product / eng (internal) | Test chunking, MiniLM embeddings, Chroma retrieval, and Mistral generation under strict policy |

No authenticated Groww user. No KYC.

## 5. Corpus (closed)

AMC: **HDFC**. Website: [groww.in](https://groww.in/). **Ingest these five URLs only.**

| Category | Scheme (Direct Growth) | Groww URL |
| --- | --- | --- |
| Large-cap | HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi-cap | HDFC Flexi Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth (legacy “equity fund” slug) |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small-cap | HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Hybrid | HDFC Balanced Advantage Fund Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Snapshots used for the prototype live in `data/raw/`. Optional `--refresh` re-fetches the same five URLs only.

**Known corpus gaps (by design):** Groww help articles (e.g. how to download a capital-gains statement) are **out of scope**. The ELSS page extract does **not** state a statutory lock-in length; the bot must not invent “3 years.”

## 6. In-scope question types (examples)

- Expense ratio of a named fund  
- Minimum SIP / lumpsum  
- Exit load (including BAF’s 15% free / 1% within 1 year wording)  
- Risk rating / benchmark  
- NAV and as-of date as **published** (not a calculated return)  
- Fund manager names, objective text, stamp duty / tax **as printed on the page**  
- Top holdings **as listed**, if present in retrieved chunks  

## 7. Out-of-scope question types (must refuse or abstain)

| Type | Behaviour |
| --- | --- |
| Advice (“Should I buy/sell?”, “which is better?”, “how much SIP?”) | Refuse; cite the most relevant of the five pages |
| Returns compare / project / calculate | Refuse; link to the Groww page as the official published figures |
| PII in the prompt | Refuse; do not echo identifiers into logs beyond the local session |
| Process not on the five pages (statements, app navigation) | Abstain: not in allowed sources + still one citation |
| Other AMCs, other HDFC schemes, news | Abstain |

## 8. Answer contract

Every assistant message must satisfy:

1. **Body ≤ 3 sentences.**
2. **Exactly one** citation: `Source: <https://groww.in/mutual-funds/...>` matching the retrieved fund (or a relevant fund page on refusal).
3. **Last updated from sources: &lt;date&gt;** (NAV/as-of from the snapshot; prototype default **28 Aug 2026**).
4. English, factual tone. No “I recommend.”

## 9. User experience

1. Welcome: *Ask factual questions about five HDFC Direct Growth funds listed on Groww. I only use those five public pages.*
2. Disclaimer always visible: **Facts-only. No investment advice.**
3. Three example chips:
   - What is the expense ratio of HDFC Large Cap Fund Direct Growth?
   - What is the minimum SIP for HDFC ELSS Tax Saver?
   - What is the exit load of HDFC Small Cap Fund Direct Growth?
4. Chat input. Sidebar lists the five corpus links.
5. Optional caption when Mistral key is missing (extractive fallback) so the RAG path is still demoable.

## 10. RAG architecture (prototype)

```
Five Groww URLs → clean snapshot → chunk (~700 chars, overlap)
        → sentence-transformers/all-MiniLM-L6-v2 → ChromaDB
Question → guardrails → same embedder → top-k chunks
        → Mistral chat completion → 3-sentence + citation + as-of
```

| Stage | Choice |
| --- | --- |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face, local, lightweight) |
| Vector store | ChromaDB persistent (`data/vector_db/`) |
| Generator | Mistral API (`MISTRAL_API_KEY`, default model `mistral-small-latest`) |
| UI | Streamlit |

**Grounding:** If retrieval score is below threshold or the model would have to guess, say the fact is not in the allowed sources.

## 11. Functional requirements

| ID | Requirement |
| --- | --- |
| FR-1 | Index only the five configured funds |
| FR-2 | Guardrails run **before** retrieval for PII, advice, and return comparison |
| FR-3 | Factual path retrieves k=4 chunks and prompts Mistral with system rules in §8 |
| FR-4 | UI matches §9 |
| FR-5 | `pytest` covers guardrail edge cases without calling Mistral |

## 12. Edge cases (test plan)

| Case | Expected |
| --- | --- |
| Empty query | Prompt to ask a factual question + citation footer |
| PAN / Aadhaar / phone / email / OTP / folio number | PII refusal |
| “Should I buy HDFC Small Cap?” | Advice refusal + small-cap Groww URL |
| “Compare returns of large vs small cap” | Performance refusal + fund page link |
| “How to download capital-gains statement?” | Passes guardrails; RAG abstains (not in corpus) |
| “ELSS lock-in?” | Only what the page says; **no invented 3-year lock-in** |
| Expense ratio / SIP min / exit load (examples) | Grounded answer + matching URL |
| Ambiguous “expense ratio” with no fund name | Best-effort retrieval; still one citation; may abstain if scores are weak |

## 13. Success metrics (prototype)

- Guardrail tests pass (`pytest`).
- Manual: three example questions return the published figures (1.02% large-cap ER; ₹500 ELSS SIP; 1% small-cap exit load within 1 year) with the correct Groww URL.
- Manual: advice and PII prompts never produce a recommendation or echoed identity document.
- Index rebuild from corpus completes on a laptop without a GPU.

## 14. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Groww HTML is noisy / JS-heavy; live scrape misses facts | Curated snapshots in `data/raw/` as source of truth for the demo |
| MiniLM retrieves the wrong fund | Fund name in metadata; citation from top hit; short answers reduce mixing |
| LLM ignores “3 sentences” or invents lock-in | System prompt + post-trim to 3 sentences; corpus explicitly says lock-in is not stated |
| Stale NAV | Footer as-of date; no live trading claims |
| Compliance (investment advice) | Hard refuse + disclaimer; not a SEBI-registered advisor product |

## 15. Open questions (out of prototype)

- Production: SEBI advertising / AI-disclaimer legal review  
- Refresh cadence vs Groww ToS for scraping  
- Whether published 3Y returns on the page may be **quoted** vs always “see page” (this prototype **does not compare**; NAV quote is allowed)

## 16. Launch checklist (build hour)

- [x] Closed corpus of five URLs  
- [x] Chunk → MiniLM → Chroma  
- [x] Mistral generation with fallback  
- [x] Streamlit UI (Groww green, examples, disclaimer)  
- [x] Guardrails + pytest  
- [ ] Operator sets `MISTRAL_API_KEY` and runs `streamlit run app.py`  
