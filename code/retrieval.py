"""Phase 5 — Retrieval: guardrails → top-k Chroma → Mistral or extractive fallback.

Order of operations (FR-2, FR-3):
  1. Guardrails (policies.py) before any retrieve.
  2. Embed the question with Phase 3's MiniLM, Chroma top-k, drop weak hits.
  3. Abstain when nothing relevant is in the five-page corpus (still 1 citation + as-of).
  4. Generate with Mistral, or an extractive fallback when no API key, then trim
     to ≤3 sentences and append Source + Last updated.

Usage:
  python -m code.retrieval "What is the exit load of HDFC Small Cap Fund?"
  python -m code.retrieval                    # interactive REPL
  python -m code.retrieval "..." --debug      # also print retrieved chunks + scores
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from code.config import (
    DEFAULT_CITATION,
    MIN_RELEVANCE,
    ROOT,
    SOURCE_AS_OF,
    TOP_K,
)
from code.policies import apply_guardrails
from code.vector_store import get_collection

load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are a facts-only FAQ assistant for five HDFC Direct Growth funds listed on Groww.

Rules:
- Answer only from the retrieved context. If the context does not contain the fact, say you do not have it in the allowed sources.
- Maximum 3 sentences in the answer body.
- Do not give investment advice or say whether someone should buy, sell, or hold.
- Do not compute, project, or compare returns. If asked for published NAV, you may quote the NAV and as-of date from context.
- Do not invent lock-in periods, tax rules, or processes (including how to download statements) unless they appear in context.
- Do not use personal data.
- End with exactly one citation line: Source: <url>
- Then end with: Last updated from sources: <date from context>
"""

_STOP = {
    "what", "the", "for", "and", "of", "how", "does", "did", "has", "have",
    "this", "that", "with", "from", "about", "please", "tell", "give", "get",
    "mean", "means", "called", "know", "could", "would", "should", "want",
    "need", "like", "can", "was", "were", "are", "been", "fund", "hdfc",
    "direct", "growth", "plan", "page", "groww", "me", "you", "your", "my",
    "not",
}

# Domain terms of exactly 3 chars that carry meaning despite the length filter.
_SHORT_TERMS = {"nav", "sip", "elss", "baf"}

_FACT_RE = re.compile(
    r"expense ratio\s*[:—]\s*\d|minimum sip\s*[:—]|exit load of \d|current exit load"
    r"|nav as of|fund benchmark|risk rating|\block-in\b"
)

_FUND_ALIASES = {
    "balanced advantage": "hdfc-baf",
    "balanced": "hdfc-baf",
    "baf": "hdfc-baf",
    "hybrid": "hdfc-baf",
    "tax saver": "hdfc-elss",
    "elss": "hdfc-elss",
    "equity fund": "hdfc-flexi-cap",
    "flexi cap": "hdfc-flexi-cap",
    "flexi": "hdfc-flexi-cap",
    "small cap": "hdfc-small-cap",
    "small": "hdfc-small-cap",
    "large cap": "hdfc-large-cap",
    "large": "hdfc-large-cap",
}


@dataclass
class RagAnswer:
    text: str
    citation: str
    blocked: bool
    kind: str | None
    used_llm: bool


def retrieve(question: str) -> list[dict]:
    """Chroma top-k=4, cosine score = 1 - distance, drop hits below threshold."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    hits = []
    for doc, meta, dist in zip(docs, metas, distances):
        score = 1 - float(dist) if dist is not None else 0
        if score < MIN_RELEVANCE:
            continue
        hits.append({"text": doc, "meta": meta or {}, "score": score})
    return hits


def _format_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["meta"]
        parts.append(
            f"[{i}] {meta.get('fund_name')} | {meta.get('source_url')} | as of {meta.get('source_as_of')}\n{hit['text']}"
        )
    return "\n\n".join(parts)


def _query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9%₹]+", question):
        token = token.lower()
        if token in _STOP:
            continue
        if len(token) > 3 or token in _SHORT_TERMS:
            terms.add(token)
    return terms


def _named_fund(question: str) -> str | None:
    q = question.lower()
    for alias, fund_id in _FUND_ALIASES.items():
        if alias in q:
            return fund_id
    return None


def _out_of_corpus_scheme(question: str) -> bool:
    """True when the question points at a scheme/AMC that is not one of the five."""
    q = question.lower()
    if any(alias in q for alias in _FUND_ALIASES):
        return False
    return bool(re.search(r"\b(?:amc|scheme|fund)\b", q))


def _wants_holdings(terms: set[str]) -> bool:
    return bool({"hold", "holdings", "portfolio", "stock", "stocks", "invested", "top"} & terms)


def _definitional(question: str) -> bool:
    return bool(re.search(r"\b(mean|means|meaning|defined|definition|what is|what's)\b", question.lower()))


def _best_hit(question: str, hits: list[dict]) -> dict:
    """Pick the chunk that actually carries the fact for the question."""
    terms = _query_terms(question)
    named = _named_fund(question)
    wants_def = _definitional(question)
    wants_hold = _wants_holdings(terms)

    def score(hit: dict) -> float:
        text = hit["text"].lower()
        meta = hit["meta"]
        section = (meta.get("section") or "").lower()
        s = hit.get("score", 0.0)
        overlap = sum(1 for t in terms if t in text)
        s += overlap * 0.08
        if named and meta.get("fund_id") == named:
            s += 0.12
        elif named:
            s -= 0.15
        if _FACT_RE.search(text):
            s += 0.15
        if "terms defined" in section:
            s += 0.25 if wants_def else -0.2
        if "a fee payable" in text and not wants_def:
            s -= 0.15
        if "holdings" in section and not wants_hold:
            s -= 0.15
        if "compare similar" in section:
            s -= 0.1
        return s

    return max(hits, key=score)


def _relevant_sentence(question: str, text: str) -> str:
    """Return the most topic-relevant lines of a chunk for extractive answers."""
    terms = _query_terms(question)
    wants_def = _definitional(question)
    wants_hold = _wants_holdings(terms)
    skip_prefixes = ("source:", "page data", "amc:", "#")
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(skip_prefixes):
            continue
        if "). source: https://groww.in" in low:
            continue
        lines.append(s)

    def line_score(ln: str) -> int:
        low = ln.lower()
        overlap = sum(1 for t in terms if t in low)
        score = overlap * 4
        if _FACT_RE.search(low):
            score += 2
        if wants_def and ": a " in low:
            score += 3
        if "a fee payable" in low and not wants_def:
            score -= 8
        if "does not describe how to download" in low and {"download", "statement", "gains"} & terms:
            score += 2
        if "holdings count" in low and not wants_hold:
            score -= 6
        if "similar fund" in low:
            score -= 6
        if wants_hold and re.search(r"\bLtd\b|%;\s", low):
            score += 6
        if re.search(r"\d", ln):
            score += 1
        return score

    ranked = sorted(lines, key=line_score, reverse=True)
    pick = " ".join(ranked[:3]) if ranked else text
    pick = re.sub(r"\s+", " ", pick).strip()
    return pick[:380]


def _fallback_extractive(question: str, hits: list[dict]) -> str:
    """Return a grounded extractive answer when Mistral is unavailable."""
    chosen = _best_hit(question, hits)
    citation = chosen["meta"].get("source_url", DEFAULT_CITATION)
    as_of = chosen["meta"].get("source_as_of", SOURCE_AS_OF)
    fund = chosen["meta"].get("fund_name")
    snippet = _relevant_sentence(question, chosen["text"])

    if "does not describe how to download" in snippet.lower():
        body = (
            f"Downloading a capital-gains statement is not described in the five allowed Groww pages. "
            f"The {fund} page states that the page does not describe the process."
        )
    else:
        body = f"According to the Groww page for {fund}, {snippet}"
    return f"{body}\n\nSource: {citation}\nLast updated from sources: {as_of}"


def _call_mistral(question: str, context: str) -> str:
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing_key")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
        temperature=0.1,
        max_tokens=280,
    )
    return resp.choices[0].message.content.strip()


def _ensure_citation(text: str, citation: str, as_of: str) -> str:
    """Post-trim to ≤3 sentences and append the single Source + as-of lines."""
    body = text.strip()
    body = re.sub(r"\n*Source:.*", "", body, flags=re.I)
    body = re.sub(r"\n*Last updated from sources:.*", "", body, flags=re.I).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    body = " ".join(sentences[:3])
    return f"{body}\n\nSource: {citation}\nLast updated from sources: {as_of}"


def answer(question: str) -> RagAnswer:
    guard = apply_guardrails(question)
    if guard.blocked:
        return RagAnswer(guard.message or "", guard.citation or DEFAULT_CITATION, True, guard.kind, False)

    hits = [] if _out_of_corpus_scheme(question) else retrieve(question)
    if not hits:
        text = (
            "That is not in the five allowed Groww fund pages. "
            "Name one of the five HDFC funds and ask about expense ratio, SIP minimum, exit load, "
            "risk rating, benchmark, NAV, or holdings."
        )
        formatted = _ensure_citation(text, DEFAULT_CITATION, SOURCE_AS_OF)
        return RagAnswer(formatted, DEFAULT_CITATION, False, "no_context", False)

    chosen = _best_hit(question, hits)
    citation = chosen["meta"].get("source_url", DEFAULT_CITATION)
    as_of = chosen["meta"].get("source_as_of", SOURCE_AS_OF)
    context = _format_context(hits)

    try:
        raw = _call_mistral(question, context)
        used_llm = True
    except Exception:
        raw = _fallback_extractive(question, hits)
        used_llm = False

    formatted = _ensure_citation(raw, citation, as_of)
    return RagAnswer(formatted, citation, False, None, used_llm)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Groww HDFC five-fund FAQ retrieval")
    parser.add_argument("question", nargs="*", help="one question (omit for an interactive REPL)")
    parser.add_argument("--debug", action="store_true", help="also print retrieved chunks + scores")
    args = parser.parse_args(argv)

    def run(question: str) -> None:
        print(f"\nQ: {question}")
        if args.debug and question.strip():
            for hit in retrieve(question):
                meta = hit["meta"]
                print(
                    f"  [{hit['score']:.3f}] {meta.get('fund_name')} | {meta.get('section')} "
                    f"| {meta.get('source_url')}"
                )
                print(f"      {hit['text'][:120]!r}")
        result = answer(question)
        prefix = "[BLOCKED] " if result.blocked else ""
        print(f"A: {prefix}{result.text}")
        if result.blocked:
            print(f"   guardrail kind: {result.kind}")
        elif not result.used_llm:
            print("   (extractive fallback — set MISTRAL_API_KEY for LLM answers)")

    if args.question:
        run(" ".join(args.question))
        return
    print("Groww HDFC FAQ retrieval. Type questions, q to quit, Ctrl-D to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in {"q", "quit", "exit"}:
            break
        run(line)


if __name__ == "__main__":
    main()