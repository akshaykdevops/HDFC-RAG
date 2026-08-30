"""Guardrails: PII, advice, and performance-comparison refusals."""

from __future__ import annotations

import re
from dataclasses import dataclass

from code.config import DEFAULT_CITATION, FUNDS, SOURCE_AS_OF

PII_PATTERNS = [
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "PAN"),
    (r"\b[2-9][0-9]{11}\b", "Aadhaar"),
    (r"\b\d{4}\s\d{4}\s\d{4}\b", "Aadhaar"),
    (r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", "phone"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email"),
    (r"\b(?:otp|one[-\s]?time\s+password)\b", "OTP"),
    (r"\b(?:account|folio|demat)\s*(?:no\.?|number|#)\s*[:#]?\s*\d{4,}\b", "account"),
]

ADVICE_PATTERNS = [
    r"\bshould i (buy|sell|invest|redeem|switch|exit|hold)\b",
    r"\b(is it|would it be) (a )?good (time|idea) to\b",
    r"\b(recommend|suggestion|advise|advice)\b",
    r"\bwhich (fund|one) (is |should i )?(better|best|safer)\b",
    r"\bbest fund\b",
    r"\b(buy|sell) (this|the) fund\b",
    r"\bportfolio (advice|recommendation)\b",
    r"\bhow much should i (sip|invest)\b",
]

COMPARE_RETURNS_PATTERNS = [
    r"\bcompar(e|ison|ing)\b.*\b(returns?|performance|cagr)\b",
    r"\b(returns?|performance|cagr)\b.*\bcompar(e|ison|ing)\b",
    r"\bwhich (has|gave) (higher|better|more) returns?\b",
    r"\bcalculate (my )?(returns?|cagr|sip)\b",
    r"\bprojected returns?\b",
    r"\bif i (invest|sip)\b.+\b(would|will|grow|become|return)\b",
]


@dataclass
class GuardrailResult:
    blocked: bool
    kind: str | None
    message: str | None
    citation: str | None = None


def _citation_for(question: str) -> str:
    q = question.lower()
    aliases = (
        ("elss", FUNDS[2]["url"]),
        ("tax saver", FUNDS[2]["url"]),
        ("flexi", FUNDS[1]["url"]),
        ("equity fund", FUNDS[1]["url"]),
        ("balanced", FUNDS[4]["url"]),
        ("hybrid", FUNDS[4]["url"]),
        ("baf", FUNDS[4]["url"]),
        ("small cap", FUNDS[3]["url"]),
        ("small-cap", FUNDS[3]["url"]),
        ("large cap", FUNDS[0]["url"]),
        ("large-cap", FUNDS[0]["url"]),
    )
    for alias, url in aliases:
        if alias in q:
            return url
    return DEFAULT_CITATION


def _footer(citation: str) -> str:
    return f"Source: {citation}\nLast updated from sources: {SOURCE_AS_OF}"


def check_pii(question: str) -> GuardrailResult:
    text = question.strip()
    for pattern, label in PII_PATTERNS:
        flags = re.I if label in {"OTP", "email", "account", "phone"} else 0
        if re.search(pattern, text, flags):
            citation = DEFAULT_CITATION
            msg = (
                "I cannot accept or store personal identifiers such as PAN, Aadhaar, "
                "account numbers, OTPs, emails, or phone numbers. Ask a factual fund "
                f"question without sharing {label}."
            )
            return GuardrailResult(True, "pii", f"{msg}\n\n{_footer(citation)}", citation)
    return GuardrailResult(False, None, None)


def check_advice(question: str) -> GuardrailResult:
    q = question.lower()
    if any(re.search(p, q) for p in ADVICE_PATTERNS):
        citation = _citation_for(question)
        msg = (
            "I cannot give buy, sell, or portfolio advice. I only share facts published "
            "on the allowed Groww fund pages, such as expense ratio, SIP minimum, exit load, "
            "risk rating, and benchmark."
        )
        return GuardrailResult(True, "advice", f"{msg}\n\n{_footer(citation)}", citation)
    return GuardrailResult(False, None, None)


def check_performance(question: str) -> GuardrailResult:
    q = question.lower()
    if any(re.search(p, q) for p in COMPARE_RETURNS_PATTERNS):
        citation = _citation_for(question)
        msg = (
            "I do not compute or compare returns. Published NAV and return figures, if you "
            "need them, are on the official Groww fund page (treat that page as the factsheet)."
        )
        return GuardrailResult(True, "performance", f"{msg}\n\n{_footer(citation)}", citation)
    return GuardrailResult(False, None, None)


def apply_guardrails(question: str) -> GuardrailResult:
    stripped = question.strip()
    if not stripped:
        return GuardrailResult(
            True,
            "empty",
            f"Please ask a factual question about one of the five HDFC funds.\n\n{_footer(DEFAULT_CITATION)}",
            DEFAULT_CITATION,
        )
    for checker in (check_pii, check_advice, check_performance):
        result = checker(stripped)
        if result.blocked:
            return result
    return GuardrailResult(False, None, None)
