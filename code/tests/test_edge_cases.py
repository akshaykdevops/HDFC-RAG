"""Edge-case tests for the facts-only Groww HDFC FAQ prototype."""

from code.policies import apply_guardrails, check_advice, check_performance, check_pii


def test_empty_query_blocked():
    result = apply_guardrails("   ")
    assert result.blocked
    assert result.kind == "empty"


def test_pan_rejected():
    result = check_pii("What is my folio ABCDE1234F expense ratio?")
    assert result.blocked
    assert result.kind == "pii"
    assert "Source:" in result.message
    assert "Last updated from sources:" in result.message


def test_email_rejected():
    result = check_pii("Mail the factsheet to user@example.com")
    assert result.blocked
    assert result.kind == "pii"


def test_phone_rejected():
    result = check_pii("Call me on 9876543210 about SIP")
    assert result.blocked


def test_otp_rejected():
    result = check_pii("My OTP is 123456, show NAV")
    assert result.blocked


def test_buy_advice_refused():
    result = check_advice("Should I buy HDFC Small Cap Fund?")
    assert result.blocked
    assert result.kind == "advice"
    assert "cannot give buy" in result.message.lower()
    assert "hdfc-small-cap" in result.citation


def test_best_fund_refused():
    result = check_advice("Which fund is better for my portfolio?")
    assert result.blocked
    assert result.kind == "advice"


def test_compare_returns_refused():
    result = check_performance("Compare returns of large cap vs small cap")
    assert result.blocked
    assert result.kind == "performance"
    assert "do not compute" in result.message.lower()


def test_projected_sip_refused():
    result = check_performance("If I SIP 5000 would it grow to a crore?")
    assert result.blocked


def test_factual_question_passes():
    result = apply_guardrails("What is the expense ratio of HDFC Large Cap Fund Direct Growth?")
    assert not result.blocked


def test_elss_sip_passes_guardrails():
    result = apply_guardrails("What is the minimum SIP for HDFC ELSS Tax Saver?")
    assert not result.blocked


def test_capital_gains_statement_not_pii_or_advice():
    """Out-of-corpus process questions must reach RAG, not a false guardrail."""
    result = apply_guardrails("How to download capital-gains statement?")
    assert not result.blocked


def test_citation_always_present_on_blocks():
    for q in (
        "Should I sell HDFC Flexi Cap?",
        "ABCDE1234F",
        "Compare returns of these two funds",
    ):
        result = apply_guardrails(q)
        assert result.blocked
        assert result.citation.startswith("https://groww.in/")
        assert "Last updated from sources:" in (result.message or "")
