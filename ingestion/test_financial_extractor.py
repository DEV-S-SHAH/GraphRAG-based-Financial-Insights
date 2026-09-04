import json
from pathlib import Path

from ingestion.financial_extractor import (
    load_document,
    get_page,
    extract_page_23_kpis,
    validate_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)


def test_normalized_document_exists():

    assert INPUT_PATH.exists()


def test_document_can_be_loaded():

    document = load_document()

    assert "pages" in document
    assert len(document["pages"]) == 275


def test_page_23_exists():

    document = load_document()

    text = get_page(
        document,
        23
    )

    assert text
    assert "Key Performance Indicators" in text


def test_page_23_contains_expected_kpis():

    document = load_document()

    text = get_page(
        document,
        23
    )

    expected_terms = [
        "Revenue",
        "EBITDA",
        "Profit After Tax",
        "EPS (Basic)",
        "EPS (Diluted)",
        "Return on Equity",
    ]

    for term in expected_terms:

        assert term in text


def test_kpi_extraction():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    assert len(facts) == 11


def test_revenue_extraction():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    revenue = next(
        fact
        for fact in facts
        if fact["metric"] == "revenue"
    )

    assert revenue["value"] == 423076
    assert revenue["unit"] == "INR million"


def test_ebitda_extraction():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    ebitda = next(
        fact
        for fact in facts
        if fact["metric"] == "ebitda"
    )

    assert ebitda["value"] == 75552
    assert ebitda["unit"] == "INR million"


def test_pat_extraction():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    pat = next(
        fact
        for fact in facts
        if fact["metric"]
        == "profit_after_tax"
    )

    assert pat["value"] == 49827
    assert pat["unit"] == "INR million"


def test_eps_diluted_extraction():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    eps = next(
        fact
        for fact in facts
        if fact["metric"]
        == "eps_diluted"
    )

    assert eps["value"] == 169.13
    assert eps["unit"] == "INR per share"


def test_all_facts_have_provenance():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    for fact in facts:

        assert (
            fact["source"]["document_id"]
            == "LTM_FY26_AR_001"
        )

        assert (
            fact["source"]["page"]
            == 23
        )

        assert fact["source"]["section"]


def test_all_facts_have_confidence():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    for fact in facts:

        assert fact["confidence"] == 1.0


def test_validation_passes():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    validate_facts(facts)


def test_expected_metrics():

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    metrics = {
        fact["metric"]
        for fact in facts
    }

    expected = {
        "market_capitalization",
        "net_worth",
        "revenue",
        "ebitda",
        "dividend_paid",
        "employees",
        "eps_basic",
        "eps_diluted",
        "return_on_equity",
        "csr_spend",
        "profit_after_tax",
    }

    assert metrics == expected