import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_financial_facts.json"
)


def load_data():

    with open(
        FACTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def test_facts_file_exists():

    assert FACTS_PATH.exists()


def test_expected_metrics_are_present():

    data = load_data()

    metrics = {
        fact["metric"]
        for fact in data["facts"]
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

    assert expected.issubset(
        metrics
    )


def test_revenue_value():

    data = load_data()

    revenue = next(
        fact
        for fact in data["facts"]
        if fact["metric"] == "revenue"
    )

    assert revenue["value"] == 423076
    assert revenue["unit"] == "INR million"


def test_ebitda_value():

    data = load_data()

    ebitda = next(
        fact
        for fact in data["facts"]
        if fact["metric"] == "ebitda"
    )

    assert ebitda["value"] == 75552


def test_pat_value():

    data = load_data()

    pat = next(
        fact
        for fact in data["facts"]
        if fact["metric"]
        == "profit_after_tax"
    )

    assert pat["value"] == 49827


def test_eps_value():

    data = load_data()

    eps = next(
        fact
        for fact in data["facts"]
        if fact["metric"]
        == "eps_diluted"
    )

    assert eps["value"] == 169.13


def test_page_provenance():

    data = load_data()

    for fact in data["facts"]:

        assert (
            fact["source"]["page"]
            == 23
        )

        assert (
            fact["source"]["document_id"]
            == "LTM_FY26_AR_001"
        )


def test_confidence():

    data = load_data()

    for fact in data["facts"]:

        assert (
            fact["confidence"]
            == 1.0
        )