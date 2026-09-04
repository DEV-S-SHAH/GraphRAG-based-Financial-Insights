from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_financial_facts.json"
)


DOCUMENT_ID = "LTM_FY26_AR_001"
COMPANY = "LTM Limited"
TICKER = "LTM"


def load_document():
    """Load the normalized annual report."""

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_page(document, page_number):
    """Return the text of a specific page."""

    for page in document["pages"]:

        if page["page"] == page_number:
            return page["text"]

    raise ValueError(
        f"Page {page_number} not found"
    )


def create_fact(
    metric,
    value,
    unit,
    page,
    section="Key Performance Indicators"
):
    """Create a validated financial fact."""

    return {
        "fact_id": (
            f"{DOCUMENT_ID}_"
            f"{metric}_FY26"
        ),

        "metric": metric,

        "value": value,

        "unit": unit,

        "period": {
            "label": "FY2025-26",
            "start": "2025-04-01",
            "end": "2026-03-31"
        },

        "company": COMPANY,

        "ticker": TICKER,

        "statement_type": "consolidated",

        "source": {
            "document_id": DOCUMENT_ID,
            "page": page,
            "section": section
        },

        "confidence": 1.0,

        "extraction_method": "structured_kpi"
    }


def extract_value(
    text,
    pattern
):
    """Extract the first value matching a pattern."""

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE |
        re.DOTALL
    )

    if not match:
        return None

    return match.group(1)


def clean_number(value):
    """Convert comma-formatted number to float."""

    value = value.replace(
        ",",
        ""
    ).strip()

    return float(value)


def extract_page_23_kpis(document):
    """
    Extract FY26 KPIs from page 23.

    Page 23 contains structured KPI blocks where
    each metric is followed by its FY26 value.
    """

    page_number = 23

    text = get_page(
        document,
        page_number
    )

    facts = []

    # --------------------------------------------------
    # Market Capitalization
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Market Capitalization\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "market_capitalization",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # Net Worth
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Net Worth\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "net_worth",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # Revenue
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Revenue\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "revenue",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # EBITDA
    # --------------------------------------------------

    value = extract_value(
        text,
        r"EBITDA\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "ebitda",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # Dividend Paid
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Dividend Paid\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "dividend_paid",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # Number of Employees
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Number of Employees\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "employees",
                int(
                    value.replace(",", "")
                ),
                "employees",
                page_number
            )
        )

    # --------------------------------------------------
    # EPS Basic
    # --------------------------------------------------

    value = extract_value(
        text,
        r"EPS\s*\(Basic\)\s*"
        r"\(INR\)\s*"
        r"([\d.]+)"
    )

    if value:
        facts.append(
            create_fact(
                "eps_basic",
                float(value),
                "INR per share",
                page_number
            )
        )

    # --------------------------------------------------
    # EPS Diluted
    # --------------------------------------------------

    value = extract_value(
        text,
        r"EPS\s*\(Diluted\)\s*"
        r"\(INR\)\s*"
        r"([\d.]+)"
    )

    if value:
        facts.append(
            create_fact(
                "eps_diluted",
                float(value),
                "INR per share",
                page_number
            )
        )

    # --------------------------------------------------
    # Return on Equity
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Return on Equity\s*"
        r"\(in %\)\s*"
        r"([\d.]+)"
    )

    if value:
        facts.append(
            create_fact(
                "return_on_equity",
                float(value),
                "%",
                page_number
            )
        )

    # --------------------------------------------------
    # CSR Spend
    # --------------------------------------------------

    value = extract_value(
        text,
        r"CSR Spend\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "csr_spend",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    # --------------------------------------------------
    # Profit After Tax
    # --------------------------------------------------

    value = extract_value(
        text,
        r"Profit After Tax\s*"
        r"\(INR in Million\)\s*"
        r"([\d,]+)"
    )

    if value:
        facts.append(
            create_fact(
                "profit_after_tax",
                clean_number(value),
                "INR million",
                page_number
            )
        )

    return facts


def validate_facts(facts):
    """
    Basic validation before writing facts to disk.
    """

    required_metrics = {
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
        "profit_after_tax"
    }

    extracted_metrics = {
        fact["metric"]
        for fact in facts
    }

    missing = (
        required_metrics
        - extracted_metrics
    )

    if missing:

        raise ValueError(
            "Missing expected metrics: "
            + ", ".join(
                sorted(missing)
            )
        )

    for fact in facts:

        if fact["value"] is None:
            raise ValueError(
                f"Missing value for "
                f"{fact['metric']}"
            )

        if fact["source"]["page"] != 23:
            raise ValueError(
                f"Unexpected source page "
                f"for {fact['metric']}"
            )

        if fact["confidence"] != 1.0:
            raise ValueError(
                f"Invalid confidence for "
                f"{fact['metric']}"
            )


def save_facts(facts):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "document_id": DOCUMENT_ID,

        "company": COMPANY,

        "ticker": TICKER,

        "reporting_period": "FY2025-26",

        "fact_count": len(facts),

        "facts": facts
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():

    print(
        "LTM Financial Fact Extractor"
    )

    print("=" * 50)

    document = load_document()

    print(
        f"Document: "
        f"{DOCUMENT_ID}"
    )

    print(
        f"Pages: "
        f"{len(document['pages'])}"
    )

    print(
        "\nExtracting structured KPIs "
        "from page 23..."
    )

    facts = extract_page_23_kpis(
        document
    )

    print(
        f"Candidate KPIs found: "
        f"{len(facts)}"
    )

    print(
        "\nValidating facts..."
    )

    validate_facts(facts)

    print(
        "Validation successful."
    )

    save_facts(facts)

    print(
        f"\nValidated KPI facts: "
        f"{len(facts)}"
    )

    print()

    for fact in facts:

        print(
            f"{fact['metric']:25}"
            f"{str(fact['value']):>15}"
            f"  {fact['unit']:<15}"
            f"page={fact['source']['page']}"
        )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()