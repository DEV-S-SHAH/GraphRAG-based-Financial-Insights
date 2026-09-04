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
    / "LTM_FY2025-26_narrative_facts.json"
)


DOCUMENT_ID = "LTM_FY26_AR_001"
COMPANY = "LTM Limited"
TICKER = "LTM"


def load_document():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_page(document, page_number):

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
    section
):

    return {
        "fact_id": (
            f"{DOCUMENT_ID}_"
            f"{metric}_FY26_narrative"
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

        "extraction_method": "narrative_rule_based"
    }


def extract_number(
    text,
    pattern
):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE |
        re.DOTALL
    )

    if not match:
        return None

    return match.group(1)


def extract_page_8_facts(document):

    page_number = 8

    text = get_page(
        document,
        page_number
    )

    facts = []

    # --------------------------------------------------
    # USD Revenue
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Revenue for FY26 stood at "
        r"USD\s+([\d.]+)\s+Billion"
    )

    if value:

        facts.append(
            create_fact(
                "revenue_usd",
                float(value),
                "USD billion",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # USD Revenue Growth
    # --------------------------------------------------

    value = extract_number(
        text,
        r"reflecting a growth of "
        r"([\d.]+)%\s+in constant currency"
    )

    if value:

        facts.append(
            create_fact(
                "revenue_growth_constant_currency",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # USD Revenue Growth
    # --------------------------------------------------

    value = extract_number(
        text,
        r"and\s+([\d.]+)%\s+growth in USD terms"
    )

    if value:

        facts.append(
            create_fact(
                "revenue_growth_usd",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # Order Inflow
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Order inflow remained robust at "
        r"USD\s+([\d.]+)\s+Billion"
    )

    if value:

        facts.append(
            create_fact(
                "order_inflow",
                float(value),
                "USD billion",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # Order Inflow Growth
    # --------------------------------------------------

    value = extract_number(
        text,
        r"year-on-year change of "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "order_inflow_growth",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # INR Revenue
    # --------------------------------------------------

    value = extract_number(
        text,
        r"INR revenue stood at "
        r"INR\s+([\d,]+)\s+Million"
    )

    if value:

        facts.append(
            create_fact(
                "revenue_inr",
                float(value.replace(",", "")),
                "INR million",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # INR Revenue Growth
    # --------------------------------------------------

    value = extract_number(
        text,
        r"year-on-year growth of "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "revenue_growth_inr",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # EBITDA Margin
    # --------------------------------------------------

    value = extract_number(
        text,
        r"EBITDA margin stood at "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "ebitda_margin",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # EBIT Margin
    # --------------------------------------------------

    value = extract_number(
        text,
        r"EBIT margin was "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "ebit_margin",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # Net Profit
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Net profit for the year stood at "
        r"INR\s+([\d,]+)\s+Million"
    )

    if value:

        facts.append(
            create_fact(
                "net_profit",
                float(value.replace(",", "")),
                "INR million",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # PAT Margin
    # --------------------------------------------------

    value = extract_number(
        text,
        r"with PAT margin at "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "pat_margin",
                float(value),
                "%",
                page_number,
                "Financial Performance and Quality of Earnings"
            )
        )

    # --------------------------------------------------
    # ROE
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Return on Equity \(ROE\).*?"
        r"stood at "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "roe",
                float(value),
                "%",
                page_number,
                "Capital Allocation and Investment Discipline"
            )
        )

    # --------------------------------------------------
    # ROCE
    # --------------------------------------------------

    value = extract_number(
        text,
        r"and "
        r"([\d.]+)%\s+respectively"
    )

    if value:

        facts.append(
            create_fact(
                "roce",
                float(value),
                "%",
                page_number,
                "Capital Allocation and Investment Discipline"
            )
        )

    # --------------------------------------------------
    # Diluted EPS
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Diluted Earnings Per Share for FY26 was "
        r"INR\s+([\d.]+)"
    )

    if value:

        facts.append(
            create_fact(
                "eps_diluted",
                float(value),
                "INR per share",
                page_number,
                "Capital Allocation and Investment Discipline"
            )
        )

    # --------------------------------------------------
    # Dividend
    # --------------------------------------------------

    value = extract_number(
        text,
        r"total dividends of "
        r"INR\s+([\d,]+)\s+Million"
    )

    if value:

        facts.append(
            create_fact(
                "dividend_paid",
                float(value.replace(",", "")),
                "INR million",
                page_number,
                "Capital Allocation and Investment Discipline"
            )
        )

    # --------------------------------------------------
    # Cash and Investments
    # --------------------------------------------------

    value = extract_number(
        text,
        r"stood at "
        r"INR\s+([\d,]+)\s+Million "
        r"as of March 31, 2026"
    )

    if value:

        facts.append(
            create_fact(
                "cash_and_investments",
                float(value.replace(",", "")),
                "INR million",
                page_number,
                "Funding, Liquidity and Balance Sheet Strength"
            )
        )

    # --------------------------------------------------
    # Operating Cash Flow Conversion
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Operating cash flow conversion remained robust at "
        r"([\d.]+)%"
    )

    if value:

        facts.append(
            create_fact(
                "operating_cash_flow_conversion",
                float(value),
                "%",
                page_number,
                "Funding, Liquidity and Balance Sheet Strength"
            )
        )

    # --------------------------------------------------
    # DSO
    # --------------------------------------------------

    value = extract_number(
        text,
        r"Days Sales Outstanding \(DSO\) stood at "
        r"([\d]+)\s+days"
    )

    if value:

        facts.append(
            create_fact(
                "days_sales_outstanding",
                int(value),
                "days",
                page_number,
                "Funding, Liquidity and Balance Sheet Strength"
            )
        )

    # --------------------------------------------------
    # Current Ratio
    # --------------------------------------------------

    value = extract_number(
        text,
        r"The current ratio of "
        r"([\d.]+)"
    )

    if value:

        facts.append(
            create_fact(
                "current_ratio",
                float(value),
                "ratio",
                page_number,
                "Funding, Liquidity and Balance Sheet Strength"
            )
        )

    return facts


def validate_facts(facts):

    if not facts:
        raise ValueError(
            "No narrative facts extracted"
        )

    for fact in facts:

        if fact["value"] is None:
            raise ValueError(
                f"Missing value: "
                f"{fact['metric']}"
            )

        if fact["source"]["page"] != 8:
            raise ValueError(
                f"Unexpected page for "
                f"{fact['metric']}"
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
        "LTM Narrative Financial Extractor"
    )

    print("=" * 50)

    document = load_document()

    facts = extract_page_8_facts(
        document
    )

    print(
        f"\nNarrative facts found: "
        f"{len(facts)}"
    )

    validate_facts(facts)

    print(
        "Validation successful."
    )

    save_facts(facts)

    print()

    for fact in facts:

        print(
            f"{fact['metric']:40}"
            f"{str(fact['value']):>12}"
            f"  {fact['unit']}"
        )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()