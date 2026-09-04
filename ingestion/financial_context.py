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
    document_id,
    source_type="KPI"
):

    return {
        "fact_id": (
            f"{document_id}_"
            f"{metric}_"
            f"FY26"
        ),

        "metric": metric,

        "value": value,

        "unit": unit,

        "period": {
            "label": "FY2025-26",
            "start": "2025-04-01",
            "end": "2026-03-31"
        },

        "company": "LTM Limited",

        "ticker": "LTM",

        "statement_type": "consolidated",

        "source": {
            "document_id": document_id,
            "page": page,
            "section": source_type
        },

        "confidence": 1.0,

        "extraction_method": "structured_kpi"
    }


def extract_page_23_kpis(document):

    page_number = 23

    text = get_page(
        document,
        page_number
    )

    facts = []

    patterns = [

        (
            "market_capitalization",
            r"Market Capitalization.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "net_worth",
            r"Net Worth.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "revenue",
            r"Revenue.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "ebitda",
            r"EBITDA.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "dividend_paid",
            r"Dividend Paid.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "employees",
            r"Number of Employees\s+([\d,]+)",
            "employees"
        ),

        (
            "eps_basic",
            r"EPS \(Basic\).*?"
            r"\(INR\).*?"
            r"([\d.]+)",
            "INR per share"
        ),

        (
            "eps_diluted",
            r"EPS \(Diluted\).*?"
            r"\(INR\).*?"
            r"([\d.]+)",
            "INR per share"
        ),

        (
            "return_on_equity",
            r"Return on Equity.*?"
            r"\(in %\).*?"
            r"([\d.]+)",
            "%"
        ),

        (
            "csr_spend",
            r"CSR Spend.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        ),

        (
            "profit_after_tax",
            r"Profit After Tax.*?"
            r"\(INR in Million\).*?"
            r"([\d,]+)",
            "INR million"
        )
    ]

    for metric, pattern, unit in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE |
                  re.DOTALL
        )

        if not match:
            continue

        raw_value = match.group(1)

        raw_value = raw_value.replace(
            ",",
            ""
        )

        if unit in [
            "employees"
        ]:
            value = int(raw_value)

        elif unit == "%":

            value = float(
                raw_value
            )

        else:

            value = float(
                raw_value
            )

        facts.append(
            create_fact(
                metric=metric,
                value=value,
                unit=unit,
                page=page_number,
                document_id=document[
                    "document_id"
                ]
            )
        )

    return facts


def main():

    print(
        "LTM Financial Fact Extractor"
    )

    print("=" * 50)

    document = load_document()

    facts = extract_page_23_kpis(
        document
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "document_id": document[
            "document_id"
        ],

        "company": "LTM Limited",

        "ticker": "LTM",

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

    print(
        f"\nValidated KPI facts: "
        f"{len(facts)}"
    )

    print()

    for fact in facts:

        print(
            f"{fact['metric']:25}"
            f"{fact['value']:>15}"
            f"  {fact['unit']}"
            f"  page={fact['source']['page']}"
        )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()