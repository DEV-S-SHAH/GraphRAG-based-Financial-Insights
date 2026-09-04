from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KPI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_financial_facts.json"
)

NARRATIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_narrative_facts.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_canonical.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def fact_key(fact):

    return (
        fact["metric"],
        fact["value"],
        fact["unit"],
        fact["period"]["label"]
    )


def merge_facts():

    kpi_data = load_json(KPI_PATH)
    narrative_data = load_json(NARRATIVE_PATH)

    all_facts = (
        kpi_data["facts"]
        + narrative_data["facts"]
    )

    unique_facts = []
    seen = set()

    for fact in all_facts:

        key = fact_key(fact)

        if key not in seen:

            seen.add(key)
            unique_facts.append(fact)

    return unique_facts


def validate_canonical_facts(facts):

    if not facts:
        raise ValueError(
            "Canonical dataset is empty"
        )

    for fact in facts:

        required_fields = [
            "fact_id",
            "metric",
            "value",
            "unit",
            "period",
            "company",
            "ticker",
            "source",
            "confidence",
            "extraction_method"
        ]

        for field in required_fields:

            if field not in fact:
                raise ValueError(
                    f"Missing field "
                    f"{field} in "
                    f"{fact.get('metric')}"
                )

        if fact["value"] is None:

            raise ValueError(
                f"Missing value for "
                f"{fact['metric']}"
            )

        if fact["confidence"] < 0.0:

            raise ValueError(
                f"Invalid confidence "
                f"for {fact['metric']}"
            )

        if fact["confidence"] > 1.0:

            raise ValueError(
                f"Invalid confidence "
                f"for {fact['metric']}"
            )

        if not fact["source"].get(
            "document_id"
        ):

            raise ValueError(
                f"Missing document provenance "
                f"for {fact['metric']}"
            )

        if not fact["source"].get(
            "page"
        ):

            raise ValueError(
                f"Missing page provenance "
                f"for {fact['metric']}"
            )


def save_canonical(facts):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "document_id": "LTM_FY26_AR_001",
        "company": "LTM Limited",
        "ticker": "LTM",
        "reporting_period": "FY2025-26",

        "dataset_type": (
            "canonical_financial_facts"
        ),

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
        "LTM Canonical Financial Fact Merger"
    )

    print("=" * 50)

    facts = merge_facts()

    print(
        f"\nFacts before validation: "
        f"{len(facts)}"
    )

    validate_canonical_facts(
        facts
    )

    save_canonical(
        facts
    )

    print(
        "\nValidation successful."
    )

    print(
        f"Canonical facts: "
        f"{len(facts)}"
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()