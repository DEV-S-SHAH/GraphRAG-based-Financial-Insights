import json
from pathlib import Path

from ingestion.normalizer import (
    build_normalized_document,
    load_parsed_document,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)


def test_normalized_file_exists():

    assert OUTPUT_PATH.exists()


def test_normalized_document_has_required_fields():

    with open(
        OUTPUT_PATH,
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    required_fields = [
        "document_id",
        "company",
        "document_type",
        "title",
        "source",
        "published_date",
        "reporting_period",
        "content",
        "financial_facts",
        "entities",
        "relationships",
        "metadata",
    ]

    for field in required_fields:
        assert field in document


def test_company_information():

    with open(
        OUTPUT_PATH,
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    assert (
        document["company"]["name"]
        == "LTM Limited"
    )

    assert (
        document["company"]["ticker"]
        == "LTM"
    )


def test_page_provenance_is_preserved():

    with open(
        OUTPUT_PATH,
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    assert len(document["pages"]) > 0

    assert document["pages"][0]["page"] == 1


def test_content_is_not_empty():

    with open(
        OUTPUT_PATH,
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    assert len(
        document["content"].strip()
    ) > 0