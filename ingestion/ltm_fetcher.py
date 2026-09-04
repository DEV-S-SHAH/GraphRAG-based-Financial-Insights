from pathlib import Path
from datetime import datetime, timezone
import hashlib

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES_FILE = PROJECT_ROOT / "sources.yaml"

RAW_ANNUAL_REPORTS = (
    PROJECT_ROOT / "data" / "raw" / "annual_reports"
)

RAW_ANNUAL_REPORTS.mkdir(
    parents=True,
    exist_ok=True
)


def load_sources():
    """Load source configuration."""

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_file(url: str, output_path: Path) -> Path:
    """Download and preserve the original document."""

    print(f"Downloading:")
    print(url)
    print()

    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(LTM Financial GraphRAG Research)"
            )
        },
    )

    response.raise_for_status()

    with open(output_path, "wb") as file:
        file.write(response.content)

    print(f"Saved:")
    print(output_path)

    return output_path


def create_document_metadata(
    file_path: Path,
    source_url: str,
    document_id: str,
):
    """Create provenance metadata."""

    content_hash = calculate_sha256(file_path)

    metadata = {
        "document_id": document_id,
        "company": "LTM Limited",
        "former_name": "LTIMindtree Limited",
        "ticker": "LTM",
        "isin": "INE214T01019",
        "source": "LTM Investor Relations",
        "source_url": source_url,
        "document_type": "annual_report",
        "publication_date": None,
        "reporting_period": "FY2025-26",
        "retrieval_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "raw_file_path": str(file_path),
        "content_hash": content_hash,
    }

    return metadata


def main():

    print("LTM Annual Report Fetcher")
    print("=" * 50)

    config = load_sources()

    report_config = config["sources"][
        "fy2026_annual_report"
    ]

    report_url = report_config["document_url"]

    document_id = "LTM_FY26_AR_001"

    output_file = (
        RAW_ANNUAL_REPORTS
        / "LTM_FY2025-26_Annual_Report.pdf"
    )

    # Idempotency check
    if output_file.exists():

        print("File already exists.")

        existing_hash = calculate_sha256(
            output_file
        )

        print(f"Existing SHA-256:")
        print(existing_hash)

        return

    download_file(
        report_url,
        output_file,
    )

    metadata = create_document_metadata(
        output_file,
        report_url,
        document_id,
    )

    print("\nDocument metadata")
    print("-" * 50)

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()