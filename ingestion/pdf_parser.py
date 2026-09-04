from pathlib import Path
from typing import List, Dict
import json

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annual_reports"
    / "LTM_FY2025-26_Annual_Report.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pdf"
    / "LTM_FY2025-26_parsed.json"
)


def parse_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extract text from every PDF page.
    """

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


def save_parsed_document(
    pages: List[Dict],
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "document_id": "LTM_FY26_AR_001",
        "company": "LTM Limited",
        "former_name": "LTIMindtree Limited",
        "ticker": "LTM",
        "document_type": "annual_report",
        "reporting_period": "FY2025-26",
        "source": "LTM Investor Relations",
        "pages": pages,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Parsing:")
    print(PDF_PATH)
    print()

    pages = parse_pdf(PDF_PATH)

    print(
        f"Extracted {len(pages)} pages."
    )

    save_parsed_document(
        pages,
        OUTPUT_PATH
    )

    print()
    print("Saved parsed document:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()