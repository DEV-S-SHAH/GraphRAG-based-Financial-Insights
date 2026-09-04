from pathlib import Path

from ingestion.pdf_parser import parse_pdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annual_reports"
    / "LTM_FY2024-25_Annual_Report.pdf"
)


def test_pdf_exists():

    assert PDF_PATH.exists()


def test_pdf_can_be_parsed():

    pages = parse_pdf(PDF_PATH)

    assert len(pages) > 0


def test_pages_have_text():

    pages = parse_pdf(PDF_PATH)

    pages_with_text = [
        page
        for page in pages
        if page["text"].strip()
    ]

    assert len(pages_with_text) > 0


def test_page_numbers_are_preserved():

    pages = parse_pdf(PDF_PATH)

    assert pages[0]["page"] == 1