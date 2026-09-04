"""
Phase 2: Download real, source-grounded LTIMindtree annual reports.

Downloads the official integrated annual reports for FY2022-23, FY2023-24 and
FY2024-25 from LTIMindtree's (now LTM) official investor sites, writes them into
data/raw/annual_reports/ with a metadata manifest, and verifies each download.

Verification:
  - HTTP 200 + Content-Type application/pdf
  - file non-empty and starts with the PDF magic bytes '%PDF'
  - optional page-count verification using PyMuPDF (fitz) if available

The script is idempotent: an already-downloaded (and verified) file is skipped
unless --force is passed.

Usage:
    python scripts/download_data.py [--force] [--no-verify-pages]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw" / "annual_reports"
METADATA_DIR = ROOT / "data" / "metadata"
MANIFEST_FILE = METADATA_DIR / "documents_manifest.json"

DOWNLOADS_DIR = "annual_reports"  # reserved

COMPANY = "LTIMindtree Limited"
TICKER = "LTIM"

# Global annual-report metadata. The URL for each fiscal year points to the
# official integrated annual report hosted on LTIMindtree's investor site.
DOCUMENTS = [
    {
        "fiscal_year": "FY2022-23",
        "label": "FY2022-23",
        "year_start": "2022-04-01",
        "year_end": "2023-03-31",
        "document_id": "LTIM_FY2022-23_Annual_Report",
        "filename": "LTM_FY2022-23_Annual_Report.pdf",
        "source_url": (
            "https://www.ltm.com/content/dam/ltimcorporatewebsite/"
            "annual-reports-2023/pdfs/ltm-annual-report-22-23.pdf"
        ),
        "description": "Integrated Annual Report 2022-23 (first LTI+Mindtree report)",
        "expected_min_bytes": 10_000_000,
    },
    {
        "fiscal_year": "FY2023-24",
        "label": "FY2023-24",
        "year_start": "2023-04-01",
        "year_end": "2024-03-31",
        "document_id": "LTIM_FY2023-24_Annual_Report",
        "filename": "LTM_FY2023-24_Annual_Report.pdf",
        "source_url": (
            "https://www.ltm.com/content/dam/ltimcorporatewebsite/"
            "annual-report/pdf/Integrated-Annual-Report-FY-2023-24.pdf"
        ),
        "description": "Integrated Annual Report 2023-24",
        "expected_min_bytes": 5_000_000,
    },
    {
        "fiscal_year": "FY2024-25",
        "label": "FY2024-25",
        "year_start": "2024-04-01",
        "year_end": "2025-03-31",
        "document_id": "LTIM_FY2024-25_Annual_Report",
        "filename": "LTM_FY2024-25_Annual_Report.pdf",
        "source_url": (
            "https://www.ltm.com/content/dam/ltimcorporatewebsite/"
            "annual-report-2025/pdfs/integrated-annual-report-fy-2024-25.pdf"
        ),
        "description": "Integrated Annual Report 2024-25",
        "expected_min_bytes": 5_000_000,
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

TIMEOUT = 300  # seconds (FY22-23 report is ~76 MB)


def build_manifest_entry(doc: dict, path: Path, sha256hash: str, pages: Optional[int]):
    """Resolve metadata for one downloaded document."""
    return {
        "company": COMPANY,
        "ticker": TICKER,
        "document_id": doc["document_id"],
        "label": doc["label"],
        "fiscal_year": doc["fiscal_year"],
        "year_start": doc["year_start"],
        "year_end": doc["year_end"],
        "filename": doc["filename"],
        "path": str(path.relative_to(ROOT)),
        "source_url": doc["source_url"],
        "description": doc["description"],
        "sha256": sha256hash,
        "size_bytes": path.stat().st_size,
        "num_pages": pages,
    }


def load_manifest():
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"company": COMPANY, "documents": []}


def save_manifest(manifest):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def verify_pdf(path: Path, expected_min_bytes: int) -> tuple[bool, str]:
    """Verify a PDF is non-empty, has correct magic bytes, and reasonable size."""
    size = path.stat().st_size
    if size < 1000:
        return False, f"too small ({size} bytes)"
    if size < expected_min_bytes:
        return False, (
            f"below expected min size ({size} < {expected_min_bytes} bytes) "
            f"- may be an error page"
        )
    with open(path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        return False, f"missing PDF magic bytes (got {header!r})"
    return True, "ok"


def count_pdf_pages(path: Path):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(doc: dict, force: bool, verify_pages: bool) -> Path:
    dest = RAW_DIR / doc["filename"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        ok, why = verify_pdf(dest, doc["expected_min_bytes"])
        if ok:
            print(f"  SKIP (already downloaded & verified): {doc['filename']}")
            return dest
        print(
            f"  existing file failed verification ({why}); re-downloading"
        )

    print(f"  downloading {doc['filename']}")
    print(f"    from {doc['source_url']}")

    try:
        with requests.get(
            doc["source_url"],
            headers=HEADERS,
            timeout=TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type and "pdf" not in content_type:
                raise RuntimeError(
                    f"unexpected Content-Type: {content_type}"
                )
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
    except Exception as exc:
        if dest.with_suffix(".part").exists():
            dest.with_suffix(".part").unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {doc['filename']}: {exc}")

    ok, why = verify_pdf(dest, doc["expected_min_bytes"])
    if not ok:
        raise RuntimeError(f"verification failed after download: {why}")

    ok, why = verify_pdf(dest, doc["expected_min_bytes"])
    if not ok:
        raise RuntimeError(f"verification failed after download: {why}")

    print(f"  OK {dest.stat().st_size:,} bytes")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if already present")
    ap.add_argument("--no-verify-pages", action="store_true",
                    help="skip PyMuPDF page-count verification")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    manifests = manifest.get("documents", [])
    by_fiscal = {m["fiscal_year"]: m for m in manifests}

    print("=" * 70)
    print("LTM ANNUAL REPORT DOWNLOADER")
    print("=" * 70)
    print(f"Company: {COMPANY} (ticker {TICKER})")
    print(f"Documents: {len(DOCUMENTS)}")
    print()

    results = []

    for doc in DOCUMENTS:
        print(f"[{doc['fiscal_year']}]")
        path = download(
            doc,
            force=args.force,
            verify_pages=not args.no_verify_pages,
        )
        pages = count_pdf_pages(path) if not args.no_verify_pages else None
        entry = build_manifest_entry(doc, path, sha256(path), pages)
        by_fiscal[doc["fiscal_year"]] = entry
        results.append(entry)
        print()

    manifest["documents"] = [by_fiscal[d["fiscal_year"]] for d in DOCUMENTS]
    save_manifest(manifest)

    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for r in manifest["documents"]:
        print(
            f"  {r['fiscal_year']:<12} {r['size_bytes']:>12,} bytes "
            f"pages={r['num_pages']}  {r['filename']}"
        )
    print()
    print(f"Manifest written to {MANIFEST_FILE}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
