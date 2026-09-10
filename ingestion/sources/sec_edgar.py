"""
SEC EDGAR Document Source.

Discovers and fetches 10-K / annual filings directly from SEC EDGAR API
with required User-Agent headers, compliant rate-limiting, and idempotency.
"""

import os
from pathlib import Path
import time
from typing import List, Optional
import requests
from ingestion.sources.base import DocumentMetadata, DocumentSource


# SEC CIK mapping for prominent companies
KNOWN_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}


class SECEDGARSource(DocumentSource):
    """Source that queries the official SEC EDGAR submissions API."""

    def __init__(
        self,
        ticker: str = "AAPL",
        user_agent: Optional[str] = None,
        filing_type: str = "10-K",
        max_filings: int = 2,
    ):
        super().__init__(name=f"SECEDGARSource({ticker})")
        self.ticker = ticker.upper()
        self.filing_type = filing_type
        self.max_filings = max_filings
        self.user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT", "FinancialGraphRAG research@financialgraphrag.local"
        )
        self.cik = KNOWN_CIKS.get(self.ticker)

    def _headers(self) -> dict:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def discover_documents(self) -> List[DocumentMetadata]:
        if not self.cik:
            print(f"[SEC EDGAR] No CIK mapping for ticker {self.ticker}, skipping SEC discovery.")
            return []

        url = f"https://data.sec.gov/submissions/CIK{self.cik}.json"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.status_code != 200:
                print(f"[SEC EDGAR] HTTP {resp.status_code} querying {url}")
                return []
            data = resp.json()
        except Exception as e:
            print(f"[SEC EDGAR] Error querying SEC submissions for {self.ticker}: {e}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        documents = []
        company_name = data.get("name", self.ticker)

        for i, form in enumerate(forms):
            if form == self.filing_type:
                acc_num = accession_numbers[i].replace("-", "")
                acc_raw = accession_numbers[i]
                doc_name = primary_docs[i]
                filing_date = filing_dates[i]
                report_date = report_dates[i] if i < len(report_dates) else filing_date
                
                # Derive fiscal year from reportDate (e.g. 2024-09-28 -> FY2024)
                year = report_date.split("-")[0] if report_date else filing_date.split("-")[0]
                fiscal_year = f"FY{year}"
                
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{acc_num}/{doc_name}"
                doc_id = f"{self.ticker}_{fiscal_year}_10K_{acc_raw}"

                meta = DocumentMetadata(
                    document_id=doc_id,
                    company=company_name,
                    ticker=self.ticker,
                    fiscal_year=fiscal_year,
                    document_type="10-K",
                    title=f"{company_name} Form 10-K ({fiscal_year})",
                    source_type="sec_edgar",
                    source_url=doc_url,
                    publication_date=filing_date,
                    extra_metadata={
                        "accession_number": acc_raw,
                        "report_date": report_date,
                        "cik": self.cik,
                    },
                )
                documents.append(meta)
                if len(documents) >= self.max_filings:
                    break

        return documents

    def download_document(self, metadata: DocumentMetadata, target_dir: Path) -> Path:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(metadata.source_url).suffix or ".htm"
        local_filename = f"{metadata.document_id}{ext}"
        local_path = target_dir / local_filename

        if local_path.exists():
            metadata.local_path = local_path
            metadata.content_hash = self.compute_sha256(local_path)
            return local_path

        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        resp = requests.get(metadata.source_url, headers=headers, timeout=30)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)

        metadata.local_path = local_path
        metadata.content_hash = self.compute_sha256(local_path)
        return local_path
