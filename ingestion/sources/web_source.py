"""
Web & Investor Relations Document Source.

Downloads filings and reports directly from investor relations portals or configured manifests.
"""

from pathlib import Path
import re
from typing import List, Optional
import requests
import yaml
from ingestion.sources.base import DocumentMetadata, DocumentSource


class WebFilingSource(DocumentSource):
    """Source that pulls reports from configured URLs or sources.yaml."""

    def __init__(self, sources_yaml_path: Optional[Path] = None):
        super().__init__(name="WebFilingSource")
        self.sources_yaml_path = sources_yaml_path

    def discover_documents(self) -> List[DocumentMetadata]:
        if not self.sources_yaml_path or not self.sources_yaml_path.exists():
            return []

        with open(self.sources_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        company_info = config.get("company", {})
        company_name = company_info.get("name", "LTIMindtree Limited")
        ticker = company_info.get("ticker", "LTIM")

        documents = []
        for key, src in config.get("sources", {}).items():
            doc_url = src.get("document_url")
            if not doc_url or not src.get("enabled", True):
                continue

            name = src.get("name", key)
            fy_match = re.search(r"(FY\d{2,4}(?:-\d{2,4})?)", name, re.IGNORECASE)
            fiscal_year = fy_match.group(1).upper() if fy_match else "FY2025-26"
            doc_id = f"{ticker}_{fiscal_year}_Annual_Report"

            meta = DocumentMetadata(
                document_id=doc_id,
                company=company_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                document_type="annual_report",
                title=name,
                source_type="web",
                source_url=doc_url,
            )
            documents.append(meta)

        return documents

    def download_document(self, metadata: DocumentMetadata, target_dir: Path) -> Path:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        local_filename = f"{metadata.document_id}.pdf"
        local_path = target_dir / local_filename

        if local_path.exists():
            metadata.local_path = local_path
            metadata.content_hash = self.compute_sha256(local_path)
            return local_path

        headers = {
            "User-Agent": "Mozilla/5.0 (Financial GraphRAG Ingestion System)"
        }
        resp = requests.get(metadata.source_url, headers=headers, timeout=120)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)

        metadata.local_path = local_path
        metadata.content_hash = self.compute_sha256(local_path)
        return local_path
