"""
Local PDF Document Source.

Scans local directories for company annual reports and filings,
deriving metadata from filenames or manifests.
"""

from pathlib import Path
import re
from typing import List, Optional
from ingestion.sources.base import DocumentMetadata, DocumentSource


class LocalPDFSource(DocumentSource):
    """Source that loads PDFs from a local directory."""

    def __init__(self, directory_path: Path, company_default: str = "LTIMindtree Limited", ticker_default: str = "LTIM"):
        super().__init__(name="LocalPDFSource")
        self.directory_path = Path(directory_path)
        self.company_default = company_default
        self.ticker_default = ticker_default

    def discover_documents(self) -> List[DocumentMetadata]:
        if not self.directory_path.exists():
            return []

        documents = []
        for pdf_file in sorted(self.directory_path.glob("*.pdf")):
            name = pdf_file.stem
            
            # Extract fiscal year if present (e.g. FY2022-23, FY2023-24)
            fy_match = re.search(r"(FY\d{4}(?:-\d{2,4})?)", name, re.IGNORECASE)
            fiscal_year = fy_match.group(1).upper() if fy_match else "FY2025-26"
            
            # Standardize document id
            doc_id = name.replace(" ", "_")
            file_hash = self.compute_sha256(pdf_file)

            meta = DocumentMetadata(
                document_id=doc_id,
                company=self.company_default,
                ticker=self.ticker_default,
                fiscal_year=fiscal_year,
                document_type="annual_report",
                title=name.replace("_", " "),
                source_type="local",
                source_url=f"file://{pdf_file.resolve()}",
                local_path=pdf_file,
                content_hash=file_hash,
            )
            documents.append(meta)

        return documents

    def download_document(self, metadata: DocumentMetadata, target_dir: Path) -> Path:
        """For local files, returns the existing local path or copies if needed."""
        if metadata.local_path and metadata.local_path.exists():
            return metadata.local_path
        raise FileNotFoundError(f"Local file not found for {metadata.document_id}")
