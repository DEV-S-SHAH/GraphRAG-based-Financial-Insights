"""
Base DocumentSource Interface.

Provides pluggable document ingestion from local folders, SEC EDGAR API,
or corporate web filings with duplicate detection via content hashing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentMetadata:
    document_id: str
    company: str
    ticker: str
    fiscal_year: str
    document_type: str  # e.g., "annual_report", "10-K", "10-Q"
    title: str
    source_type: str    # "local", "sec_edgar", "web"
    source_url: str
    local_path: Optional[Path] = None
    content_hash: Optional[str] = None
    publication_date: Optional[str] = None
    retrieval_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentSource(ABC):
    """Abstract base class for all document sources."""

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file for idempotency and duplicate checking."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    @abstractmethod
    def discover_documents(self) -> List[DocumentMetadata]:
        """Discover available documents from the source."""
        pass

    @abstractmethod
    def download_document(self, metadata: DocumentMetadata, target_dir: Path) -> Path:
        """Download document if not already local, returning path to local file."""
        pass

    def is_duplicate(self, metadata: DocumentMetadata, existing_hashes: set) -> bool:
        """Check if document has already been ingested using its content hash or ID."""
        if metadata.content_hash and metadata.content_hash in existing_hashes:
            return True
        return False
