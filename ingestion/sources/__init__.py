from ingestion.sources.base import DocumentMetadata, DocumentSource
from ingestion.sources.local_pdf import LocalPDFSource
from ingestion.sources.sec_edgar import SECEDGARSource
from ingestion.sources.web_source import WebFilingSource

__all__ = [
    "DocumentMetadata",
    "DocumentSource",
    "LocalPDFSource",
    "SECEDGARSource",
    "WebFilingSource",
]
