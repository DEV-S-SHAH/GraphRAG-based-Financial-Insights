"""
Document Ingestion CLI.

Supports:
- Local PDF annual reports
- SEC EDGAR API discovery and download
- Web / Investor Relations filings

Usage:
    python scripts/ingest.py --source local
    python scripts/ingest.py --source sec --ticker AAPL
    python scripts/ingest.py --source web
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pipeline import FinancialIngestionPipeline
from ingestion.sources.local_pdf import LocalPDFSource
from ingestion.sources.sec_edgar import SECEDGARSource
from ingestion.sources.web_source import WebFilingSource


def main():
    parser = argparse.ArgumentParser(description="Ingest financial filings into PostgreSQL and Neo4j.")
    parser.add_argument(
        "--source",
        choices=["all", "local", "sec", "web", "chunks"],
        default="local",
        help="Source to ingest from: 'chunks' (instant load from data/chunks), 'local' (raw PDFs), 'sec', 'web', 'all'",
    )
    parser.add_argument("--ticker", default="LTIM", help="Company ticker (default: LTIM)")
    parser.add_argument("--max-docs", type=int, default=5, help="Maximum documents to process")

    args = parser.parse_args()

    pipeline = FinancialIngestionPipeline()

    if args.source == "chunks":
        chunks_dir = PROJECT_ROOT / "data" / "chunks"
        pipeline.ingest_from_chunks(chunks_dir=chunks_dir)
        return

    raw_dir = PROJECT_ROOT / "data" / "raw" / "annual_reports"
    sources = []

    if args.source in ["all", "local"]:
        sources.append(LocalPDFSource(directory_path=raw_dir))
    if args.source in ["all", "sec"]:
        sources.append(SECEDGARSource(ticker=args.ticker))
    if args.source in ["all", "web"]:
        sources.append(WebFilingSource(sources_yaml_path=PROJECT_ROOT / "sources.yaml"))

    pipeline.run_ingestion(sources=sources, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
