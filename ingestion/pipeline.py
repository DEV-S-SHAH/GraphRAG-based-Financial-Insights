"""
Unified Ingestion Pipeline for Financial GraphRAG.

Coordinates:
1. Source Discovery (Local PDFs, SEC EDGAR, Web filings)
2. Duplicate Detection via SHA-256 Content Hashing
3. Docling PDF Parsing (preserving tables, sections, headings, page numbers)
4. Semantic Structure-Aware Chunking
5. Embedding Generation (Ollama nomic-embed-text or OpenAI)
6. PostgreSQL + pgvector Insertion
7. Neo4j Knowledge Graph Enrichment
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from database.financial_knowledge_graph import FinancialKnowledgeGraph
from database.postgres_client import PostgresVectorClient
from ingestion.docling_parser import DoclingParser
from ingestion.semantic_chunker import SemanticChunker
from ingestion.sources.base import DocumentMetadata, DocumentSource
from ingestion.sources.local_pdf import LocalPDFSource
from ingestion.sources.sec_edgar import SECEDGARSource
from ingestion.sources.web_source import WebFilingSource
from rag.embeddings_provider import get_embeddings_provider

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FinancialIngestionPipeline:
    """End-to-end ingestion pipeline for corporate financial reports."""

    def __init__(
        self,
        pg_client: Optional[PostgresVectorClient] = None,
        kg_client: Optional[FinancialKnowledgeGraph] = None,
        use_docling: bool = True,
    ):
        self.pg = pg_client or PostgresVectorClient()
        self.kg = kg_client or FinancialKnowledgeGraph()
        self.parser = DoclingParser(use_docling_converter=use_docling)
        self.chunker = SemanticChunker()
        self.embedder = get_embeddings_provider()

    def get_existing_content_hashes(self) -> set:
        """Fetch all ingested content hashes from PostgreSQL."""
        hashes = set()
        try:
            with self.pg.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT content_hash FROM documents WHERE content_hash IS NOT NULL;")
                    for r in cur.fetchall():
                        hashes.add(r[0])
        except Exception as e:
            logger.warning(f"Error fetching existing content hashes: {e}")
        return hashes

    def run_ingestion(
        self,
        sources: Optional[List[DocumentSource]] = None,
        target_dir: Optional[Path] = None,
        max_docs: int = 10,
    ) -> Dict[str, Any]:
        """Execute full ingestion across configured document sources."""
        raw_dir = target_dir or (PROJECT_ROOT / "data" / "raw" / "annual_reports")
        raw_dir.mkdir(parents=True, exist_ok=True)

        if sources is None:
            sources = [
                LocalPDFSource(directory_path=raw_dir),
                WebFilingSource(sources_yaml_path=PROJECT_ROOT / "sources.yaml"),
            ]

        existing_hashes = self.get_existing_content_hashes()
        results = {
            "discovered": 0,
            "downloaded": 0,
            "processed": 0,
            "skipped_duplicate": 0,
            "chunks_created": 0,
            "documents": [],
        }

        print("=" * 70)
        print("FINANCIAL GRAPHRAG INGESTION PIPELINE")
        print("=" * 70)

        for source in sources:
            print(f"\n[SOURCE] Running discovery for {source.name}...")
            docs = source.discover_documents()
            results["discovered"] += len(docs)

            for doc_meta in docs:
                if results["processed"] >= max_docs:
                    break

                # 1. Duplicate check
                if source.is_duplicate(doc_meta, existing_hashes):
                    print(f"  [SKIP] Duplicate detected: {doc_meta.document_id}")
                    results["skipped_duplicate"] += 1
                    continue

                # 2. Download / resolve local path
                try:
                    local_path = source.download_document(doc_meta, raw_dir)
                    results["downloaded"] += 1
                except Exception as e:
                    print(f"  [ERROR] Downloading {doc_meta.document_id}: {e}")
                    continue

                # Re-check hash after download
                if not doc_meta.content_hash:
                    doc_meta.content_hash = DocumentSource.compute_sha256(local_path)
                if doc_meta.content_hash in existing_hashes:
                    print(f"  [SKIP] Duplicate hash {doc_meta.content_hash[:10]} for {doc_meta.document_id}")
                    results["skipped_duplicate"] += 1
                    continue

                print(f"  [PROCESS] Parsing {doc_meta.title} via Docling...")
                # 3. Parse with Docling
                parsed_doc = self.parser.parse_pdf(
                    pdf_path=local_path,
                    document_id=doc_meta.document_id,
                    company=doc_meta.company,
                    ticker=doc_meta.ticker,
                    fiscal_year=doc_meta.fiscal_year,
                    source_url=doc_meta.source_url,
                )

                # 4. Semantic Chunking
                chunks = self.chunker.chunk_document(parsed_doc, source_url=doc_meta.source_url)
                print(f"    -> Extracted {len(chunks)} semantic chunks ({len(parsed_doc.tables)} tables)")

                # 5. Insert Document into PostgreSQL
                self.pg.insert_document({
                    "document_id": doc_meta.document_id,
                    "company": doc_meta.company,
                    "ticker": doc_meta.ticker,
                    "source": source.name,
                    "source_url": doc_meta.source_url,
                    "document_type": doc_meta.document_type,
                    "title": doc_meta.title,
                    "reporting_period": doc_meta.fiscal_year,
                    "raw_file_path": str(local_path),
                    "content_hash": doc_meta.content_hash,
                    "metadata": {"num_pages": parsed_doc.num_pages, "num_chunks": len(chunks)},
                })

                # 6. Embed and Insert Chunks
                print(f"    -> Generating vector embeddings ({len(chunks)} chunks)...")
                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                for c, emb in zip(chunks, embeddings):
                    self.pg.insert_chunk(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        fiscal_year=c.fiscal_year,
                        page=c.page,
                        section=c.section,
                        chunk_type=c.chunk_type,
                        text=c.text,
                        embedding=emb,
                        metadata=c.metadata,
                    )

                existing_hashes.add(doc_meta.content_hash)
                results["processed"] += 1
                results["chunks_created"] += len(chunks)
                results["documents"].append(doc_meta.document_id)

        # 7. Enrich Neo4j Graph
        print("\n[GRAPH] Enriching Neo4j Knowledge Graph...")
        self.kg.build_full_graph()
        print("Knowledge Graph enriched.")

        print("\n" + "=" * 70)
        print(f"INGESTION COMPLETE: {results['processed']} new docs processed, {results['chunks_created']} chunks stored.")
        print("=" * 70)
        return results

    def ingest_from_chunks(self, chunks_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Directly embed and ingest pre-processed JSON chunk files into PostgreSQL."""
        c_dir = chunks_dir or (PROJECT_ROOT / "data" / "chunks")
        chunk_files = sorted(c_dir.glob("*_chunks.json"))

        results = {"files_processed": 0, "chunks_ingested": 0, "documents": []}
        print("=" * 70)
        print(f"INGESTING FROM PRE-CHUNKED FILES ({len(chunk_files)} files)")
        print(f"Embedding Model: {self.embedder.model_name} (Dim: {self.embedder.dimension})")
        print("=" * 70)

        for cf in chunk_files:
            try:
                with open(cf, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as e:
                logger.warning(f"Could not read chunk file {cf}: {e}")
                continue

            doc_id = payload.get("document_id", cf.stem.replace("_chunks", ""))
            chunks_list = payload.get("chunks", [])
            if not chunks_list:
                continue

            fy = chunks_list[0].get("fiscal_year", "FY2023-24")
            print(f"\n[CHUNKS] Embedding {doc_id} ({len(chunks_list)} chunks, {fy})...")

            # Insert document
            self.pg.insert_document({
                "document_id": doc_id,
                "company": "LTIMindtree Limited",
                "ticker": "LTIM",
                "source": "Pre-processed JSON Chunks",
                "source_url": "",
                "document_type": "annual_report",
                "title": doc_id,
                "reporting_period": fy,
                "raw_file_path": str(cf),
                "content_hash": doc_id,
                "metadata": {"num_chunks": len(chunks_list)},
            })

            # Embed texts in batches
            texts = [c.get("text", "") for c in chunks_list]
            embeddings = self.embedder.embed_texts(texts)

            for c, emb in zip(chunks_list, embeddings):
                self.pg.insert_chunk(
                    chunk_id=c.get("chunk_id", f"{doc_id}_{c.get('page', 1)}"),
                    document_id=doc_id,
                    fiscal_year=c.get("fiscal_year", fy),
                    page=c.get("page", 1),
                    section=c.get("section", ""),
                    chunk_type=c.get("chunk_type", "prose"),
                    text=c.get("text", ""),
                    embedding=emb,
                    metadata={"block_indices": c.get("block_indices", [])},
                )

            results["files_processed"] += 1
            results["chunks_ingested"] += len(chunks_list)
            results["documents"].append(doc_id)
            print(f"  -> Ingested {len(chunks_list)} chunks into PostgreSQL.")

        print("\n[GRAPH] Enriching Neo4j Knowledge Graph...")
        self.kg.build_full_graph()

        print("\n" + "=" * 70)
        print(f"CHUNKS INGESTION COMPLETE: {results['chunks_ingested']} total chunks stored.")
        print("=" * 70)
        return results


if __name__ == "__main__":
    pipeline = FinancialIngestionPipeline()
    pipeline.run_ingestion(max_docs=1)
