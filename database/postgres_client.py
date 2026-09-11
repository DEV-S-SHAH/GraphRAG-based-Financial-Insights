"""
PostgreSQL + pgvector Client for Financial GraphRAG.

Manages:
- Documents and metadata
- Chunks and pgvector 768-dim embeddings
- Cosine similarity search with metadata filtering
- Table inspection and stats
"""

from contextlib import contextmanager
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import psycopg
from dotenv import load_dotenv

try:
    from psycopg_pool import ConnectionPool
    _HAS_POOL = True
except ImportError:
    _HAS_POOL = False

load_dotenv()
logger = logging.getLogger(__name__)


class PostgresVectorClient:
    """PostgreSQL client utilizing pgvector for similarity search with connection pooling."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        raw_host = host or os.getenv("POSTGRES_HOST", "127.0.0.1")
        # Normalize localhost to 127.0.0.1 to avoid Windows IPv6 (::1) port forwarding collisions
        self.host = "127.0.0.1" if raw_host.lower() == "localhost" else raw_host
        self.port = int(port or os.getenv("POSTGRES_PORT", 5432))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "financial_db")
        self.user = user or os.getenv("POSTGRES_USER", "financial_user")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "password123")
        self.dimension = int(dimension or os.getenv("EMBEDDING_DIM", "768"))
        self.current_dimension = self.dimension
        self.pool_min_size = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))
        self._pool = None

        if _HAS_POOL:
            try:
                conninfo = (
                    f"host={self.host} port={self.port} dbname={self.dbname} "
                    f"user={self.user} password={self.password} connect_timeout=10"
                )
                self._pool = ConnectionPool(
                    conninfo,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    open=True,
                    kwargs={"autocommit": True},
                )
            except Exception as e:
                logger.warning(f"Could not initialize connection pool: {e}. Using direct connections.")

        self._ensure_tables_and_indexes()

    def _create_direct_connection(self):
        try:
            return psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                autocommit=True,
                connect_timeout=10,
            )
        except Exception as primary_exc:
            alt_host = "localhost" if self.host == "127.0.0.1" else "127.0.0.1"
            try:
                return psycopg.connect(
                    host=alt_host,
                    port=self.port,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password,
                    autocommit=True,
                    connect_timeout=5,
                )
            except Exception:
                raise primary_exc

    @contextmanager
    def get_connection(self):
        """Yield a PostgreSQL connection from the connection pool or a fallback direct connection."""
        if self._pool is not None:
            try:
                with self._pool.connection() as conn:
                    yield conn
                    return
            except Exception as e:
                logger.debug(f"Pool checkout notice: {e}; using direct connection fallback.")
        conn = self._create_direct_connection()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def close(self):
        """Close connection pool and release resources."""
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception as e:
                logger.debug(f"Error closing connection pool: {e}")
            self._pool = None

    def _ensure_tables_and_indexes(self):
        """Ensure pgvector extension, vector_chunks table, and HNSW index exist."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS vector_chunks (
                            vector_id BIGSERIAL PRIMARY KEY,
                            chunk_id TEXT UNIQUE NOT NULL,
                            document_id TEXT NOT NULL,
                            fiscal_year TEXT NOT NULL,
                            page INTEGER,
                            section TEXT,
                            chunk_type TEXT,
                            text TEXT NOT NULL,
                            embedding vector({self.dimension}),
                            metadata JSONB DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        """
                    )
                    # Detect existing table vector dimension
                    try:
                        cur.execute(
                            """
                            SELECT atttypmod FROM pg_attribute 
                            WHERE attrelid = 'vector_chunks'::regclass AND attname = 'embedding';
                            """
                        )
                        row = cur.fetchone()
                        if row and row[0] > 0:
                            existing_dim = row[0]
                            if existing_dim != self.dimension:
                                cur.execute("SELECT count(*) FROM vector_chunks;")
                                cnt = cur.fetchone()[0]
                                if cnt == 0:
                                    cur.execute("DROP INDEX IF EXISTS idx_vector_chunks_hnsw;")
                                    cur.execute(f"ALTER TABLE vector_chunks ALTER COLUMN embedding TYPE vector({self.dimension});")
                                    self.current_dimension = self.dimension
                                else:
                                    self.current_dimension = existing_dim
                            else:
                                self.current_dimension = self.dimension
                        else:
                            self.current_dimension = self.dimension
                    except Exception:
                        self.current_dimension = self.dimension

                    # Create HNSW index for fast cosine distance search
                    try:
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_vector_chunks_hnsw
                            ON vector_chunks USING hnsw (embedding vector_cosine_ops);
                            """
                        )
                    except Exception as e:
                        logger.info(f"HNSW index note: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize PostgreSQL tables: {e}")

    def insert_document(self, doc_data: Dict[str, Any]):
        """Insert or update a document in PostgreSQL."""
        query = """
        INSERT INTO documents (
            document_id, company, ticker, source, source_url,
            document_type, title, reporting_period, raw_file_path,
            content_hash, metadata
        ) VALUES (
            %(document_id)s, %(company)s, %(ticker)s, %(source)s, %(source_url)s,
            %(document_type)s, %(title)s, %(reporting_period)s, %(raw_file_path)s,
            %(content_hash)s, %(metadata)s
        )
        ON CONFLICT (document_id) DO UPDATE SET
            updated_at = NOW(),
            metadata = EXCLUDED.metadata;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "document_id": doc_data["document_id"],
                        "company": doc_data.get("company", "LTIMindtree Limited"),
                        "ticker": doc_data.get("ticker", "LTIM"),
                        "source": doc_data.get("source", "Annual Report"),
                        "source_url": doc_data.get("source_url", ""),
                        "document_type": doc_data.get("document_type", "annual_report"),
                        "title": doc_data.get("title", doc_data["document_id"]),
                        "reporting_period": doc_data.get("reporting_period", "FY2025-26"),
                        "raw_file_path": doc_data.get("raw_file_path", ""),
                        "content_hash": doc_data.get("content_hash", doc_data["document_id"]),
                        "metadata": json.dumps(doc_data.get("metadata", {})),
                    },
                )

    def insert_chunk(
        self,
        chunk_id: str,
        document_id: str,
        fiscal_year: str,
        page: int,
        section: str,
        chunk_type: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Insert or update a single chunk with its vector embedding."""
        if len(embedding) != self.current_dimension:
            logger.warning(
                f"Embedding vector dimension ({len(embedding)}) differs from table dimension ({self.current_dimension}). Adapting vector."
            )
            if len(embedding) > self.current_dimension:
                embedding = embedding[: self.current_dimension]
            else:
                embedding = embedding + [0.0] * (self.current_dimension - len(embedding))

        query = """
        INSERT INTO vector_chunks (
            chunk_id, document_id, fiscal_year, page, section,
            chunk_type, text, embedding, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        chunk_id,
                        document_id,
                        fiscal_year,
                        page,
                        section,
                        chunk_type,
                        text,
                        embedding,
                        json.dumps(metadata or {}),
                    ),
                )

    def search_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        fiscal_year: Optional[str] = None,
        section: Optional[str] = None,
        chunk_type: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search chunks using pgvector cosine distance `<=>`.
        Returns similarity score = 1 - cosine_distance.
        """
        # Adapt query dimension to match table column dimension if needed
        if len(query_embedding) != self.current_dimension:
            logger.warning(
                f"Query embedding dimension ({len(query_embedding)}) differs from table dimension ({self.current_dimension}). "
                "Adapting vector. Ensure EMBEDDING_MODEL and EMBEDDING_DIM match indexed data."
            )
            if len(query_embedding) > self.current_dimension:
                query_embedding = query_embedding[: self.current_dimension]
            else:
                query_embedding = query_embedding + [0.0] * (self.current_dimension - len(query_embedding))

        where_clauses = ["embedding IS NOT NULL"]
        where_params: List[Any] = []

        if fiscal_year:
            where_clauses.append("fiscal_year = %s")
            where_params.append(fiscal_year)

        if section:
            where_clauses.append("LOWER(section) LIKE %s")
            where_params.append(f"%{section.lower()}%")

        if chunk_type:
            where_clauses.append("chunk_type = %s")
            where_params.append(chunk_type)

        where_sql = " AND ".join(where_clauses)
        params = [query_embedding] + where_params + [query_embedding, top_k]

        sql = f"""
        SELECT
            chunk_id,
            document_id,
            fiscal_year,
            page,
            section,
            chunk_type,
            text,
            metadata,
            (1 - (embedding <=> %s::vector)) AS similarity
        FROM vector_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        results = []
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    sim = float(row[8])
                    if sim >= min_similarity:
                        results.append(
                            {
                                "chunk_id": row[0],
                                "document_id": row[1],
                                "fiscal_year": row[2],
                                "page": row[3],
                                "section": row[4],
                                "chunk_type": row[5],
                                "text": row[6],
                                "metadata": row[7],
                                "score": round(sim, 4),
                            }
                        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics from PostgreSQL."""
        stats = {}
        tables = ["documents", "vector_chunks", "chunks", "financial_facts", "entities", "relationships"]
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for t in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {t}")
                        stats[t] = cur.fetchone()[0]
                    except Exception:
                        stats[t] = 0

                # Chunks by fiscal year
                try:
                    cur.execute("SELECT fiscal_year, COUNT(*) FROM vector_chunks GROUP BY fiscal_year ORDER BY fiscal_year")
                    stats["chunks_by_year"] = {row[0]: row[1] for row in cur.fetchall()}
                except Exception:
                    stats["chunks_by_year"] = {}

        return stats

    def execute_query(self, query: str, max_rows: int = 50) -> Tuple[List[str], List[Tuple]]:
        """Safe read-only query execution for the UI SQL inspector."""
        query_strip = query.strip().upper()
        if not (query_strip.startswith("SELECT") or query_strip.startswith("EXPLAIN")):
            raise ValueError("Only SELECT or EXPLAIN queries are permitted in the inspector.")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchmany(max_rows)
                return cols, rows
