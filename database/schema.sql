-- ============================================================
-- Financial GraphRAG: Core Relational & Vector Schema
-- PostgreSQL Database Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 1. DOCUMENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,

    company TEXT NOT NULL,
    ticker TEXT NOT NULL,

    source TEXT NOT NULL,
    source_url TEXT NOT NULL,

    document_type TEXT NOT NULL,
    title TEXT,

    publication_date DATE,
    reporting_period TEXT,

    retrieval_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    raw_file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,

    content TEXT,

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 2. CHUNKS
-- ============================================================

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,

    text TEXT NOT NULL,

    section TEXT,
    page INTEGER,

    company TEXT NOT NULL,
    ticker TEXT NOT NULL,

    document_type TEXT,
    reporting_period TEXT,

    source TEXT,
    source_url TEXT,

    embedding vector(768),

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(document_id, chunk_index)
);


-- ============================================================
-- 2b. VECTOR CHUNKS (Optimized for Fast HNSW Cosine Distance)
-- ============================================================

CREATE TABLE IF NOT EXISTS vector_chunks (
    vector_id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    fiscal_year TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    chunk_type TEXT,
    text TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_hnsw
    ON vector_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_fy
    ON vector_chunks (fiscal_year);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc
    ON vector_chunks (document_id);


-- ============================================================
-- 3. FINANCIAL FACTS
-- ============================================================

CREATE TABLE IF NOT EXISTS financial_facts (
    fact_id TEXT PRIMARY KEY,

    company TEXT NOT NULL,
    ticker TEXT NOT NULL,

    metric TEXT NOT NULL,

    value_numeric DOUBLE PRECISION,
    value_text TEXT,

    unit TEXT,

    period TEXT,

    fact_type TEXT NOT NULL,

    source_document_id TEXT NOT NULL
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    source_chunk_id TEXT
        REFERENCES chunks(chunk_id)
        ON DELETE SET NULL,

    source_text TEXT,

    confidence DOUBLE PRECISION,

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 4. ENTITIES
-- ============================================================

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,

    entity_type TEXT NOT NULL,

    canonical_name TEXT NOT NULL,

    ticker TEXT,

    description TEXT,

    aliases JSONB DEFAULT '[]'::jsonb,

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(entity_type, canonical_name)
);


-- ============================================================
-- 5. ENTITY MENTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id TEXT PRIMARY KEY,

    entity_id TEXT NOT NULL
        REFERENCES entities(entity_id)
        ON DELETE CASCADE,

    document_id TEXT
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    chunk_id TEXT
        REFERENCES chunks(chunk_id)
        ON DELETE CASCADE,

    mention_text TEXT NOT NULL,

    start_position INTEGER,
    end_position INTEGER,

    confidence DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 6. RELATIONSHIPS
-- ============================================================

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,

    source_entity_id TEXT NOT NULL
        REFERENCES entities(entity_id)
        ON DELETE CASCADE,

    target_entity_id TEXT NOT NULL
        REFERENCES entities(entity_id)
        ON DELETE CASCADE,

    relationship_type TEXT NOT NULL,

    description TEXT,

    document_id TEXT
        REFERENCES documents(document_id)
        ON DELETE SET NULL,

    chunk_id TEXT
        REFERENCES chunks(chunk_id)
        ON DELETE SET NULL,

    confidence DOUBLE PRECISION,

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 7. INGESTION RUNS
-- ============================================================

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,

    source TEXT NOT NULL,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    completed_at TIMESTAMPTZ,

    status TEXT NOT NULL,

    documents_discovered INTEGER DEFAULT 0,

    documents_downloaded INTEGER DEFAULT 0,

    documents_processed INTEGER DEFAULT 0,

    documents_failed INTEGER DEFAULT 0,

    error_message TEXT,

    metadata JSONB DEFAULT '{}'::jsonb
);


-- ============================================================
-- 8. MARKET DATA
-- ============================================================

CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,

    ticker TEXT NOT NULL,

    timestamp TIMESTAMPTZ NOT NULL,

    price DOUBLE PRECISION,

    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,

    volume BIGINT,

    market_cap DOUBLE PRECISION,

    source TEXT NOT NULL,

    metadata JSONB DEFAULT '{}'::jsonb,

    UNIQUE(ticker, timestamp, source)
);


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_documents_company
    ON documents(company);

CREATE INDEX IF NOT EXISTS idx_documents_ticker
    ON documents(ticker);

CREATE INDEX IF NOT EXISTS idx_documents_type
    ON documents(document_type);

CREATE INDEX IF NOT EXISTS idx_documents_period
    ON documents(reporting_period);

CREATE INDEX IF NOT EXISTS idx_documents_hash
    ON documents(content_hash);


CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_company
    ON chunks(company);

CREATE INDEX IF NOT EXISTS idx_chunks_period
    ON chunks(reporting_period);


CREATE INDEX IF NOT EXISTS idx_facts_company
    ON financial_facts(company);

CREATE INDEX IF NOT EXISTS idx_facts_metric
    ON financial_facts(metric);

CREATE INDEX IF NOT EXISTS idx_facts_period
    ON financial_facts(period);

CREATE INDEX IF NOT EXISTS idx_facts_type
    ON financial_facts(fact_type);


CREATE INDEX IF NOT EXISTS idx_entities_type
    ON entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_entities_name
    ON entities(canonical_name);


CREATE INDEX IF NOT EXISTS idx_relationships_source
    ON relationships(source_entity_id);

CREATE INDEX IF NOT EXISTS idx_relationships_target
    ON relationships(target_entity_id);

CREATE INDEX IF NOT EXISTS idx_relationships_type
    ON relationships(relationship_type);


CREATE INDEX IF NOT EXISTS idx_market_ticker
    ON market_data(ticker);

CREATE INDEX IF NOT EXISTS idx_market_timestamp
    ON market_data(timestamp);