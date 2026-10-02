-- Schema for the Support Ticket Resolution Assistant (IMPLEMENTATION_PLAN.md §15).
-- Deviations from the plan (documented in ARCHITECTURE.md):
--   * tickets.complaint_embedding: complaint-only vector used by the k-NN intent
--     classifier (the retrieval vector embeds complaint + resolution).
--   * tickets.split: provenance split assignment stored in-row.
--   * kb_article_versions: keeps superseded KB versions when an article is updated.
--   * ANN indexes are created by scripts/init_db.py AFTER ingestion so that
--     ivfflat `lists` can be sized from the real row count (§15.2).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    complaint TEXT NOT NULL,
    resolution TEXT,
    category VARCHAR(100),
    product VARCHAR(100),
    severity VARCHAR(20),
    sentiment VARCHAR(20),
    sentiment_score FLOAT,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    source VARCHAR(50),
    label_source VARCHAR(50),
    verified BOOLEAN DEFAULT FALSE,
    split VARCHAR(30) DEFAULT 'retrieval_corpus',
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384),
    complaint_embedding vector(384),
    text_search tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(complaint, '') || ' ' || coalesce(resolution, ''))
    ) STORED,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS tickets_text_search_idx ON tickets USING GIN(text_search);
CREATE INDEX IF NOT EXISTS tickets_category_idx ON tickets(category);
CREATE INDEX IF NOT EXISTS tickets_product_idx ON tickets(product);
CREATE INDEX IF NOT EXISTS tickets_severity_idx ON tickets(severity);
CREATE INDEX IF NOT EXISTS tickets_source_idx ON tickets(source);
CREATE INDEX IF NOT EXISTS tickets_created_at_idx ON tickets(created_at);

CREATE TABLE IF NOT EXISTS kb_articles (
    article_id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    product VARCHAR(100),
    tags TEXT[],
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384),
    text_search tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS kb_text_search_idx ON kb_articles USING GIN(text_search);
CREATE INDEX IF NOT EXISTS kb_category_idx ON kb_articles(category);
CREATE INDEX IF NOT EXISTS kb_product_idx ON kb_articles(product);
CREATE INDEX IF NOT EXISTS kb_updated_at_idx ON kb_articles(updated_at);

CREATE TABLE IF NOT EXISTS kb_article_versions (
    article_id VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    archived_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (article_id, version)
);

CREATE TABLE IF NOT EXISTS intent_taxonomy (
    intent_id SERIAL PRIMARY KEY,
    intent_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    examples JSONB DEFAULT '[]'::jsonb,
    parent_intent_id INTEGER REFERENCES intent_taxonomy(intent_id),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS product_taxonomy (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50),
    aliases TEXT[],
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS resolution_requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complaint_hash VARCHAR(64) NOT NULL,
    complaint_redacted TEXT,
    extracted_metadata JSONB,
    retrieved_sources JSONB,
    retrieval_scores FLOAT[],
    generated_resolution JSONB,
    citations JSONB,
    citation_accuracy FLOAT,
    citation_coverage FLOAT,
    groundedness_score FLOAT,
    validation_issues JSONB,
    heuristic_confidence FLOAT,
    confidence_components JSONB,
    decision VARCHAR(20),
    decision_reason TEXT,
    flags JSONB,
    generator VARCHAR(50),
    latency_ms INTEGER,
    stage_latency_ms JSONB,
    feedback_rating INTEGER,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS resolution_requests_created_at_idx ON resolution_requests(created_at);
CREATE INDEX IF NOT EXISTS resolution_requests_decision_idx ON resolution_requests(decision);
CREATE INDEX IF NOT EXISTS resolution_requests_confidence_idx ON resolution_requests(heuristic_confidence);
