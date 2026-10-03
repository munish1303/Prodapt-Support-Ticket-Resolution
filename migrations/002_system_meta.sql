-- Key/value facts about how the data was built, e.g. which embedding model produced the stored vectors.
-- scripts/reembed.py compares system_meta.embedding_model with settings.EMBEDDING_MODEL on startup and, when they
-- differ, resizes the vector columns (if the dimension changed) and re-embeds the corpus.
CREATE TABLE IF NOT EXISTS system_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
