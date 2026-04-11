-- ============================================================
-- MORRIGAN — Schema Supabase pour Danann (pgvector)
-- A executer une fois dans l'editeur SQL de Supabase.
-- ============================================================

-- 1. Extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table des chunks de connaissance
CREATE TABLE IF NOT EXISTS morrigan_chunks (
  id         BIGSERIAL PRIMARY KEY,
  content    TEXT NOT NULL,
  embedding  VECTOR(384),
  metadata   JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Index de recherche vectorielle (cosinus)
CREATE INDEX IF NOT EXISTS morrigan_chunks_embedding_idx
  ON morrigan_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- 4. Fonction RPC pour la recherche
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding VECTOR(384),
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
  SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) AS similarity
  FROM morrigan_chunks
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- 5. (Optionnel) Index sur metadata pour filtrer par collection
CREATE INDEX IF NOT EXISTS morrigan_chunks_metadata_idx
  ON morrigan_chunks USING gin (metadata);
