"""
DANANN — Backend Supabase pgvector.

Stockage persistant des connaissances dans Supabase avec l'extension pgvector.
Utilise une fonction RPC `match_chunks` cote base de donnees.

SQL a executer dans Supabase une fois (voir scripts/supabase_schema.sql) :

  CREATE EXTENSION IF NOT EXISTS vector;

  CREATE TABLE IF NOT EXISTS morrigan_chunks (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL,
    embedding  VECTOR(384),
    metadata   JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
  );

  CREATE INDEX IF NOT EXISTS morrigan_chunks_embedding_idx
    ON morrigan_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

  CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(384),
    match_count INT DEFAULT 5
  ) RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
  ) LANGUAGE SQL STABLE AS $$
    SELECT id, content, metadata,
           1 - (embedding <=> query_embedding) AS similarity
    FROM morrigan_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
  $$;
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("morrigan.danann.supabase")


class SupabaseVectorStore:
    """Backend Supabase pour Danann."""

    TABLE_NAME = "morrigan_chunks"
    MATCH_FUNCTION = "match_chunks"

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.client = None

    def connect(self) -> bool:
        """Etablit la connexion a Supabase."""
        if not self.url or not self.key:
            logger.warning("URL ou cle Supabase manquante")
            return False

        try:
            from supabase import create_client
            self.client = create_client(self.url, self.key)
            logger.info("Supabase connecte: %s", self.url)
            return True
        except Exception as e:
            logger.error("Erreur connexion Supabase: %s", e)
            return False

    def insert(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insere un chunk dans la base."""
        if self.client is None:
            return False

        try:
            self.client.table(self.TABLE_NAME).insert({
                "content": text,
                "embedding": embedding,
                "metadata": metadata or {},
            }).execute()
            return True
        except Exception as e:
            logger.error("Erreur insertion: %s", e)
            return False

    def insert_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Insere plusieurs chunks en une fois."""
        if self.client is None:
            return 0

        if metadatas is None:
            metadatas = [{} for _ in texts]

        rows = [
            {"content": text, "embedding": emb, "metadata": meta}
            for text, emb, meta in zip(texts, embeddings, metadatas)
        ]

        try:
            self.client.table(self.TABLE_NAME).insert(rows).execute()
            logger.info("Insere %d chunks dans Supabase", len(rows))
            return len(rows)
        except Exception as e:
            logger.error("Erreur insertion batch: %s", e)
            return 0

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Recherche les chunks les plus proches via RPC."""
        if self.client is None:
            return []

        try:
            response = self.client.rpc(
                self.MATCH_FUNCTION,
                {"query_embedding": query_embedding, "match_count": top_k},
            ).execute()

            if not response.data:
                return []

            return [
                (row["content"], float(row["similarity"]), row.get("metadata") or {})
                for row in response.data
            ]
        except Exception as e:
            logger.error("Erreur recherche: %s", e)
            return []

    def count(self) -> int:
        """Nombre de chunks en base."""
        if self.client is None:
            return 0
        try:
            response = (
                self.client.table(self.TABLE_NAME)
                .select("id", count="exact")
                .execute()
            )
            return response.count or 0
        except Exception as e:
            logger.error("Erreur count: %s", e)
            return 0
