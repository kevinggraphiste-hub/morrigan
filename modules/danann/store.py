"""
DANANN — Interface de memoire vectorielle.

Gere le stockage et la recherche de connaissances.
Backends supportes :
- "memory" : store en RAM (Phase 0, dev)
- "supabase" : Supabase pgvector (Phase 1+, persistant)
"""

import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.types import ModuleInput, ModuleOutput, MorriganModule
from modules.danann.embeddings import EmbeddingEngine
from modules.danann.supabase_backend import SupabaseVectorStore

logger = logging.getLogger("morrigan.danann")

# Stopwords FR/EN pour le boost lexical — mots trop courants pour discriminer
_STOPWORDS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "est",
    "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "ce", "ces",
    "cette", "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
    "au", "aux", "en", "dans", "sur", "par", "pour", "avec", "sans",
    "a", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "pas", "ne", "ni", "si", "plus", "moins", "tres", "peu", "bien",
    "tout", "tous", "toute", "toutes", "comme", "ainsi", "alors", "mais",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were",
    "qu", "c", "n", "l", "d", "s", "t", "m",
    "entre", "chez", "vers", "sous", "deux", "trois",
    "explique", "compare", "difference", "qu'est-ce", "quelle",
}


def _tokenize(text: str) -> Set[str]:
    """Lowercase + strip accents + split en tokens significatifs."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", no_accents)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


class Danann(MorriganModule):
    """
    Memoire vectorielle de Morrigan.

    Backends :
    - memory   : store en RAM avec embeddings locaux
    - supabase : Supabase pgvector pour la persistence
    """

    def __init__(
        self,
        backend: str = "memory",
        supabase_url: str = "",
        supabase_key: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self.backend = backend
        self.top_k = top_k

        # Store en memoire (toujours disponible en fallback)
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

        # Moteur d'embeddings (lazy load)
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)

        # Backend Supabase (lazy init)
        self.supabase: Optional[SupabaseVectorStore] = None
        if backend == "supabase":
            url = supabase_url or os.getenv("SUPABASE_URL", "")
            key = supabase_key or os.getenv("SUPABASE_KEY", "")
            self.supabase = SupabaseVectorStore(url, key)
            if not self.supabase.connect():
                logger.warning("Bascule en mode memoire (Supabase indisponible)")
                self.backend = "memory"
                self.supabase = None

        logger.info("Danann initialisee (backend=%s)", self.backend)

    def _ensure_embeddings_loaded(self) -> None:
        """Charge le modele d'embeddings si pas encore fait."""
        if self.embedding_engine.model is None:
            self.embedding_engine.load()

    def index(
        self, texts: List[str], metadata: Optional[List[Dict]] = None
    ) -> int:
        """
        Indexe une liste de textes.

        Retourne le nombre de chunks effectivement indexes.
        """
        if not texts:
            return 0

        self._ensure_embeddings_loaded()
        new_embeddings = self.embedding_engine.encode(texts)

        if metadata is None:
            metadata = [{} for _ in texts]

        # Backend Supabase
        if self.backend == "supabase" and self.supabase:
            inserted = self.supabase.insert_batch(
                texts, new_embeddings, metadata
            )
            return inserted

        # Backend memoire
        new_embeddings_arr = np.array(new_embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings_arr
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_arr])

        self.chunks.extend(texts)
        self.metadata.extend(metadata)

        logger.info(
            "Danann [memory] — %d chunks indexes (total: %d)",
            len(texts),
            len(self.chunks),
        )
        return len(texts)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Recherche les chunks les plus proches."""
        k = top_k or self.top_k
        self._ensure_embeddings_loaded()
        query_emb = self.embedding_engine.encode([query])[0]

        # Backend Supabase
        if self.backend == "supabase" and self.supabase:
            return self.supabase.search(query_emb, top_k=k)

        # Backend memoire
        if not self.chunks or self.embeddings is None:
            return []

        query_vec = np.array(query_emb)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        scores = np.dot(self.embeddings, query_vec) / (norms + 1e-10)

        # Boost lexical : +0.05 par token rare de la query present dans le chunk
        # (plafonne a +0.20). Corrige les cas ou deux chunks ont un score cosine
        # proche mais un seul mentionne explicitement le sujet.
        query_tokens = _tokenize(query)
        if query_tokens:
            lexical_boost = np.zeros(len(self.chunks), dtype=np.float32)
            for i, chunk in enumerate(self.chunks):
                chunk_tokens = _tokenize(chunk)
                overlap = len(query_tokens & chunk_tokens)
                lexical_boost[i] = min(0.30, 0.08 * overlap)
            scores = scores + lexical_boost

        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[i], float(scores[i]), self.metadata[i])
            for i in top_indices
        ]

    def count(self) -> int:
        """Nombre de chunks indexes."""
        if self.backend == "supabase" and self.supabase:
            return self.supabase.count()
        return len(self.chunks)

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Recherche dans la memoire vectorielle."""
        logger.info("Danann cherche: %s", input.query[:60])

        results = self.search(input.query)

        if not results:
            return ModuleOutput(
                result={"chunks": [], "query": input.query},
                confidence=0.0,
                metadata={
                    "backend": self.backend,
                    "note": "Memoire vide — indexez des connaissances",
                },
            )

        chunks_data = [
            {"text": text, "score": score, "metadata": meta}
            for text, score, meta in results
        ]

        top_score = results[0][1] if results else 0.0

        return ModuleOutput(
            result={"chunks": chunks_data, "query": input.query},
            confidence=float(top_score),
            metadata={
                "backend": self.backend,
                "top_k": len(results),
                "total_indexed": self.count(),
            },
        )

    async def health_check(self) -> bool:
        if self.backend == "supabase" and self.supabase:
            return self.supabase.client is not None
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Danann",
            "type": "vector_memory",
            "backend": self.backend,
            "capabilities": [
                "semantic_search",
                "knowledge_retrieval",
                "fact_storage",
            ],
            "indexed_chunks": self.count(),
        }
