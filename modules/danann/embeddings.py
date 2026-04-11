"""
DANANN — Encodage d'embeddings local.

Utilise sentence-transformers pour générer des vecteurs
sans dépendance à une API externe.
"""

import logging
from typing import List

logger = logging.getLogger("morrigan.danann.embeddings")


class EmbeddingEngine:
    """
    Moteur d'embeddings local.

    Modèle par défaut : all-MiniLM-L6-v2 (384 dims, ~80MB, CPU).
    Alternatives : BGE-small-en-v1.5, nomic-embed-text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        logger.info("EmbeddingEngine créé (modèle: %s, non chargé)", model_name)

    def load(self) -> None:
        """Charge le modèle d'embeddings en mémoire."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modèle '%s' chargé", self.model_name)
        except Exception as e:
            logger.error("Erreur chargement modèle: %s", e)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode une liste de textes en vecteurs."""
        if self.model is None:
            logger.warning("Modèle non chargé, appel de load()")
            self.load()

        if self.model is None:
            logger.error("Impossible de charger le modèle d'embeddings")
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
