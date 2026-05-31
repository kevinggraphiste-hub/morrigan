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
            # Force CPU : cohérent avec la philo Morrigan ("tourne sur PC
            # modeste"), et évite les CUDA errors quand torch détecte un
            # GPU sans kernels compatibles (machines de dev hétérogènes).
            # Même approche que modules/brigid/embedder.py.
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Modèle '%s' chargé (device=cpu)", self.model_name)
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

        # normalize_embeddings=True → norme L2 = 1 par vecteur. Tout le
        # module Danann (store, quantization int8/binary, ann IVF) suppose
        # cette normalisation : le produit scalaire vaut alors le cosinus.
        # Aligne aussi Danann sur modules/brigid/embedder.py (même modèle
        # MiniLM partagé, qui normalise déjà).
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.tolist()
