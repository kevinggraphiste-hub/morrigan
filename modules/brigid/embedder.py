"""
BRIGID — Encodeur sémantique des queries.

Wrapper minimal autour de `sentence-transformers/all-MiniLM-L6-v2` (déjà
utilisé par Danann → cache HF partagé, pas de modèle supplémentaire à
télécharger). Produit des embeddings 384-D normalisés.

Singleton : on évite de recharger le modèle (≈ 80 Mo) à chaque appel.
La 1re instanciation peut prendre 1-3 s (chargement) ; les suivantes
sont quasi-instantanées.

Cohérence checkpoint ↔ inférence : tant que le nom de modèle reste
`EMBED_MODEL_NAME`, train et inference produisent les mêmes embeddings
et le checkpoint reste valable. Changer ce nom invalide les
checkpoints existants — bumper la version du modèle dans
`brigid_cfc.pt` si on le fait.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger("morrigan.brigid.embedder")

# Modèle d'embedding partagé avec Danann. NE PAS changer sans bump de
# version du checkpoint Brigid (incompatibilité de représentation).
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


class IntentEmbedder:
    """Encodeur singleton pour les queries d'intent classification."""

    _instance: "IntentEmbedder | None" = None
    _model = None  # type: ignore[var-annotated]

    def __new__(cls) -> "IntentEmbedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Charge le modèle au 1er usage (lazy)."""
        if self._model is not None:
            return
        # Import retardé : sentence-transformers tire torch (gros) ; on
        # n'a pas envie de payer ça à l'import du module.
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Chargement de l'embedder %s", EMBED_MODEL_NAME)
        # Force CPU : cohérent avec la philo Morrigan ("tourne sur PC
        # modeste"), et évite les CUDA errors quand torch détecte un
        # GPU sans kernels compatibles (machines de dev hétérogènes).
        IntentEmbedder._model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    def encode(self, texts: Sequence[str]):
        """Encode une liste de textes en un tensor (N, EMBED_DIM).

        Renvoie un `torch.Tensor` float32. Embeddings normalisés
        (norme L2 = 1) pour stabilité du gradient en aval.
        """
        if not texts:
            raise ValueError("encode() reçue avec une liste vide")

        self._ensure_loaded()
        import torch  # noqa: PLC0415

        # convert_to_tensor=True → renvoie déjà un torch.Tensor.
        # normalize_embeddings=True → norme L2 = 1 par vecteur.
        embeddings = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # sentence-transformers peut renvoyer sur GPU si dispo — on
        # force CPU ici, c'est notre cible (LNN compact sur CPU).
        return embeddings.to(torch.device("cpu")).float()

    def encode_one(self, text: str):
        """Raccourci pour un seul texte → tensor (EMBED_DIM,)."""
        return self.encode([text])[0]


def get_embedder() -> IntentEmbedder:
    """Accès au singleton."""
    return IntentEmbedder()
