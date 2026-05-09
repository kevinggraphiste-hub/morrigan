"""
DANANN — Reranker cross-encoder.

Prend les top-K candidats du retrieval cosine et les re-classe
avec un modele cross-encoder plus precis (mais plus lent).

Phase 2 : cross-encoder/ms-marco-MiniLM-L-6-v2 (~22 Mo, CPU)
   - Input : paire (query, chunk_text)
   - Output : score de pertinence [-inf, +inf] (plus haut = plus pertinent)
   - Latence : ~5-15 ms par paire sur CPU i5

Le cross-encoder est BEAUCOUP plus precis que le cosine similarity
car il voit la query ET le chunk ensemble (attention croisee),
alors que le bi-encoder les encode separement.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("morrigan.danann.reranker")


class CrossEncoderReranker:
    """
    Reranker cross-encoder pour Danann.

    Lazy-load : le modele n'est charge qu'au premier appel.
    """

    # Modele par defaut : ms-marco-MiniLM-L-6-v2
    # - Entrainement : MS MARCO passage ranking
    # - Taille : ~22 Mo
    # - Latence : ~5-15 ms/paire CPU
    # - Qualite : NDCG@10 ~0.39 sur MS MARCO (excellent pour sa taille)
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        logger.info(
            "CrossEncoderReranker cree (modele: %s, non charge)",
            self.model_name,
        )

    def load(self) -> bool:
        """Charge le modele cross-encoder en memoire."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker '%s' charge", self.model_name)
            return True
        except ImportError:
            logger.error(
                "sentence-transformers non installe — reranker desactive"
            )
            return False
        except Exception as e:
            logger.error("Erreur chargement reranker: %s", e)
            return False

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Re-classe les candidats par pertinence cross-encoder.

        Args:
            query: la requete utilisateur
            candidates: liste de (text, score_cosine, metadata) depuis Danann
            top_k: nombre de resultats a retourner (defaut: tous)

        Returns:
            liste triee par score cross-encoder decroissant,
            avec le score original preserve dans metadata["score_cosine"]
        """
        if not candidates:
            return []

        # Lazy load
        if self.model is None:
            if not self.load():
                logger.warning(
                    "Reranker indisponible — retour des candidats tels quels"
                )
                return candidates

        # Construire les paires (query, chunk_text) pour le cross-encoder
        pairs = [(query, text) for text, _, _ in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error("Erreur prediction reranker: %s", e)
            return candidates

        # Recombiner avec les scores cross-encoder
        reranked = []
        for i, (text, cosine_score, meta) in enumerate(candidates):
            enriched_meta = {
                **meta,
                "score_cosine": cosine_score,
                "score_reranker": float(scores[i]),
            }
            reranked.append((text, float(scores[i]), enriched_meta))

        # Trier par score cross-encoder decroissant
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        logger.info(
            "Reranker: %d candidats re-classes, top score=%.3f",
            len(reranked),
            reranked[0][1] if reranked else 0.0,
        )

        return reranked
