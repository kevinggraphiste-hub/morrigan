"""
DANANN — Reranker cross-encoder.

Prend les top-K candidats du retrieval cosine et les re-classe
avec un modele cross-encoder plus precis (mais plus lent).

Phase 2D : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingue, CPU)
   - Input : paire (query, chunk_text)
   - Output : score de pertinence [-inf, +inf] (plus haut = plus pertinent)
   - Gain MESURÉ sur requêtes FR (jeu scripts/eval_rag.py, pool 16,
     troncature 1000) : hit@3 48/56 vs 43/56 sans reranker, pour
     ~1.5 s/req sur le CPU dev. Les configs allégées perdent le gain
     (pool 8 + cut 500 → 44/56, dans le bruit) : ne pas re-trimmer sans
     re-mesurer via eval_rag.py.

Le cross-encoder voit la query ET le chunk ensemble (attention croisée),
là où le bi-encoder les encode séparément. Historique : l'audit
2026-06-12 ne mesurait « aucun gain fiable » — cause racine identifiée
en 2D = modèle ms-marco **anglais** + pool k*3=9 + troncature 512, soit
précisément la combinaison qui ne gagne rien.

Post-audit : `device="cpu"` par défaut — sans device explicite,
sentence-transformers choisit CUDA si disponible, ce qui plantait
silencieusement sur GPU non supporté (les candidats repartaient
non re-classés).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("morrigan.danann.reranker")


class CrossEncoderReranker:
    """
    Reranker cross-encoder pour Danann.

    Lazy-load : le modele n'est charge qu'au premier appel.
    """

    # Modele par defaut : mmarco-mMiniLMv2-L12-H384-v1
    # - Entrainement : mMARCO passage ranking (multilingue, dont FR)
    # - Taille : ~135 Mo (33M params)
    # - Seul des deux candidats testés à apporter un gain FR mesuré
    #   (cf. docstring module) ; ms-marco-MiniLM-L-6-v2 (anglais) reste
    #   passable via model_name pour un corpus 100% EN.
    DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_passage_chars: Optional[int] = 1000,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        # Troncature des passages envoyés au cross-encoder. Coût quasi
        # linéaire en longueur de texte, mais descendre sous 1000 chars
        # mange le gain qualité (mesuré : cut 500 → 44/56 vs 46-48/56,
        # cf. docstring module). None = passages complets.
        self.max_passage_chars = max_passage_chars
        self.model = None
        logger.info(
            "CrossEncoderReranker cree (modele: %s, device=%s, non charge)",
            self.model_name, self.device,
        )

    def load(self) -> bool:
        """Charge le modele cross-encoder en memoire."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
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

        # Construire les paires (query, chunk_text) pour le cross-encoder,
        # passages tronqués (cf. max_passage_chars).
        cut = self.max_passage_chars
        pairs = [(query, text[:cut] if cut else text) for text, _, _ in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error("Erreur prediction reranker: %s", e)
            return candidates

        # Recombiner avec les scores cross-encoder. Si Danann a déjà posé
        # le cosinus PUR dans la meta (cf. store._candidates_from), on le
        # préserve — `cosine_score` ici est le score boosté lexicalement.
        reranked = []
        for i, (text, cosine_score, meta) in enumerate(candidates):
            enriched_meta = {
                **meta,
                "score_cosine": meta.get("score_cosine", cosine_score),
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
