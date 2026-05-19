"""
SCATHACH — Module langage : generation de texte.

Phase 1 : Templates Jinja2 avec variantes + assemblage intelligent
des resultats des autres modules.
Phase 2 : Backend RWKV (0.19B-1.5B) pour une vraie generation.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.types import ModuleInput, ModuleOutput, MorriganModule

logger = logging.getLogger("morrigan.scathach")

# Stopwords minimal pour detecter les tokens "rares" de la query
_STOPWORDS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "est",
    "que", "qui", "quoi", "quel", "quelle", "ce", "ces", "au", "aux", "en",
    "dans", "sur", "par", "pour", "avec", "sans", "il", "elle", "je", "tu",
    "connais", "sais", "tu", "me", "moi", "toi", "se", "leur", "son", "sa",
    "a", "the", "an", "of", "to", "and", "or", "is",
    "qu", "c", "n", "l", "d", "s", "t", "m",
}


def _query_tokens(text: str) -> Set[str]:
    """Tokens significatifs de la query (sans accents, > 2 chars, hors stopwords)."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", no_accents)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _clean_chunk(text: str) -> str:
    """Nettoie un chunk pour l'affichage : retire les headers markdown."""
    # Retire les # en debut de ligne (## Titre -> Titre)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    # Retire les ## inline au milieu d'un paragraphe concatene
    text = re.sub(r"\s*##+\s+", ". ", text)
    # Normalise les espaces multiples
    text = re.sub(r"\s+", " ", text)
    # Evite les points doubles crees par la substitution ci-dessus
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip()


class Scathach(MorriganModule):
    """
    Module de generation de texte de Morrigan.

    Phase 1 : Templates Jinja2 avec bascule selon le contexte.
    """

    # Seuil de similarite minimal pour considerer un chunk comme pertinent.
    # Si le reranker est actif, on utilise le score cosine original
    # (stocke dans metadata["score_cosine"]) car le score cross-encoder
    # est sur une echelle differente (-inf, +inf).
    MIN_RELEVANCE_SCORE = 0.42

    def __init__(
        self,
        backend: str = "template",
        templates_dir: str = "modules/scathach/templates",
    ):
        self.backend = backend
        self.templates_dir = Path(templates_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info(
            "Scathach — backend=%s, templates=%s",
            backend,
            self.templates_dir,
        )

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Genere du texte a partir des donnees des autres modules."""
        logger.info("Scathach genere pour: %s", input.query[:60])

        previous = input.context.get("previous_results", {})

        if self.backend == "template":
            # Phase 2 : si Morrigan-Code a tourne, on rend sa verification.
            if "morrigan_code" in previous:
                response = self._render_code_verification(input.query, previous)
            else:
                response = self._render_from_modules(input.query, previous)
        else:
            response = f"[Backend '{self.backend}' non implemente]"

        return ModuleOutput(
            result=response,
            confidence=0.6,
            metadata={
                "backend": self.backend,
                "modules_used": list(previous.keys()),
            },
        )

    def _render_from_modules(
        self, query: str, previous: Dict[str, Any]
    ) -> str:
        """
        Assemble une reponse a partir des outputs des modules precedents.
        Decide du template a utiliser selon le contexte.
        """
        # Extraire les chunks Danann
        chunks = self._extract_chunks(previous)

        # Extraire le type de structure Ogham
        structure_type = self._extract_structure_type(previous)

        # Filtrer les chunks par pertinence.
        # Si le reranker est actif, le "score" est le score cross-encoder
        # (echelle -inf/+inf). On utilise score_cosine (original) pour le seuil.
        relevant_chunks = []
        for c in chunks:
            cosine = c.get("metadata", {}).get("score_cosine", c.get("score", 0))
            if cosine >= self.MIN_RELEVANCE_SCORE:
                relevant_chunks.append(c)

        # Garde anti-faux-positif : au moins un token rare de la query
        # doit apparaitre dans le top chunk. Sinon on considere que c'est
        # hors corpus.
        q_tokens = _query_tokens(query)
        if relevant_chunks and q_tokens:
            top_text = relevant_chunks[0].get("text", "").lower()
            top_text_norm = "".join(
                c for c in unicodedata.normalize("NFKD", top_text)
                if not unicodedata.combining(c)
            )
            if not any(tok in top_text_norm for tok in q_tokens):
                logger.info(
                    "Scathach: aucun token rare de la query dans le top chunk "
                    "-> fallback not_found"
                )
                relevant_chunks = []

        # Nettoyer le markdown de tous les chunks retenus
        if relevant_chunks:
            cleaned = []
            for c in relevant_chunks:
                c_copy = dict(c)
                c_copy["text"] = _clean_chunk(c_copy.get("text", ""))
                cleaned.append(c_copy)
            relevant_chunks = cleaned

        # Decider si on montre les chunks secondaires :
        # seulement si proches en score cosine du meilleur (evite les tangentes).
        # On utilise score_cosine pour comparer (echelle stable 0-1).
        show_extras = False
        if len(relevant_chunks) > 1:
            def _cosine(c: Dict) -> float:
                return c.get("metadata", {}).get(
                    "score_cosine", c.get("score", 0)
                )
            top_score = _cosine(relevant_chunks[0])
            second_score = _cosine(relevant_chunks[1])
            show_extras = second_score >= top_score * 0.90
        display_chunks = relevant_chunks if show_extras else relevant_chunks[:1]

        # Cas 1 : pas de chunks pertinents et pas de Cauldron -> not_found
        if not relevant_chunks and "cauldron" not in previous:
            return self._render("not_found.j2", query=query, suggestions=[])

        # Cas 2 : on a des chunks -> reponse factuelle/comparaison/explication
        if relevant_chunks:
            # Pour une comparaison, on garde TOUS les chunks pertinents
            # (le template comparison.j2 les liste tous)
            chunks_for_template = (
                relevant_chunks if structure_type == "comparison"
                else display_chunks
            )

            template_name = {
                "comparison": "comparison.j2",
                "explanation": "explanation.j2",
                "analysis": "explanation.j2",
            }.get(structure_type, "factual.j2")

            return self._render(
                template_name,
                query=query,
                chunks=chunks_for_template,
                structure_type=structure_type,
                intro=None,
                synthesis=None,
                sources_note=None,
            )

        # Cas 3 : conversation (Cauldron + pas de chunks)
        turn_count = 0
        if "cauldron" in previous:
            cauldron_result = previous["cauldron"].result
            if cauldron_result and isinstance(cauldron_result, dict):
                turn_count = cauldron_result.get("turn_count", 0)

        return self._render(
            "conversation.j2",
            query=query,
            query_lower=query.lower().strip().rstrip("!?."),
            turn_count=turn_count,
        )

    def _render_code_verification(
        self, query: str, previous: Dict[str, Any]
    ) -> str:
        """Rend la verification Morrigan-Code via le template dedie."""
        code_out = previous.get("morrigan_code")
        if not code_out or not isinstance(code_out.result, dict):
            return self._render("not_found.j2", query=query, suggestions=[])

        result = code_out.result
        return self._render(
            "code_verification.j2",
            query=query,
            verified=result.get("verified", []),
            all_valid=result.get("all_valid", False),
            blocks_verified=code_out.metadata.get("blocks_verified", 0),
        )

    def _render(self, template_name: str, **context: Any) -> str:
        """Rend un template Jinja2 avec le contexte fourni."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context).strip()
        except Exception as e:
            logger.error("Erreur rendu template '%s': %s", template_name, e)
            return f"[Morrigan] Erreur de generation: {e}"

    @staticmethod
    def _extract_chunks(previous: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait les chunks de l'output de Danann."""
        if "danann" not in previous:
            return []
        danann_result = previous["danann"].result
        if not danann_result or not isinstance(danann_result, dict):
            return []
        return danann_result.get("chunks", [])

    @staticmethod
    def _extract_structure_type(previous: Dict[str, Any]) -> str:
        """Extrait le type de structure de l'output d'Ogham."""
        if "ogham" not in previous:
            return "explanation"
        ogham_result = previous["ogham"].result
        if not ogham_result or not isinstance(ogham_result, dict):
            return "explanation"
        return ogham_result.get("type", "explanation")

    async def health_check(self) -> bool:
        # Verifier que les templates existent
        required = ["factual.j2", "comparison.j2", "explanation.j2", "conversation.j2", "not_found.j2", "code_verification.j2"]
        for t in required:
            if not (self.templates_dir / t).exists():
                logger.warning("Template manquant: %s", t)
                return False
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Scathach",
            "type": "language_generation",
            "backend": self.backend,
            "capabilities": [
                "text_generation",
                "template_filling",
                "context_aware_response",
            ],
        }
