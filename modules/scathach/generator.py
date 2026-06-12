"""
SCATHACH — Module langage : generation de texte.

Phase 1 : Templates Jinja2 avec variantes + assemblage intelligent
des resultats des autres modules.
Phase 2 : Backend RWKV (0.19B-1.5B) pour une vraie generation.
"""

import asyncio
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
)

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.types import ModuleInput, ModuleOutput, MorriganModule

logger = logging.getLogger("morrigan.scathach")


async def _aiter_in_thread(
    make_iter: Callable[[], Iterator[str]]
) -> AsyncIterator[str]:
    """Pompe un générateur synchrone *bloquant* dans un thread et relaie ses
    morceaux en async via une queue — pour ne PAS bloquer l'event loop avec
    l'inférence llama.cpp (cf. audit F2). Le générateur est construit DANS le
    thread (`make_iter()`) pour que l'ouverture du modèle soit aussi offloadée.
    Les exceptions du producteur sont relayées au consommateur.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()  # non bornée : flux fini (max_tokens)
    _DONE = object()

    def _produce() -> None:
        try:
            for item in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # relayée côté consommateur
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    fut = loop.run_in_executor(None, _produce)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Attend la fin réelle du thread : le travail CPU compte jusqu'au bout
        # (le sémaphore de l'API reste tenu tant que la génération tourne).
        await fut

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
    # Lu sur metadata["score_cosine"] = cosinus PUR posé par Danann (le
    # score brut inclut un boost lexical jusqu'à +0.30 qui fausserait le
    # gate ; le score reranker est sur une échelle différente).
    #
    # 0.84 : calibré pour l'embedder multilingual-e5 (Phase 2A) sur le jeu
    # scripts/eval_rag.py (56 in-corpus / 24 hors-corpus, index code 46k) —
    # in-corpus passants identiques au seuil historique (52/56), refus
    # hors-corpus 16/24 → 20/24. L'ancien 0.42 datait de MiniLM : e5
    # concentre les cosinus (~0.78-0.95), TOUT passait 0.42 et seul le
    # garde « token rare » refusait. Au-delà de 0.85 l'in-corpus
    # s'effondre (0.87 → 25/56). Surcharger via MORRIGAN_MIN_RELEVANCE.
    MIN_RELEVANCE_SCORE = 0.84

    # Nombre max de chunks passés en contexte au backend RWKV.
    # Réduit de 4 à 2 après mesure : le prefill RWKV est ~linéaire en
    # longueur de prompt, et le time-to-first-token chute de ~2.3s
    # (4 chunks) à ~0.9s (1-2 chunks). Les top-2 chunks (les mieux
    # classés par Danann) portent l'essentiel de la pertinence ;
    # au-delà on dilue le contexte ET on rallonge le prefill.
    RWKV_CONTEXT_CHUNKS = 2

    def __init__(
        self,
        backend: str = "template",
        templates_dir: str = "modules/scathach/templates",
        rwkv_backend: Optional[Any] = None,
        strict_rag: bool = True,
    ):
        """
        backend :
          - "template" : templates Jinja2 uniquement (Phase 1/2, défaut)
          - "rwkv"     : génération RWKV, fallback template si indispo
          - "auto"     : alias de "rwkv" (RWKV si dispo, sinon template)
        rwkv_backend : instance injectable (tests). Si None et backend
          le requiert, un RWKVBackend() est créé paresseusement.
        strict_rag (défaut True) : en génération RWKV, n'autorise la
          génération QUE si un contexte (chunks Danann + faits Ogham)
          est disponible, et instruit le modèle de répondre "Je ne
          sais pas" si la réponse n'y est pas. Sans contexte → pas
          d'appel LLM, on renvoie un "je ne sais pas" déterministe via
          template. C'est le cœur du "0 hallucination" de Morrigan.
        """
        self.backend = backend
        self.templates_dir = Path(templates_dir)
        self._rwkv = rwkv_backend
        self.strict_rag = strict_rag
        # Surcharge optionnelle du seuil de pertinence (cf. attribut de
        # classe) — ex. recalibrage après changement d'embedder ou de
        # corpus, sans toucher au code. Invalide → ignoré (jamais
        # d'exception au boot).
        raw = os.environ.get("MORRIGAN_MIN_RELEVANCE", "").strip()
        if raw:
            try:
                self.MIN_RELEVANCE_SCORE = float(raw)
            except ValueError:
                logger.warning(
                    "MORRIGAN_MIN_RELEVANCE=%r invalide — seuil %s conservé",
                    raw, self.MIN_RELEVANCE_SCORE,
                )
        # Trace du dernier chemin de génération ("rwkv" / "template"),
        # exposée pour l'observabilité (/stats), notamment en streaming.
        self.last_generated_by: Optional[str] = None

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

    def _get_rwkv(self) -> Optional[Any]:
        """Renvoie le backend RWKV (injecté ou créé paresseusement). None si mode template."""
        if self._rwkv is not None:
            return self._rwkv
        if self.backend in ("rwkv", "auto"):
            from modules.scathach.rwkv_backend import RWKVBackend  # noqa: PLC0415
            self._rwkv = RWKVBackend()
            return self._rwkv
        return None

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Genere du texte a partir des donnees des autres modules."""
        logger.info("Scathach genere pour: %s", input.query[:60])

        previous = input.context.get("previous_results", {})
        generated_by = "template"

        # La verification de code reste sur template (sortie structuree,
        # mieux servie par un template que par de la generation libre).
        if "morrigan_code" in previous:
            response = self._render_code_verification(input.query, previous)
        else:
            response = None
            # Tentative RWKV si le backend le demande. L'inférence (et le
            # retrieval qu'elle déclenche) est bloquante → offload hors de
            # l'event loop pour ne pas geler l'API (audit F2).
            if self.backend in ("rwkv", "auto"):
                rwkv_response = await asyncio.to_thread(
                    self._generate_rwkv, input.query, previous
                )
                if rwkv_response is not None:
                    response = rwkv_response
                    generated_by = "rwkv"
            # Fallback template (ou mode template pur).
            if response is None:
                response = self._render_from_modules(input.query, previous)

        self.last_generated_by = generated_by
        return ModuleOutput(
            result=response,
            confidence=0.7 if generated_by == "rwkv" else 0.6,
            metadata={
                "backend": self.backend,
                "generated_by": generated_by,
                "modules_used": list(previous.keys()),
            },
        )

    # Humanisation des prédicats du KG Ogham pour le contexte RAG.
    _PREDICATE_FR = {
        "is_a": "est",
        "has": "possède",
        "uses": "utilise",
        "of": "de",
        "co_occurs_with": "est lié à",
    }
    # Plafond de faits Ogham injectés (évite de noyer le prompt).
    OGHAM_CONTEXT_FACTS = 8

    def _rwkv_context(
        self, query: str, previous: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Décide si on génère via RWKV et avec quel contexte.

        Renvoie :
          - None  → fallback template (backend indispo, OU mode strict
            sans aucun contexte fiable = "je ne sais pas" déterministe)
          - list  → générer via RWKV avec ce contexte (peut être [] en
            mode non-strict = génération libre)

        Partagé par _generate_rwkv (process) et stream().
        """
        backend = self._get_rwkv()
        if backend is None or not backend.is_available():
            return None

        chunks = self._relevant_chunks(query, previous)
        context = [c.get("text", "") for c in chunks[: self.RWKV_CONTEXT_CHUNKS]]
        context += self._ogham_context(previous)
        context = [c for c in context if c.strip()]

        if self.strict_rag and not context:
            logger.info(
                "RAG strict : aucun contexte fiable → fallback template "
                "(réponse 'je ne sais pas')"
            )
            return None
        return context

    def _generate_rwkv(
        self, query: str, previous: Dict[str, Any]
    ) -> Optional[str]:
        """Génère via RWKV avec RAG. None → fallback template."""
        context = self._rwkv_context(query, previous)
        if context is None:
            return None
        try:
            return self._get_rwkv().answer(
                query, context=context or None, strict=self.strict_rag
            )
        except Exception as e:  # pragma: no cover - dépend de l'env llama.cpp
            logger.error("RWKV génération échouée (%s) — fallback template", e)
            return None

    async def stream(self, input: ModuleInput) -> AsyncIterator[str]:
        """Génère en streaming : yield les morceaux de réponse au fil de l'eau.

        Même logique de décision que process() :
          - morrigan_code → template (yieldé en un bloc)
          - RWKV dispo + contexte (ou non-strict) → stream des tokens
          - sinon → template / refus (yieldé en un bloc)

        Pensé pour la CLI/Telegram : la réponse s'affiche en direct,
        ce qui masque la latence d'un 1.6B sur CPU (cf. benchmarks).
        """
        previous = input.context.get("previous_results", {})

        if "morrigan_code" in previous:
            self.last_generated_by = "template"
            yield self._render_code_verification(input.query, previous)
            return

        if self.backend in ("rwkv", "auto"):
            # _rwkv_context déclenche le retrieval (embeddings) → bloquant.
            context = await asyncio.to_thread(
                self._rwkv_context, input.query, previous
            )
            if context is not None:
                try:
                    backend = self._get_rwkv()
                    # answer_stream est un générateur synchrone bloquant : on
                    # le pompe dans un thread pour ne pas geler l'event loop.
                    async for piece in _aiter_in_thread(
                        lambda: backend.answer_stream(
                            input.query,
                            context=context or None,
                            strict=self.strict_rag,
                        )
                    ):
                        yield piece
                    self.last_generated_by = "rwkv"
                    return
                except Exception as e:  # pragma: no cover - env llama.cpp
                    logger.error("RWKV streaming échoué (%s) — fallback template", e)

        # Fallback template / refus : yieldé en un seul bloc.
        self.last_generated_by = "template"
        yield self._render_from_modules(input.query, previous)

    def _ogham_context(self, previous: Dict[str, Any]) -> List[str]:
        """Convertit les faits du KG Ogham (compare/facts) en lignes de contexte FR."""
        if "ogham" not in previous:
            return []
        result = previous["ogham"].result
        if not result or not isinstance(result, dict):
            return []

        lines: List[str] = []

        compare = result.get("compare")
        if compare:
            a, b = compare.get("a", "?"), compare.get("b", "?")
            common = [n.get("label", n.get("id", "")) for n in compare.get("common_neighbors", [])]
            a_only = [n.get("label", n.get("id", "")) for n in compare.get("a_only", [])]
            b_only = [n.get("label", n.get("id", "")) for n in compare.get("b_only", [])]
            if common:
                lines.append(f"{a} et {b} ont en commun : {', '.join(common)}.")
            if a_only:
                lines.append(f"{a} spécifiquement : {', '.join(a_only)}.")
            if b_only:
                lines.append(f"{b} spécifiquement : {', '.join(b_only)}.")

        facts = result.get("facts")
        if facts:
            for r in facts.get("relations", [])[: self.OGHAM_CONTEXT_FACTS]:
                subj = r.get("subject", "")
                obj = r.get("object", "")
                pred = self._PREDICATE_FR.get(r.get("predicate", ""), r.get("predicate", ""))
                if subj and obj:
                    lines.append(f"{subj} {pred} {obj}.")

        return lines

    def _relevant_chunks(
        self, query: str, previous: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extrait + filtre les chunks Danann pertinents pour la query.

        Partagé par le rendu template ET la génération RWKV (RAG).
        Applique : seuil de score cosine, garde anti-faux-positif
        (≥ 1 token rare de la query dans le top chunk), nettoyage markdown.
        """
        chunks = self._extract_chunks(previous)

        relevant_chunks = []
        for c in chunks:
            cosine = c.get("metadata", {}).get("score_cosine", c.get("score", 0))
            if cosine >= self.MIN_RELEVANCE_SCORE:
                relevant_chunks.append(c)

        # Garde anti-faux-positif : au moins un token rare de la query
        # doit apparaitre dans le top chunk. Sinon c'est hors corpus.
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
                    "-> hors corpus"
                )
                return []

        # Nettoyer le markdown de tous les chunks retenus
        cleaned = []
        for c in relevant_chunks:
            c_copy = dict(c)
            c_copy["text"] = _clean_chunk(c_copy.get("text", ""))
            cleaned.append(c_copy)
        return cleaned

    def _render_from_modules(
        self, query: str, previous: Dict[str, Any]
    ) -> str:
        """
        Assemble une reponse a partir des outputs des modules precedents.
        Decide du template a utiliser selon le contexte.
        """
        # Extraire le type de structure Ogham
        structure_type = self._extract_structure_type(previous)

        # Chunks pertinents (extraction + filtrage + nettoyage partagés).
        relevant_chunks = self._relevant_chunks(query, previous)

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
        """Extrait le sous-type de structure de l'output d'Ogham.

        Depuis le chantier KG (PR #13), Ogham renvoie
        `result["type"] == "structured_response"` et le sous-type
        (comparison/explanation/analysis) sous `result["structure_type"]`.
        On lit `structure_type` en priorité, avec rétrocompat sur l'ancien
        `type` (qui portait le sous-type avant le KG).
        """
        if "ogham" not in previous:
            return "explanation"
        ogham_result = previous["ogham"].result
        if not ogham_result or not isinstance(ogham_result, dict):
            return "explanation"
        sub = ogham_result.get("structure_type")
        if sub:
            return sub
        legacy = ogham_result.get("type", "explanation")
        # Si "type" vaut le marqueur stable, pas un sous-type → défaut.
        return "explanation" if legacy == "structured_response" else legacy

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
