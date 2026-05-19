"""
AN DAGDA — L'Orchestrateur Central de Morrigan.

Le "Bon Dieu" celtique, père des Tuatha Dé Danann.
Analyse chaque input, détermine quel(s) module(s) activer,
dans quel ordre, et assemble les résultats.
"""

import asyncio
import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

import yaml


def _normalize(text: str) -> str:
    """Lowercase + strip accents pour un matching robuste."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))

from core.types import (
    ModuleInput,
    ModuleOutput,
    MorriganModule,
    QueryType,
    RoutingDecision,
)

logger = logging.getLogger("morrigan.dagda")


class AnDagda:
    """Orchestrateur principal de Morrigan."""

    def __init__(self, config_path: str = "config/morrigan.yaml"):
        self.modules: Dict[str, MorriganModule] = {}
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        logger.info("An Dagda s'éveille...")

    async def initialize(self) -> None:
        """Charge la configuration et initialise les modules."""
        self._load_config()
        await self._check_modules_health()
        logger.info(
            "An Dagda initialisé avec %d module(s): %s",
            len(self.modules),
            list(self.modules.keys()),
        )

    def _load_config(self) -> None:
        """Charge la configuration YAML."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info("Configuration chargée depuis %s", self.config_path)
        except FileNotFoundError:
            logger.warning(
                "Fichier de configuration %s introuvable, utilisation des défauts",
                self.config_path,
            )
            self.config = {}

    def register_module(self, name: str, module: MorriganModule) -> None:
        """Enregistre un module auprès de l'orchestrateur."""
        self.modules[name] = module
        logger.info("Module '%s' enregistré", name)

    async def _check_modules_health(self) -> None:
        """Vérifie la santé de tous les modules enregistrés."""
        for name, module in self.modules.items():
            try:
                healthy = await module.health_check()
                status = "OK" if healthy else "DÉGRADÉ"
                logger.info("Module '%s': %s", name, status)
            except Exception as e:
                logger.error("Module '%s': ERREUR — %s", name, e)

    # Phase 2 : detection de domaine par mots-cles dans la query.
    # Permet de passer un filtre domain= a Danann pour desambiguer
    # (ex: "projet Morrigan" vs "Morrigan mythologie").
    _DOMAIN_HINTS = {
        "projet": ["projet morrigan", "le projet", "architecture morrigan",
                    "module morrigan", "pipeline morrigan", "roadmap"],
        "reseau": ["tcp", "udp", "http", "https", "dns", "ip ", "port ",
                    "protocole", "firewall", "routage", "ssh", "tls",
                    "reseau", "paquet", "serveur"],
        "ia": ["transformer", "llm", "embedding", "neurone", "entrainement",
               "inference", "mamba", "rwkv", "cfc", "lnn", "kan",
               "intelligence artificielle", "deep learning", "machine learning"],
        "mythologie": ["dieu", "deesse", "celtique", "druide", "tuatha",
                       "dagda", "brigid", "ogham", "cuchulainn", "scathach",
                       "mythologie", "legende"],
        "code": ["python", "javascript", "bash", "sql", "html", "css",
                 "fonction", "variable", "class ", "import ", "code"],
    }

    def _detect_domain_hint(self, query_norm: str) -> Optional[str]:
        """Detecte un indice de domaine dans la query normalisee."""
        scores: Dict[str, int] = {}
        for domain, keywords in self._DOMAIN_HINTS.items():
            score = sum(1 for kw in keywords if kw in query_norm)
            if score > 0:
                scores[domain] = score

        if not scores:
            return None

        # Retourne le domaine avec le plus de hits
        best = max(scores, key=scores.get)  # type: ignore
        # Seuil : au moins 1 keyword match
        return best if scores[best] >= 1 else None

    # Phase 2 : detection de fence markdown ```lang ... ``` (signal fort code).
    _CODE_FENCE_PATTERN = re.compile(r"```\w*\s*\n", re.MULTILINE)

    # Phase 2+ : seuil minimal de confidence Brigid pour court-circuiter
    # les heuristiques. Sous ce seuil, on tombe sur les mots-clés (qui
    # font mieux sur les cas ambigus pour le modèle, ex: "pourquoi" en
    # début de phrase = reasoning fort, alors que Brigid peut hésiter
    # entre factual/reasoning sur du général).
    BRIGID_CONFIDENCE_THRESHOLD = 0.5

    # Mapping unifié QueryType → liste de modules dans l'ordre d'exécution.
    # Utilisé pour les routages Brigid ET heuristiques pour éviter les
    # divergences silencieuses.
    _ROUTING_MAP: Dict[QueryType, List[str]] = {
        QueryType.FACTUAL: ["danann", "ogham", "scathach"],
        QueryType.REASONING: ["danann", "ogham", "scathach"],
        QueryType.CREATIVE: ["brigid", "scathach"],
        QueryType.CONVERSATION: ["cauldron", "scathach"],
        QueryType.COMPLEX: ["cauldron", "danann", "brigid", "ogham", "scathach"],
        QueryType.CODE: ["morrigan_code", "scathach"],
    }

    def _route_via_brigid(
        self, query: str, query_norm: str
    ) -> Optional[RoutingDecision]:
        """Tente une classification via Brigid (LNN). Renvoie None si
        Brigid n'est pas registered, pas chargeable, ou pas assez sûr.

        L'appelant doit alors fallback aux heuristiques.
        """
        brigid = self.modules.get("brigid")
        if brigid is None or not hasattr(brigid, "classify_intent"):
            return None

        classif = brigid.classify_intent(query)
        if classif is None:
            return None  # checkpoint indisponible
        if classif.confidence < self.BRIGID_CONFIDENCE_THRESHOLD:
            logger.info(
                "Brigid : %s (%.2f) sous le seuil %.2f → fallback heuristiques",
                classif.label, classif.confidence, self.BRIGID_CONFIDENCE_THRESHOLD,
            )
            return None

        try:
            qt = QueryType(classif.label)
        except ValueError:
            # Label Brigid hors enum (ne devrait pas arriver si LABELS
            # est sync avec QueryType — garde-fou).
            logger.warning("Brigid : label inconnu %r → fallback", classif.label)
            return None

        return RoutingDecision(
            query_type=qt,
            modules=self._ROUTING_MAP[qt],
            reasoning=f"Brigid LNN (confidence {classif.confidence:.2f})",
            domain_hint=self._detect_domain_hint(query_norm),
        )

    def classify_query(self, query: str) -> RoutingDecision:
        """
        Classifie une requête et détermine le plan de routage.

        Ordre de décision :
          1. Fence markdown ```lang … ``` → CODE (signal trop fort pour discuter)
          2. Brigid-Classifier (LNN/CfC) si registered et confidence ≥ seuil
          3. Heuristiques mots-clés (fallback, ex-Phase 0)

        Le step 2 est désactivé silencieusement si Brigid est absent,
        non-trained, ou peu confiant — pas de régression côté routage.
        """
        # 1. code en priorite absolue si fence markdown detectee.
        if self._CODE_FENCE_PATTERN.search(query):
            return RoutingDecision(
                query_type=QueryType.CODE,
                modules=self._ROUTING_MAP[QueryType.CODE],
                reasoning="Bloc de code détecté (fence markdown)",
                domain_hint="code",
            )

        query_norm = _normalize(query.strip())

        # 2. Brigid LNN si dispo et confiant.
        brigid_decision = self._route_via_brigid(query, query_norm)
        if brigid_decision is not None:
            return brigid_decision

        # 3. Heuristiques mots-clés (fallback).
        # Salutations : priorite absolue -> conversation
        greetings = (
            "salut", "bonjour", "bonsoir", "hello", "hey", "coucou",
            "yo ", "ca va", "comment ca va", "comment tu vas",
            "comment vas tu", "quoi de neuf", "merci", "au revoir",
        )
        if any(query_norm.startswith(g) or query_norm == g.strip()
               for g in greetings):
            return RoutingDecision(
                query_type=QueryType.CONVERSATION,
                modules=self._ROUTING_MAP[QueryType.CONVERSATION],
                reasoning="Salutation ou conversation sociale",
            )

        # Heuristiques simples — Phase 0 (mots-clés déjà normalisés)
        creative_keywords = [
            "ecris", "invente", "imagine", "poeme", "histoire",
            "cree", "compose", "raconte",
        ]
        reasoning_keywords = [
            "pourquoi", "explique", "compare", "difference",
            "logique", "raisonne", "analyse",
            "comment fonctionne", "comment marche",
        ]
        # Mots interrogatifs en début de phrase → signal factuel fort
        # Note: "comment" seul est ambigu (comment ca va vs comment fonctionne),
        # il est geree via reasoning_keywords plus specifiques.
        interrogative_starts = (
            "qui ", "que ", "quoi ", "quel ", "quelle ", "quels ", "quelles ",
            "quand ", "ou ", "combien ", "c'est quoi",
            "qu'est-ce", "qu est-ce", "donne", "liste", "cite", "nomme",
            "parle moi", "parle-moi", "raconte moi", "dis moi", "dis-moi",
        )
        factual_keywords = [
            "definition", "signifie",
        ]

        # Un point d'interrogation est un signal factuel fort
        has_question_mark = "?" in query_norm
        starts_interrogative = any(
            query_norm.startswith(w) for w in interrogative_starts
        )

        # Phase 2 : detection de domaine
        domain_hint = self._detect_domain_hint(query_norm)

        if any(kw in query_norm for kw in creative_keywords):
            qt = QueryType.CREATIVE
            reason = "Mots-clés créatifs détectés"
        elif any(kw in query_norm for kw in reasoning_keywords):
            qt = QueryType.REASONING
            reason = "Mots-clés de raisonnement détectés"
        elif starts_interrogative or has_question_mark or any(
            kw in query_norm for kw in factual_keywords
        ):
            qt = QueryType.FACTUAL
            reason = "Question factuelle détectée"
        else:
            qt = QueryType.CONVERSATION
            reason = "Conversation courante (défaut)"

        return RoutingDecision(
            query_type=qt,
            modules=self._ROUTING_MAP[qt],
            reasoning=reason,
            domain_hint=domain_hint,
        )

    async def process(self, user_input: str, session_id: str = "default") -> str:
        """
        Point d'entrée principal — traite un input utilisateur.

        1. Classifie la requête
        2. Route vers les modules appropriés
        3. Assemble la réponse finale
        """
        start_time = time.time()
        logger.info("Input reçu [session=%s]: %s", session_id, user_input[:100])

        # 1. Classification
        routing = self.classify_query(user_input)
        logger.info(
            "Routage: %s → %s (%s) domain=%s",
            routing.query_type.value,
            routing.modules,
            routing.reasoning,
            routing.domain_hint,
        )

        # 2. Exécution séquentielle des modules
        # Phase 2 : passer le domain_hint en parametre pour Danann
        parameters = {}
        if routing.domain_hint:
            parameters["domain"] = routing.domain_hint

        module_input = ModuleInput(
            query=user_input,
            context={"session_id": session_id, "routing": routing},
            parameters=parameters,
        )

        accumulated_result: Dict[str, Any] = {}
        for module_name in routing.modules:
            if module_name not in self.modules:
                logger.warning("Module '%s' non disponible, skip", module_name)
                continue

            module = self.modules[module_name]
            try:
                # Passer les résultats précédents dans le contexte
                module_input.context["previous_results"] = accumulated_result
                output = await module.process(module_input)
                accumulated_result[module_name] = output
                logger.info(
                    "Module '%s' terminé (confiance=%.2f)",
                    module_name,
                    output.confidence,
                )
            except Exception as e:
                logger.error("Erreur module '%s': %s", module_name, e)
                accumulated_result[module_name] = ModuleOutput(
                    result=None, errors=[str(e)]
                )

        # 3. Assemblage de la réponse
        response = self._assemble_response(accumulated_result, routing)

        elapsed = time.time() - start_time
        logger.info("Réponse générée en %.2fs", elapsed)

        return response

    def _assemble_response(
        self,
        results: Dict[str, ModuleOutput],
        routing: RoutingDecision,
    ) -> str:
        """
        Assemble la réponse finale à partir des outputs des modules.

        Phase 0 : retourne le résultat du dernier module de la chaîne.
        Phase 2+ : assemblage intelligent multi-modules.
        """
        # Prendre le résultat du dernier module qui a répondu
        for module_name in reversed(routing.modules):
            if module_name in results:
                output = results[module_name]
                if output.success and output.result is not None:
                    return str(output.result)

        return "[Morrigan] Aucun module n'a pu traiter cette requête."
