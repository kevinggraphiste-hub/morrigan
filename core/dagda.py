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
from typing import Any, AsyncIterator, Dict, List, Optional

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
        # Observabilité (/stats) : compteurs cumulés + trace de la
        # dernière requête.
        self.stats: Dict[str, Any] = {
            "queries": 0,
            "by_type": {},          # query_type -> count
            "by_generated_by": {},  # rwkv|template -> count
            "total_latency_s": 0.0,
        }
        self._last_brigid: Any = None       # IntentClassification ou None
        self.last_routing: Optional[RoutingDecision] = None
        self.last_latency_s: float = 0.0
        self.last_generated_by: Optional[str] = None
        logger.info("An Dagda s'éveille...")

    def _record_query(
        self, routing: RoutingDecision, latency_s: float, generated_by: Optional[str]
    ) -> None:
        """Met à jour les compteurs + la trace de la dernière requête."""
        self.stats["queries"] += 1
        qt = routing.query_type.value
        self.stats["by_type"][qt] = self.stats["by_type"].get(qt, 0) + 1
        if generated_by:
            self.stats["by_generated_by"][generated_by] = (
                self.stats["by_generated_by"].get(generated_by, 0) + 1
            )
        self.stats["total_latency_s"] += latency_s
        self.last_routing = routing
        self.last_latency_s = latency_s
        self.last_generated_by = generated_by

    def format_stats(self) -> str:
        """Rend les statistiques d'observabilité en texte (CLI / Telegram)."""
        s = self.stats
        n = s["queries"]
        lines = ["📊 Morrigan — observabilité", ""]
        lines.append(f"Modules enregistrés : {', '.join(self.modules) or '(aucun)'}")
        lines.append(f"Requêtes traitées   : {n}")

        if n:
            avg = s["total_latency_s"] / n
            lines.append(f"Latence moyenne     : {avg:.2f}s")
            if s["by_type"]:
                rep = ", ".join(f"{k}={v}" for k, v in sorted(s["by_type"].items()))
                lines.append(f"Par type            : {rep}")
            if s["by_generated_by"]:
                rep = ", ".join(f"{k}={v}" for k, v in sorted(s["by_generated_by"].items()))
                lines.append(f"Génération          : {rep}")

        if self.last_routing is not None:
            lines += ["", "Dernière requête :"]
            lines.append(f"  type      : {self.last_routing.query_type.value}")
            lines.append(f"  routage   : {self.last_routing.reasoning}")
            lines.append(f"  modules   : {' → '.join(self.last_routing.modules)}")
            if self.last_routing.domain_hint:
                lines.append(f"  domaine   : {self.last_routing.domain_hint}")
            if self.last_generated_by:
                lines.append(f"  généré par: {self.last_generated_by}")
            lines.append(f"  latence   : {self.last_latency_s:.2f}s")
            if self._last_brigid is not None:
                b = self._last_brigid
                top = sorted(
                    b.probabilities.items(), key=lambda kv: -kv[1]
                )[:3]
                top_str = ", ".join(f"{k} {p*100:.0f}%" for k, p in top)
                lines.append(
                    f"  brigid    : {b.label} ({b.confidence*100:.0f}%) "
                    f"[top: {top_str}]"
                )

        return "\n".join(lines)

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
        # Mémorise la classification pour /stats (même sous le seuil).
        self._last_brigid = classif
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
        # Reset de la trace Brigid pour cette requête (renseignée par
        # _route_via_brigid si Brigid est consultée).
        self._last_brigid = None

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

        # 4. Observabilité (/stats)
        sca = accumulated_result.get("scathach")
        gen_by = sca.metadata.get("generated_by") if sca and not sca.errors else None
        self._record_query(routing, elapsed, gen_by)

        return response

    async def process_stream(
        self, user_input: str, session_id: str = "default"
    ) -> AsyncIterator[str]:
        """Variante streaming de process().

        Exécute normalement tous les modules sauf le dernier de la
        chaîne, puis **streame** la sortie du dernier (Scáthach) si
        celui-ci expose une méthode `stream()`. Sinon, fallback : on
        process() le dernier et on yield sa réponse en un bloc.

        Permet à la CLI/Telegram d'afficher la réponse au fil de l'eau
        (le 1er token arrive en <1s même si le total prend plusieurs
        secondes sur CPU).
        """
        start_time = time.time()
        routing = self.classify_query(user_input)
        logger.info(
            "Routage (stream): %s → %s (%s)",
            routing.query_type.value, routing.modules, routing.reasoning,
        )

        parameters: Dict[str, Any] = {}
        if routing.domain_hint:
            parameters["domain"] = routing.domain_hint

        module_input = ModuleInput(
            query=user_input,
            context={"session_id": session_id, "routing": routing},
            parameters=parameters,
        )

        if not routing.modules:
            yield "[Morrigan] Aucun module à activer."
            return

        *pre_modules, last_name = routing.modules
        accumulated: Dict[str, Any] = {}

        # 1. Modules amont (tous sauf le dernier) — exécution normale.
        for name in pre_modules:
            if name not in self.modules:
                logger.warning("Module '%s' non disponible, skip", name)
                continue
            module_input.context["previous_results"] = accumulated
            try:
                accumulated[name] = await self.modules[name].process(module_input)
            except Exception as e:
                logger.error("Erreur module '%s': %s", name, e)
                accumulated[name] = ModuleOutput(result=None, errors=[str(e)])

        # 2. Dernier module : streamé s'il le supporte.
        module_input.context["previous_results"] = accumulated
        last_mod = self.modules.get(last_name)

        if last_mod is None:
            yield "[Morrigan] Aucun module n'a pu traiter cette requête."
        elif hasattr(last_mod, "stream"):
            async for piece in last_mod.stream(module_input):  # type: ignore[attr-defined]
                yield piece
        else:
            out = await last_mod.process(module_input)
            yield str(out.result) if out.success and out.result is not None else \
                "[Morrigan] Aucun module n'a pu traiter cette requête."

        elapsed = time.time() - start_time
        logger.info("Réponse (stream) en %.2fs", elapsed)

        # Observabilité (/stats) : generated_by exposé par Scáthach.
        gen_by = getattr(last_mod, "last_generated_by", None) if last_mod else None
        self._record_query(routing, elapsed, gen_by)

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
