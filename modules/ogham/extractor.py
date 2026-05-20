"""
OGHAM — Extraction d'entités et relations depuis du texte FR.

PR 2 du chantier knowledge graph. Pure-Python, **zéro dépendance NLP
externe** (pas de spaCy/Stanza/transformers ici — c'est le rôle d'un
upgrade futur, hors scope Phase 2). On vise une extraction simple,
déterministe, explicable, qui couvre raisonnablement le corpus
Morrigan actuel (4 docs : réseau, mythologie celtique, IA, projet).

Stratégie :
  - Entités : groupes de mots Capitalisés en milieu de phrase +
    acronymes en MAJ (TCP, HTTP, ADN, AGI…). On ignore les premiers
    mots de phrase capitalisés par convention (pour éviter d'extraire
    « Le » ou « La » comme entité).
  - Relations : 5 patterns simples — « X est un Y » → is_a,
    « X possède Y » → has, « X utilise Y » → uses, « X de Y » → of,
    plus la co-occurrence par phrase → co_occurs_with (relation
    faible, confidence 0.3).

Pas parfait, c'est attendu. PR 4 amortira les bruits en filtrant par
confiance + en privilégiant les relations vues plusieurs fois dans le
corpus (agrégation déjà gérée par `KnowledgeGraph.add_relation`).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Tuple

from modules.ogham.knowledge_graph import Entity, KnowledgeGraph, Relation

logger = logging.getLogger("morrigan.ogham.extractor")


# ─── Slugification d'id stables ────────────────────────────────────


def slugify(label: str) -> str:
    """Normalise un label en id stable (lowercase, sans accents, sans espaces).

    « Brigid » → « brigid ». « Empire Romain » → « empire_romain ».
    « TCP » → « tcp ». Volontairement basique — pour rester reproductible
    sans dep externe.
    """
    nfkd = unicodedata.normalize("NFKD", label.strip().lower())
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Remplace tout non-alphanumérique par underscore, puis dédoublonne.
    slugged = re.sub(r"[^a-z0-9]+", "_", no_accents).strip("_")
    return slugged or "unknown"


# ─── Découpage en phrases (basique) ────────────────────────────────


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ÿ])")


def split_sentences(text: str) -> List[str]:
    """Découpe en phrases sur `[.!?]` + espace + lettre majuscule.

    Pragmatique : rate les abréviations (« M. Dupont »), mais évite la
    sur-segmentation sur les listes à puces ou les nombres décimaux.
    Suffit pour le corpus Morrigan actuel.
    """
    if not text.strip():
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ─── Extraction d'entités ──────────────────────────────────────────


# Mots/groupes Capitalisés (incluant accentués) ou ACRONYMES.
# Évite les caractères ASCII restreints en utilisant la classe Unicode
# via les ranges courants du français.
_ENTITY_PATTERN = re.compile(
    r"\b("
    r"[A-ZÀ-Ÿ][a-zà-ÿ]+(?:[\s\-][A-ZÀ-Ÿ][a-zà-ÿ]+)*"  # Title Case (multi-mots OK)
    r"|[A-Z]{2,10}"                                      # Acronymes (TCP, HTTP, ADN…)
    r")\b"
)

# Stopwords qu'on rejette même s'ils sont capitalisés (faux positifs
# typiques en début de phrase, déterminants, conjonctions usuelles).
_STOPWORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "ce", "cette", "ces", "cet", "son", "sa", "ses", "leur", "leurs",
    "il", "elle", "ils", "elles", "on", "nous", "vous",
    "qui", "que", "quoi", "dont", "où",
    "tous", "toutes", "tout", "toute",
    "puis", "alors", "ainsi", "aussi", "encore",
    "voici", "voilà",
    "monsieur", "madame", "mademoiselle",
})


def extract_entities(text: str) -> List[Entity]:
    """Extrait les entités d'un texte. Renvoie une liste sans doublons (par id).

    Heuristique :
      - On scanne avec `_ENTITY_PATTERN`.
      - On exclut les matches dont le slug est dans `_STOPWORDS`.
      - Les acronymes (>=2 lettres MAJ) gardent le type "concept" mais
        leur label est conservé en MAJ d'origine.
    """
    seen: dict[str, Entity] = {}
    for match in _ENTITY_PATTERN.finditer(text):
        label = match.group(1).strip()
        slug = slugify(label)
        if slug in _STOPWORDS or len(slug) < 2:
            continue
        if slug in seen:
            continue
        # Type heuristique : acronyme MAJ pure → "acronym", sinon "concept".
        ent_type = "acronym" if label.isupper() and len(label) >= 2 else "concept"
        seen[slug] = Entity(id=slug, label=label, type=ent_type)
    return list(seen.values())


# ─── Extraction de relations ───────────────────────────────────────


# Patterns ordre = priorité (le premier qui match s'applique). Chaque
# pattern doit capturer (subject_label, object_label) — l'extracteur
# les slugifie ensuite.
_RELATION_PATTERNS: List[Tuple[re.Pattern[str], str, float]] = [
    # « X est un/une/le/la/des Y », « X sont Y »
    (re.compile(
        r"\b([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10})\s+"
        r"(?:est|sont|étaient|était)\s+(?:un|une|le|la|les|des|du|de\sla|de\sl')?\s*"
        r"([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10}|[a-zà-ÿ][\wà-ÿ\-]+)"
    ), "is_a", 0.8),
    # « X possède/contient Y »
    (re.compile(
        r"\b([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10})\s+"
        r"(?:possède|contient|inclut|comprend)\s+(?:un|une|le|la|les|des|du)?\s*"
        r"([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10}|[a-zà-ÿ][\wà-ÿ\-]+)"
    ), "has", 0.7),
    # « X utilise/emploie Y »
    (re.compile(
        r"\b([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10})\s+"
        r"(?:utilise|emploie|requiert|nécessite)\s+(?:un|une|le|la|les|des|du)?\s*"
        r"([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10}|[a-zà-ÿ][\wà-ÿ\-]+)"
    ), "uses", 0.7),
    # « X de Y » (relation de possession / appartenance, plus faible)
    (re.compile(
        r"\b([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10})\s+"
        r"(?:de|du|de\sla|de\sl')\s+"
        r"([A-ZÀ-Ÿ][\wÀ-ÿ\-]+(?:\s[A-ZÀ-Ÿ][\wÀ-ÿ\-]+)*|[A-Z]{2,10})"
    ), "of", 0.5),
]


# Confidence des relations de co-occurrence : faible par design, mais
# non nulle — l'agrégation `add_relation` montera la confidence si la
# co-occurrence est observée plusieurs fois dans le corpus.
_CO_OCCURRENCE_CONFIDENCE = 0.3


def extract_relations(
    text: str, source: Optional[str] = None
) -> List[Relation]:
    """Extrait les relations d'un texte.

    Deux passes :
      1. Patterns explicites (`_RELATION_PATTERNS`) → predicate typés.
      2. Co-occurrence par phrase → predicate `co_occurs_with`,
         confidence faible (0.3). Utile pour bootstrap quand les
         patterns explicites ne matchent pas.
    """
    out: List[Relation] = []
    seen: set[Tuple[str, str, str]] = set()

    # --- Passe 1 : patterns explicites
    for pattern, predicate, conf in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            s_label, o_label = match.group(1).strip(), match.group(2).strip()
            s_id, o_id = slugify(s_label), slugify(o_label)
            if s_id == o_id or s_id in _STOPWORDS or o_id in _STOPWORDS:
                continue
            key = (s_id, predicate, o_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(Relation(
                subject_id=s_id, predicate=predicate, object_id=o_id,
                confidence=conf, source=source,
            ))

    # --- Passe 2 : co-occurrence dans la même phrase
    for sentence in split_sentences(text):
        entities = extract_entities(sentence)
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                if a.id == b.id:
                    continue
                key = (a.id, "co_occurs_with", b.id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Relation(
                    subject_id=a.id, predicate="co_occurs_with",
                    object_id=b.id, confidence=_CO_OCCURRENCE_CONFIDENCE,
                    source=source,
                ))

    return out


# ─── Ingestion dans un graphe ──────────────────────────────────────


def populate_graph(
    kg: KnowledgeGraph,
    text: str,
    source: Optional[str] = None,
) -> Tuple[int, int]:
    """Extrait + ingère dans `kg`. Renvoie (n_entities_added, n_relations_added).

    Les entités sont ajoutées **avant** les relations, pour que les
    nodes du KG aient le bon label et le bon type (sinon
    `add_relation` créerait des nodes par défaut avec label = id).
    """
    entities = extract_entities(text)
    relations = extract_relations(text, source=source)
    for entity in entities:
        kg.add_entity(entity)
    for relation in relations:
        kg.add_relation(relation)
    logger.debug(
        "Extraction depuis '%s' : %d entités, %d relations",
        source or "(no source)", len(entities), len(relations),
    )
    return len(entities), len(relations)
