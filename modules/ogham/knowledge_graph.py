"""
OGHAM — Knowledge Graph.

Représentation de la connaissance Morrigan sous forme de graphe orienté
d'**entités** reliées par des **relations** typées. Stockage en mémoire
via `networkx.DiGraph` (déjà dans les deps), sérialisable en JSON pour
persistance simple.

Brique fondationale (PR 1 du chantier KG). N'extrait rien (PR 2),
n'ingère rien (PR 3), n'est pas encore branché dans Ogham (PR 4) — juste
le modèle de données + l'API de requête.

Choix d'archi :
  - `networkx.DiGraph` plutôt que MultiDiGraph : on agrège les relations
    duplicates (mêmes (s, p, o)) en augmentant un compteur — plus stable
    pour les requêtes et les diffs. Si besoin de plusieurs occurrences
    distinctes, passer en MultiDiGraph + relation_id.
  - JSON plutôt que pickle pour la persistance : lisible, diff-able en
    review, compatible cross-version Python.
  - Pas de schéma rigide pour les `attributes` : c'est le rôle de la
    couche extraction (PR 2) de produire des attributs cohérents.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("morrigan.ogham.kg")


# ─── Modèle de données ─────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    """Une entité du graphe.

    `id` est la clé unique (souvent un slug normalisé du label).
    `label` est la forme d'affichage (peut contenir accents/maj).
    `type` est un tag libre ("person", "protocol", "concept", …). On
    ne fige pas la taxonomie à ce stade — PR 2/3 décideront.
    """

    id: str
    label: str
    type: str = "concept"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Entity.id ne peut être vide")
        if not self.label.strip():
            raise ValueError("Entity.label ne peut être vide")


@dataclass(frozen=True)
class Relation:
    """Une relation orientée subject → object, typée par `predicate`.

    `confidence` est un float dans [0, 1] (1 = certitude, 0 = supposition
    faible). Source est l'identifiant du document/chunk d'où la relation
    a été extraite — utile pour traçabilité dans les réponses Ogham.
    """

    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Relation.confidence doit être dans [0, 1], reçu {self.confidence}"
            )
        if not self.predicate.strip():
            raise ValueError("Relation.predicate ne peut être vide")


# ─── Knowledge Graph ───────────────────────────────────────────────


def _entity_to_node(entity: Entity) -> Dict[str, Any]:
    """Convertit une `Entity` en dict d'attributs networkx."""
    return {
        "label": entity.label,
        "type": entity.type,
        "attributes": dict(entity.attributes),
    }


def _node_to_entity(entity_id: str, data: Dict[str, Any]) -> Entity:
    return Entity(
        id=entity_id,
        label=data.get("label", entity_id),
        type=data.get("type", "concept"),
        attributes=dict(data.get("attributes", {})),
    )


class KnowledgeGraph:
    """Graphe orienté d'entités + relations.

    Gère le merge des doublons :
      - `add_entity` sur un id déjà présent fusionne les `attributes`
        (les nouveaux écrasent les anciens sur les clés communes).
      - `add_relation` sur un triplet (s, p, o) déjà présent incrémente
        un compteur `count` et garde la confidence max ; les sources
        s'accumulent dans une liste.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ─── Mutation ──────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        """Ajoute (ou fusionne) une entité dans le graphe."""
        if entity.id in self._graph:
            existing = self._graph.nodes[entity.id]
            existing["label"] = entity.label  # preference au plus récent
            existing["type"] = entity.type
            merged = dict(existing.get("attributes", {}))
            merged.update(entity.attributes)
            existing["attributes"] = merged
            return _node_to_entity(entity.id, existing)
        self._graph.add_node(entity.id, **_entity_to_node(entity))
        return entity

    def add_relation(self, relation: Relation) -> Relation:
        """Ajoute (ou agrège) une relation. Crée les nodes si absents.

        Note : on **n'invente pas de label/type** pour les entités créées
        à la volée par add_relation — on prend l'id comme label, type
        "concept". L'extracteur (PR 2) doit `add_entity` *avant*
        d'`add_relation` pour récupérer des métadonnées riches.
        """
        for ent_id in (relation.subject_id, relation.object_id):
            if ent_id not in self._graph:
                self.add_entity(Entity(id=ent_id, label=ent_id))

        edge_key = (relation.subject_id, relation.object_id)
        if self._graph.has_edge(*edge_key):
            edge = self._graph.edges[edge_key]
            preds: Dict[str, Dict[str, Any]] = edge.setdefault("predicates", {})
        else:
            preds = {}
            self._graph.add_edge(*edge_key, predicates=preds)

        pred = preds.get(relation.predicate)
        if pred is None:
            preds[relation.predicate] = {
                "count": 1,
                "confidence": relation.confidence,
                "sources": [relation.source] if relation.source else [],
            }
        else:
            pred["count"] += 1
            pred["confidence"] = max(pred["confidence"], relation.confidence)
            if relation.source and relation.source not in pred["sources"]:
                pred["sources"].append(relation.source)

        return relation

    # ─── Lecture / requêtes ───────────────────────────────────

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._graph

    @property
    def relation_count(self) -> int:
        """Nombre total de triplets distincts (s, p, o)."""
        return sum(
            len(d.get("predicates", {})) for _, _, d in self._graph.edges(data=True)
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if entity_id not in self._graph:
            return None
        return _node_to_entity(entity_id, self._graph.nodes[entity_id])

    def entities(self, type: Optional[str] = None) -> List[Entity]:
        """Liste toutes les entités, éventuellement filtrées par type."""
        out: List[Entity] = []
        for nid, data in self._graph.nodes(data=True):
            ent = _node_to_entity(nid, data)
            if type is None or ent.type == type:
                out.append(ent)
        return out

    def relations(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> List[Relation]:
        """Liste les relations, filtrables par tout sous-ensemble du triplet."""
        out: List[Relation] = []
        for s, o, data in self._graph.edges(data=True):
            if subject_id is not None and s != subject_id:
                continue
            if object_id is not None and o != object_id:
                continue
            for pred, info in data.get("predicates", {}).items():
                if predicate is not None and pred != predicate:
                    continue
                sources = info.get("sources") or []
                out.append(Relation(
                    subject_id=s,
                    predicate=pred,
                    object_id=o,
                    confidence=info.get("confidence", 1.0),
                    source=sources[0] if sources else None,
                ))
        return out

    def neighbors(
        self, entity_id: str, predicate: Optional[str] = None
    ) -> List[Tuple[Entity, str]]:
        """Voisins sortants d'une entité, avec le prédicat utilisé.

        Renvoie `[(Entity, predicate), …]`. Si l'entité n'existe pas,
        renvoie une liste vide (pas d'erreur — le call site décide).
        """
        if entity_id not in self._graph:
            return []
        result: List[Tuple[Entity, str]] = []
        for _, neighbor_id, data in self._graph.out_edges(entity_id, data=True):
            for pred in data.get("predicates", {}):
                if predicate is not None and pred != predicate:
                    continue
                neighbor = _node_to_entity(neighbor_id, self._graph.nodes[neighbor_id])
                result.append((neighbor, pred))
        return result

    def facts_about(self, entity_id: str) -> List[Relation]:
        """Toutes les relations impliquant l'entité (sortantes + entrantes)."""
        out = self.relations(subject_id=entity_id)
        out.extend(self.relations(object_id=entity_id))
        return out

    def compare(self, a_id: str, b_id: str) -> Dict[str, Any]:
        """Compare deux entités. Renvoie points communs et différences.

        Structure : `{
            "common_neighbors": [(Entity, [predicates])],
            "a_only": [(Entity, predicate)],
            "b_only": [(Entity, predicate)],
            "direct_relations": [Relation],  # a → b ou b → a
        }`

        Utile pour Ogham : « Compare TCP et UDP » → réponse structurée.
        """
        a_neighbors = self.neighbors(a_id)
        b_neighbors = self.neighbors(b_id)

        a_set = {n.id for n, _ in a_neighbors}
        b_set = {n.id for n, _ in b_neighbors}

        common_ids = a_set & b_set
        common: List[Tuple[Entity, List[str]]] = []
        for nid in common_ids:
            preds_a = [p for n, p in a_neighbors if n.id == nid]
            preds_b = [p for n, p in b_neighbors if n.id == nid]
            ent = self.get_entity(nid)
            assert ent is not None  # garanti par appartenance au graphe
            common.append((ent, sorted(set(preds_a) | set(preds_b))))

        a_only = [(n, p) for n, p in a_neighbors if n.id not in common_ids]
        b_only = [(n, p) for n, p in b_neighbors if n.id not in common_ids]

        direct = [
            r for r in (
                self.relations(subject_id=a_id, object_id=b_id)
                + self.relations(subject_id=b_id, object_id=a_id)
            )
        ]

        return {
            "common_neighbors": common,
            "a_only": a_only,
            "b_only": b_only,
            "direct_relations": direct,
        }

    # ─── Persistance ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le graphe en dict (prêt pour `json.dumps`)."""
        return {
            "schema_version": 1,
            "entities": [
                {"id": nid, **_entity_to_node(_node_to_entity(nid, data))}
                for nid, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {
                    "subject": s,
                    "object": o,
                    "predicates": dict(data.get("predicates", {})),
                }
                for s, o, data in self._graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KnowledgeGraph":
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Schema version inconnu : {payload.get('schema_version')!r}"
            )
        kg = cls()
        for ent in payload.get("entities", []):
            kg.add_entity(Entity(
                id=ent["id"],
                label=ent.get("label", ent["id"]),
                type=ent.get("type", "concept"),
                attributes=dict(ent.get("attributes", {})),
            ))
        for edge in payload.get("edges", []):
            s, o = edge["subject"], edge["object"]
            for pred, info in edge.get("predicates", {}).items():
                # info peut contenir count + confidence + sources
                sources = info.get("sources") or []
                conf = info.get("confidence", 1.0)
                # On re-applique add_relation count fois pour préserver
                # le compteur (chaque appel incrémente count de 1).
                for i in range(info.get("count", 1)):
                    src = sources[i] if i < len(sources) else None
                    kg.add_relation(Relation(
                        subject_id=s,
                        predicate=pred,
                        object_id=o,
                        confidence=conf,
                        source=src,
                    ))
        return kg

    def save(self, path: Path) -> Path:
        """Persiste en JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "KG sauvé : %d entités, %d triplets → %s",
            len(self), self.relation_count, path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        if not path.exists():
            raise FileNotFoundError(
                f"KG introuvable : {path}. "
                "Lance `python scripts/build_knowledge_graph.py` (PR 3 à venir)."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
