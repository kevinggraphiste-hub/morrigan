"""
BRIGID — Chargement du dataset d'entraînement du classifieur d'intention.

Format : un fichier JSONL où chaque ligne est `{"query": str, "label": str}`.
Le label doit être l'une des valeurs canoniques de `QueryType` côté code
(en chaîne, sans le préfixe `QueryType.`).

Source de vérité de la liste des labels : `LABELS` ci-dessous. C'est aussi
ce qui détermine l'ordre des logits du modèle (`label_to_id` est utilisé
en train et en inférence pour garder l'alignement).

UNKNOWN volontairement absent du dataset : c'est un état de sortie quand
le modèle n'est pas confiant, pas une classe à apprendre.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# Ordre canonique des classes — gèle la correspondance label ↔ id pour
# que les checkpoints restent compatibles entre runs. NE PAS réordonner.
LABELS: Tuple[str, ...] = (
    "factual",
    "reasoning",
    "creative",
    "conversation",
    "complex",
    "code",
)

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# Chemin par défaut du dataset livré avec le repo.
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "training"
    / "intent_classification.jsonl"
)


@dataclass(frozen=True)
class LabeledExample:
    """Un exemple étiqueté pour le classifieur Brigid."""

    query: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABEL_TO_ID:
            raise ValueError(
                f"Label inconnu : {self.label!r} (attendus : {LABELS})"
            )
        if not self.query.strip():
            raise ValueError("query vide")

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> List[LabeledExample]:
    """Lit le JSONL et renvoie la liste des exemples étiquetés.

    Lève `FileNotFoundError` si le fichier manque, et `ValueError` (via
    `LabeledExample.__post_init__`) sur la première ligne malformée —
    on ne tolère pas un dataset partiellement corrompu.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}. "
            "Vérifie le chemin ou regénère le fichier."
        )

    examples: List[LabeledExample] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Ligne {line_no} : JSON invalide ({e.msg})"
                ) from e
            if "query" not in obj or "label" not in obj:
                raise ValueError(
                    f"Ligne {line_no} : clés 'query' et 'label' requises"
                )
            examples.append(LabeledExample(query=obj["query"], label=obj["label"]))

    return examples


def class_balance(examples: Iterable[LabeledExample]) -> Counter:
    """Compte les exemples par classe — utile pour vérifier l'équilibre."""
    return Counter(ex.label for ex in examples)


def split_train_val(
    examples: List[LabeledExample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[LabeledExample], List[LabeledExample]]:
    """Split train/val *stratifié par classe* et déterministe.

    Stratifié : on tire `val_ratio` de chaque classe séparément pour
    garantir que chaque classe est représentée dans le split val (sinon
    la métrique d'accuracy de validation devient trompeuse).

    Déterministe : même seed → même split. Permet de comparer deux runs
    d'entraînement sans confusion liée au shuffle.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio doit être dans ]0, 1[, reçu {val_ratio}")

    rng = random.Random(seed)
    by_label: dict[str, List[LabeledExample]] = {label: [] for label in LABELS}
    for ex in examples:
        by_label[ex.label].append(ex)

    train: List[LabeledExample] = []
    val: List[LabeledExample] = []
    for label, items in by_label.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    # Re-shuffle global pour ne pas laisser un ordre par classe.
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
