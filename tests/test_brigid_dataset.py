"""Tests du dataset d'entraînement Brigid + loader/split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from modules.brigid.dataset import (
    DEFAULT_DATASET_PATH,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LABELS,
    LabeledExample,
    class_balance,
    load_dataset,
    split_train_val,
)


# ─── Constantes / contrat ──────────────────────────────────────────


def test_labels_match_query_type():
    """Les LABELS doivent correspondre à un sous-ensemble de QueryType."""
    from core.types import QueryType

    qt_values = {qt.value for qt in QueryType}
    for label in LABELS:
        assert label in qt_values, f"{label!r} absent de QueryType"


def test_label_to_id_is_bijective():
    assert len(LABEL_TO_ID) == len(LABELS)
    assert len(ID_TO_LABEL) == len(LABELS)
    for label in LABELS:
        assert ID_TO_LABEL[LABEL_TO_ID[label]] == label


# ─── LabeledExample ────────────────────────────────────────────────


def test_labeled_example_rejects_unknown_label():
    with pytest.raises(ValueError, match="Label inconnu"):
        LabeledExample(query="hello", label="bogus")


def test_labeled_example_rejects_empty_query():
    with pytest.raises(ValueError, match="vide"):
        LabeledExample(query="   ", label="conversation")


def test_labeled_example_label_id():
    ex = LabeledExample(query="Salut", label="conversation")
    assert ex.label_id == LABEL_TO_ID["conversation"]


# ─── load_dataset ──────────────────────────────────────────────────


def test_default_dataset_loads_and_is_balanced():
    """Le dataset livré doit charger et être raisonnablement équilibré."""
    examples = load_dataset()
    assert len(examples) >= 200, (
        f"Dataset trop petit ({len(examples)} < 200) — la roadmap demande "
        "200 à 500 exemples."
    )

    balance = class_balance(examples)
    # Toutes les classes présentes
    for label in LABELS:
        assert balance[label] > 0, f"Classe {label!r} absente du dataset"
    # Pas de classe trop sur-représentée (≤ 2× la plus petite)
    counts = list(balance.values())
    assert max(counts) <= 2 * min(counts), (
        f"Déséquilibre fort : {dict(balance)}"
    )


def test_load_dataset_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.jsonl")


def test_load_dataset_invalid_json(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "ok", "label": "factual"}\n{not json}\n')
    with pytest.raises(ValueError, match="JSON invalide"):
        load_dataset(bad)


def test_load_dataset_missing_keys(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "no label here"}\n')
    with pytest.raises(ValueError, match="requises"):
        load_dataset(bad)


def test_load_dataset_skips_blank_lines(tmp_path: Path):
    f = tmp_path / "ok.jsonl"
    f.write_text(
        '{"query": "a", "label": "factual"}\n'
        "\n"
        '{"query": "b", "label": "code"}\n'
    )
    examples = load_dataset(f)
    assert len(examples) == 2


# ─── split_train_val ───────────────────────────────────────────────


def test_split_is_stratified_and_covers_all_classes():
    examples = load_dataset()
    train, val = split_train_val(examples, val_ratio=0.2, seed=42)

    # Sizes plausibles
    assert len(train) + len(val) == len(examples)
    assert len(val) > 0
    assert len(train) > len(val)

    # Toutes les classes présentes dans les deux splits
    train_classes = {ex.label for ex in train}
    val_classes = {ex.label for ex in val}
    for label in LABELS:
        assert label in train_classes, f"Classe {label!r} absente du train"
        assert label in val_classes, f"Classe {label!r} absente du val"


def test_split_is_deterministic():
    examples = load_dataset()
    a_train, a_val = split_train_val(examples, val_ratio=0.2, seed=123)
    b_train, b_val = split_train_val(examples, val_ratio=0.2, seed=123)
    assert [e.query for e in a_train] == [e.query for e in b_train]
    assert [e.query for e in a_val] == [e.query for e in b_val]


def test_split_different_seed_gives_different_order():
    examples = load_dataset()
    a_train, _ = split_train_val(examples, val_ratio=0.2, seed=1)
    b_train, _ = split_train_val(examples, val_ratio=0.2, seed=2)
    assert [e.query for e in a_train] != [e.query for e in b_train]


def test_split_invalid_ratio():
    with pytest.raises(ValueError):
        split_train_val([], val_ratio=0.0)
    with pytest.raises(ValueError):
        split_train_val([], val_ratio=1.0)


# ─── Path du dataset ───────────────────────────────────────────────


def test_default_dataset_path_resolves():
    assert DEFAULT_DATASET_PATH.exists(), (
        f"Dataset par défaut absent : {DEFAULT_DATASET_PATH}"
    )
    # Sanity : le fichier est bien du JSONL.
    first = DEFAULT_DATASET_PATH.read_text().splitlines()[0]
    obj = json.loads(first)
    assert "query" in obj and "label" in obj
