"""Tests de `core.knowledge.build_danann` (Phase 5 — index persisté au runtime).

`build_danann` est le point d'entrée partagé par le CLI et le bot : il
charge un index compressé persisté si `MORRIGAN_INDEX`/`index_path` est
valide, sinon il ingère `data/knowledge`. On vérifie les deux branches et
la dégradation gracieuse, avec des dossiers temporaires (zéro pollution).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")

from core.knowledge import INDEX_ENV, build_danann
from modules.danann.store import Danann


def _write_corpus(dir_: Path) -> Path:
    """Mini corpus markdown sous un tmp_path → renvoie le dossier."""
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "celtique.md").write_text(
        "# Mythologie\n\n"
        "Brigid est une déesse celtique. Dagda est un dieu majeur.\n"
        "Les Tuatha Dé Danann sont un peuple mythique.\n",
        encoding="utf-8",
    )
    return dir_


def _build_saved_index(dir_: Path) -> int:
    """Construit puis sauve un index int8 dans `dir_`. Renvoie le count."""
    d = Danann(compression="int8", use_reranker=False)
    d.index(
        ["Brigid est une déesse celtique.", "Dagda est un dieu majeur."],
        [{"domain": "test"}, {"domain": "test"}],
    )
    d.save_index(dir_)
    return d.count()


# ─── Branche ingestion (pas d'index) ───────────────────────────────


def test_ingests_knowledge_dir_when_no_index(tmp_path: Path):
    kdir = _write_corpus(tmp_path / "knowledge")
    danann = build_danann(knowledge_dir=kdir, use_reranker=False)
    assert danann.count() > 0


def test_empty_when_knowledge_dir_missing(tmp_path: Path):
    """Pas d'index, dossier corpus absent → Danann vide, pas de crash."""
    danann = build_danann(knowledge_dir=tmp_path / "absent", use_reranker=False)
    assert danann.count() == 0


# ─── Branche index persisté ────────────────────────────────────────


def test_loads_persisted_index_via_arg(tmp_path: Path):
    idx = tmp_path / "index"
    saved = _build_saved_index(idx)
    # knowledge_dir pointe vers un corpus *différent* : il ne doit PAS
    # être ingéré quand un index valide est fourni.
    kdir = _write_corpus(tmp_path / "knowledge")

    danann = build_danann(
        knowledge_dir=kdir, index_path=str(idx), use_reranker=False
    )
    assert danann.count() == saved
    assert danann.compression == "int8"


def test_loads_persisted_index_via_env(tmp_path: Path, monkeypatch):
    idx = tmp_path / "index"
    saved = _build_saved_index(idx)
    monkeypatch.setenv(INDEX_ENV, str(idx))

    danann = build_danann(
        knowledge_dir=tmp_path / "absent", use_reranker=False
    )
    assert danann.count() == saved


def test_arg_overrides_env(tmp_path: Path, monkeypatch):
    """`index_path` explicite l'emporte sur MORRIGAN_INDEX."""
    idx = tmp_path / "index"
    _build_saved_index(idx)
    monkeypatch.setenv(INDEX_ENV, str(tmp_path / "env_invalide"))

    danann = build_danann(
        knowledge_dir=tmp_path / "absent",
        index_path=str(idx),
        use_reranker=False,
    )
    assert danann.count() == 2


# ─── Dégradation gracieuse (index invalide) ────────────────────────


def test_invalid_index_falls_back_to_ingestion(tmp_path: Path, caplog):
    bad = tmp_path / "pas_un_index"
    bad.mkdir()  # existe mais sans corpus.json / vectors.npz
    kdir = _write_corpus(tmp_path / "knowledge")

    with caplog.at_level(logging.WARNING, logger="morrigan.knowledge"):
        danann = build_danann(
            knowledge_dir=kdir, index_path=str(bad), use_reranker=False
        )
    assert danann.count() > 0  # corpus ingéré en fallback
    assert any("invalide" in r.message for r in caplog.records)
