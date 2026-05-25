"""Tests du script `scripts/ingest_wikipedia.py` (Phase 5 — ingestion à l'échelle).

On appelle `main(argv=...)` directement (comme `test_build_knowledge_graph`),
et on **monkeypatch `_iter_articles`** pour ne dépendre ni du réseau ni du
dataset `datasets` : les tests bout-en-bout fournissent de faux articles.
Le filtre des articles courts et la marge `islice` sont testés à part en
injectant un faux module `datasets`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from modules.danann.store import Danann
from scripts.ingest_wikipedia import MIN_ARTICLE_CHARS, _iter_articles, main


def _fake_articles(*titles: str):
    """Fabrique un faux `_iter_articles(config, max_articles)`.

    Ignore `max_articles` (c'est `main` qui borne via son `break`) ; chaque
    article a un texte assez long pour produire au moins un chunk.
    """

    def _iter(config: str, max_articles: int):
        for title in titles:
            yield title, f"{title} est un sujet encyclopédique. " * 8

    return _iter


# ─── main() — bout en bout sur de faux articles ────────────────────


def test_main_indexes_and_saves(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "scripts.ingest_wikipedia._iter_articles",
        _fake_articles("Brigid", "Cúchulainn", "Dagda"),
    )
    out = tmp_path / "index_wiki"
    rc = main(["--max-articles", "3", "--output", str(out)])

    assert rc == 0
    assert (out / "corpus.json").exists()
    assert (out / "vectors.npz").exists()

    # Rechargeable sans réembedder, contenu cohérent.
    d = Danann.load_index(out)
    assert d.count() > 0
    assert d.compression == "int8"  # défaut du script


def test_metadata_tags_domain_and_source(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "scripts.ingest_wikipedia._iter_articles",
        _fake_articles("Brigid", "Dagda"),
    )
    out = tmp_path / "index_wiki"
    main(["--max-articles", "2", "--output", str(out)])

    d = Danann.load_index(out)
    assert all(m["domain"] == "wikipedia" for m in d.metadata)
    assert set(m["source"] for m in d.metadata) <= {"Brigid", "Dagda"}


def test_main_respects_max_articles(monkeypatch, tmp_path: Path):
    """`--max-articles` borne la consommation, même si la source en a plus."""
    consumed: list[str] = []

    def _iter(config: str, max_articles: int):
        for i in range(10):
            consumed.append(f"A{i}")
            yield f"A{i}", "Phrase encyclopédique de test. " * 8

    monkeypatch.setattr("scripts.ingest_wikipedia._iter_articles", _iter)
    out = tmp_path / "index_wiki"
    main(["--max-articles", "2", "--output", str(out)])

    assert consumed == ["A0", "A1"]  # arrêt net au 2e article


def test_main_supports_compression_choice(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "scripts.ingest_wikipedia._iter_articles",
        _fake_articles("Brigid", "Dagda"),
    )
    out = tmp_path / "index_wiki"
    rc = main(["--max-articles", "2", "--compression", "binary", "--output", str(out)])

    assert rc == 0
    assert Danann.load_index(out).compression == "binary"


# ─── tolérance aux pannes (try/except de main) ─────────────────────


def test_partial_save_on_error(monkeypatch, tmp_path: Path):
    """Une erreur en cours d'ingestion sauve ce qui a déjà été indexé."""

    def _iter(config: str, max_articles: int):
        yield "Brigid", "Brigid est une déesse celtique majeure. " * 8
        raise RuntimeError("réseau coupé")

    monkeypatch.setattr("scripts.ingest_wikipedia._iter_articles", _iter)
    out = tmp_path / "index_wiki"
    rc = main(["--max-articles", "5", "--output", str(out)])

    assert rc == 0
    assert Danann.load_index(out).count() > 0


def test_exit_when_nothing_ingested(monkeypatch, tmp_path: Path):
    """Erreur avant tout chunk → sortie non nulle (rien à sauver)."""

    def _iter(config: str, max_articles: int):
        raise RuntimeError("dataset introuvable")
        yield  # pragma: no cover — rend la fonction génératrice

    monkeypatch.setattr("scripts.ingest_wikipedia._iter_articles", _iter)
    with pytest.raises(SystemExit) as exc_info:
        main(["--max-articles", "5", "--output", str(tmp_path / "index_wiki")])
    assert "Aucun chunk" in str(exc_info.value)


# ─── _iter_articles — filtre + marge islice (faux datasets) ────────


def _inject_fake_datasets(monkeypatch, rows: list[dict]):
    """Injecte un faux module `datasets` dont `load_dataset` rend `rows`."""

    def fake_load_dataset(name, config, split="train", streaming=False):
        assert streaming is True
        return list(rows)

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


def test_iter_articles_filters_short(monkeypatch):
    _inject_fake_datasets(monkeypatch, [
        {"title": "Long A", "text": "x" * (MIN_ARTICLE_CHARS + 10)},
        {"title": "Court", "text": "y" * (MIN_ARTICLE_CHARS - 10)},
        {"title": "Long B", "text": "z" * (MIN_ARTICLE_CHARS + 10)},
    ])
    titles = [t for t, _ in _iter_articles("20231101.fr", max_articles=10)]
    assert titles == ["Long A", "Long B"]  # l'article court est filtré


def test_iter_articles_islice_margin(monkeypatch):
    """La marge `max_articles * 3` borne le nombre de lignes examinées."""
    rows = [{"title": f"A{i}", "text": "x" * (MIN_ARTICLE_CHARS + 10)} for i in range(10)]
    _inject_fake_datasets(monkeypatch, rows)
    out = list(_iter_articles("20231101.fr", max_articles=1))
    assert len(out) == 3  # 1 * 3, pas les 10 lignes
