"""Tests du script `scripts/build_knowledge_graph.py` (PR 3 du chantier KG).

On appelle `main(argv=...)` directement plutôt que via subprocess —
plus rapide, plus simple à débugger, et l'effet de bord (KG sauvé sur
disque) est testable au même niveau.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from modules.ogham.knowledge_graph import KnowledgeGraph
from scripts.build_knowledge_graph import _clean_markdown, main


# ─── _clean_markdown ──────────────────────────────────────────────


def test_clean_strips_headers():
    out = _clean_markdown("# Titre principal\n## Sous-titre\nContenu.")
    assert "Titre principal" in out
    assert "Sous-titre" in out
    assert "#" not in out


def test_clean_strips_bullets():
    out = _clean_markdown("- Premier item\n- Deuxième item")
    assert "Premier item" in out
    assert out.lstrip()[0] != "-"


def test_clean_drops_code_fences():
    """Le contenu d'un fence ```` ``` ```` ne doit pas finir dans le texte."""
    md = "Texte avant.\n```python\nprint('secret')\n```\nTexte après."
    out = _clean_markdown(md)
    assert "Texte avant" in out
    assert "Texte après" in out
    assert "secret" not in out


def test_clean_flattens_links():
    out = _clean_markdown("Voir [Wikipedia](https://wikipedia.org) pour plus.")
    assert "Wikipedia" in out
    assert "https://" not in out
    assert "(" not in out  # parens autour de l'url retirées


def test_clean_strips_inline_code():
    out = _clean_markdown("La fonction `print` affiche.")
    assert "print" in out
    assert "`" not in out


# ─── main() — bout en bout sur un mini-corpus ──────────────────────


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """Crée un mini-corpus 2 docs sous un tmp_path."""
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "reseau.md").write_text(
        "# Réseaux\n\n"
        "TCP est un protocole fiable. UDP est un protocole rapide.\n"
        "TCP et UDP utilisent IP.\n",
        encoding="utf-8",
    )
    (src / "celtique.md").write_text(
        "## Divinités\n\n"
        "Brigid est une déesse celtique. Cúchulainn est un héros.\n"
        "Brigid et Dagda appartiennent aux Tuatha Dé Danann.\n",
        encoding="utf-8",
    )
    return src


def test_main_produces_kg_json(mini_corpus: Path, tmp_path: Path):
    out = tmp_path / "kg.json"
    rc = main([
        "--source", str(mini_corpus),
        "--output", str(out),
        "--min-entities", "3",
        "--min-relations", "3",
    ])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert len(data["entities"]) >= 3
    assert len(data["edges"]) >= 1


def test_main_kg_is_reloadable(mini_corpus: Path, tmp_path: Path):
    out = tmp_path / "kg.json"
    main([
        "--source", str(mini_corpus),
        "--output", str(out),
        "--min-entities", "3",
        "--min-relations", "3",
    ])
    kg = KnowledgeGraph.load(out)
    assert "tcp" in kg
    assert "udp" in kg
    assert "brigid" in kg


def test_main_fails_on_min_entities(mini_corpus: Path, tmp_path: Path):
    out = tmp_path / "kg.json"
    rc = main([
        "--source", str(mini_corpus),
        "--output", str(out),
        "--min-entities", "9999",  # impossible à atteindre
        "--min-relations", "1",
    ])
    assert rc == 1  # garde-fou CI


def test_main_fails_on_missing_source(tmp_path: Path):
    out = tmp_path / "kg.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--source", str(tmp_path / "absent"),
            "--output", str(out),
        ])
    assert "introuvable" in str(exc_info.value)


def test_main_fails_on_empty_source(tmp_path: Path):
    """Source existe mais aucun fichier ne matche les patterns."""
    src = tmp_path / "empty"
    src.mkdir()
    (src / "ignored.py").write_text("# pas markdown")
    out = tmp_path / "kg.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--source", str(src),
            "--output", str(out),
        ])
    assert "Aucun fichier" in str(exc_info.value)


def test_main_respects_custom_patterns(tmp_path: Path):
    """--patterns permet d'ingérer un autre format que .md/.txt."""
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "data.rst").write_text(
        "Brigid est une déesse. Dagda est un dieu.\n",
        encoding="utf-8",
    )
    out = tmp_path / "kg.json"
    rc = main([
        "--source", str(src),
        "--output", str(out),
        "--patterns", "*.rst",
        "--min-entities", "2",
        "--min-relations", "1",
    ])
    assert rc == 0
    kg = KnowledgeGraph.load(out)
    assert "brigid" in kg
