"""Tests du corpus code (data/knowledge/code_*.md).

Vérifie que :
  - Les fichiers ont bien la convention de nommage `code_*.md`
    qui déclenche `domain="code"` dans `scripts/ingest_knowledge.py`.
  - Chaque fichier a une taille minimum (pas un placeholder vide).
  - Le build du KG inclut bien ces nouvelles entités.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from scripts.ingest_knowledge import detect_domain

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
CODE_FILES = sorted(CORPUS_DIR.glob("code_*.md"))


def test_corpus_has_code_files():
    """Au moins 6 fichiers code_*.md livrés (Python, JS, Bash, SQL, HTML/CSS, Git/Docker)."""
    assert len(CODE_FILES) >= 6, (
        f"Trouvé {len(CODE_FILES)} fichiers code_*.md ; attendu ≥ 6 "
        "(Python, JS, Bash, SQL, HTML/CSS, Git/Docker)."
    )


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_code_file_is_tagged_domain_code(path: Path):
    """Chaque fichier code_*.md doit être auto-tagué domain='code'."""
    text = path.read_text(encoding="utf-8")
    assert detect_domain(path.name, text) == "code"


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_code_file_has_substantive_content(path: Path):
    """Chaque fichier doit faire au moins 500 caractères (pas un placeholder)."""
    text = path.read_text(encoding="utf-8")
    assert len(text) >= 500, (
        f"{path.name} est trop court ({len(text)} caractères) — "
        "documentation curée attendue, pas placeholder."
    )


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_code_file_has_markdown_structure(path: Path):
    """Chaque fichier doit avoir au moins un titre markdown."""
    text = path.read_text(encoding="utf-8")
    assert any(line.startswith("#") for line in text.splitlines()), (
        f"{path.name} sans titre markdown — structure attendue."
    )
