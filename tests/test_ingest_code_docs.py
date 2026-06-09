"""Tests Phase 2B — ingestion du corpus code (scripts/ingest_code_docs.py).

CI-safe : aucun réseau, aucun gros download. On teste le chunker code-aware,
le parsing du bundle (fixture tmp), et l'introspection pydoc (stdlib).
L'ingestion réelle dans Danann (qui tire e5) n'est pas testée ici — couverte
par test_multilingual_retrieval.py côté retrieval.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import shutil

import pytest

from scripts.ingest_code_docs import (
    _BACKSPACE_RE,
    _hard_split,
    chunk_code_doc,
    extract_pydoc,
    iter_bundle_docs,
    iter_source,
    man_language,
    render_man,
)


# ─── Chunker : préservation du code ───────────────────────────────────


def test_chunk_preserves_indentation_and_newlines():
    text = (
        "Exemple\n"
        "=======\n"
        "Voici une fonction :\n"
        "\n"
        "    def add(a, b):\n"
        "        return a + b\n"
    )
    chunks = chunk_code_doc(text)
    joined = "\n".join(c for c, _ in chunks)
    # Indentation du corps de fonction conservée (le chunker générique l'écrasait).
    assert "    def add(a, b):" in joined
    assert "        return a + b" in joined


def test_chunk_attaches_section_title():
    text = "Les listes\n==========\nUne liste est une séquence ordonnée et mutable."
    chunks = chunk_code_doc(text)
    assert chunks
    assert chunks[0][1] == "Les listes"


def test_chunk_respects_max_chars():
    # Beaucoup de paragraphes courts → plusieurs chunks bornés.
    paras = "\n\n".join(f"Paragraphe numéro {i} avec un peu de texte." for i in range(100))
    chunks = chunk_code_doc(paras, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c, _ in chunks)


def test_chunk_keeps_code_block_intact():
    code = (
        "    for index, value in enumerate(items):\n"
        "        print(index, value)\n"
        "        total += value"
    )
    text = f"Boucle\n======\n{code}"
    chunks = chunk_code_doc(text, max_chars=500)
    # Le bloc de code tient dans un seul chunk, non fragmenté ni désindenté.
    assert any(code in c for c, _ in chunks)


def test_hard_split_preserves_line_breaks():
    block = "\n".join(f"    ligne_indentee_{i} = {i}" for i in range(50))
    pieces = _hard_split(block, max_chars=120)
    assert len(pieces) > 1
    # Chaque morceau garde l'indentation et des newlines.
    assert all(p.startswith("    ligne_indentee_") for p in pieces)
    assert all("\n" in p for p in pieces if p.count("ligne") > 1)


def test_chunk_empty_text():
    assert chunk_code_doc("") == []
    assert chunk_code_doc("   \n  \n") == []


# ─── pydoc (stdlib, CI-safe) ──────────────────────────────────────────


def test_extract_pydoc_json():
    out = dict(extract_pydoc(("json",)))
    assert "pydoc/json" in out
    assert "json" in out["pydoc/json"].lower()


def test_extract_pydoc_skips_bad_module():
    # Module inexistant ignoré sans lever.
    out = dict(extract_pydoc(("module_qui_nexiste_pas_xyz", "json")))
    assert "pydoc/json" in out
    assert "pydoc/module_qui_nexiste_pas_xyz" not in out


# ─── Parsing du bundle (fixture tmp, pas de réseau) ───────────────────


def test_iter_bundle_docs(tmp_path):
    # Reproduit l'arborescence d'un bundle extrait : python-X/<cat>/*.txt
    top = tmp_path / "python-3.13.0-docs-text"
    (top / "tutorial").mkdir(parents=True)
    (top / "howto").mkdir(parents=True)
    (top / "tutorial" / "intro.txt").write_text("Intro\n=====\nbonjour", encoding="utf-8")
    (top / "howto" / "sorting.txt").write_text("Sorting\n=======\nsort", encoding="utf-8")

    docs = dict(iter_bundle_docs(tmp_path, ("tutorial", "howto")))
    assert "tutorial/intro.txt" in docs
    assert "howto/sorting.txt" in docs
    assert "bonjour" in docs["tutorial/intro.txt"]


def test_iter_bundle_docs_missing_category(tmp_path):
    top = tmp_path / "python-3.13.0-docs-text"
    (top / "tutorial").mkdir(parents=True)
    (top / "tutorial" / "a.txt").write_text("A\n=\nx", encoding="utf-8")
    # 'library' absent → ignoré sans erreur.
    docs = dict(iter_bundle_docs(tmp_path, ("tutorial", "library")))
    assert "tutorial/a.txt" in docs
    assert len(docs) == 1


# ─── Source man + registre multi-langage ──────────────────────────────


def test_backspace_overstrike_stripped():
    # man rend le gras/souligné en overstrike `X\x08X` / `_\x08X`.
    assert _BACKSPACE_RE.sub("", "N\x08NA\x08AM\x08ME\x08E") == "NAME"
    assert _BACKSPACE_RE.sub("", "_\x08i_\x08t_\x08a_\x08l") == "ital"


def test_man_language_mapping():
    assert man_language("bash") == "bash"
    assert man_language("git") == "git"
    assert man_language("git-commit") == "git"
    assert man_language("grep") == "shell"


def test_iter_source_unknown_raises():
    with pytest.raises(ValueError, match="Source inconnue"):
        list(iter_source("inexistant", bundle_dir=".", categories=(),
                         pydoc_modules=(), man_pages=()))


@pytest.mark.skipif(shutil.which("man") is None, reason="man absent")
def test_render_man_bash_or_skip():
    text = render_man("bash")
    if text is None:
        pytest.skip("page man bash indisponible sur cet hôte")
    # Texte propre (overstrike retiré) et contenu plausible.
    assert "\x08" not in text
    assert "bash" in text.lower()


@pytest.mark.skipif(shutil.which("man") is None, reason="man absent")
def test_iter_source_man_tags_language():
    out = list(iter_source("man", bundle_dir=".", categories=(),
                           pydoc_modules=(), man_pages=("bash",)))
    if not out:
        pytest.skip("page man bash indisponible")
    origin, text, source, language = out[0]
    assert origin == "man/bash"
    assert source == "man"
    assert language == "bash"
