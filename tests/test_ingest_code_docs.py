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
    fetch_postgres_html,
    html_to_text,
    iter_bundle_docs,
    iter_docker_docs,
    iter_mdn_docs,
    iter_postgres_docs,
    iter_source,
    man_language,
    mdn_language,
    parse_docker_page,
    parse_mdn_page,
    render_man,
    resolve_pg_tarball_url,
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


# ─── Source MDN (markdown, fixtures tmp, zéro réseau) ─────────────────


_MDN_PAGE = """---
title: Array.prototype.map()
slug: Web/JavaScript/Reference/Global_Objects/Array/map
page-type: javascript-instance-method
---

{{JSRef}}

The {{jsxref("Array")}} method **`map()`** creates a new array.

## Syntax

```js
# pas un titre : commentaire dans une fence
map(callbackFn)
```

## Examples

Some example text.

{{Compat}}
"""


def test_parse_mdn_page_front_matter_and_macros():
    title, body = parse_mdn_page(_MDN_PAGE)
    assert title == "Array.prototype.map()"
    assert "slug:" not in body                # front-matter retiré
    assert "{{JSRef}}" not in body            # ligne macro-only droppée
    assert "{{Compat}}" not in body
    assert "`Array`" in body                  # xref → argument conservé
    assert "{{" not in body                   # plus aucune macro résiduelle


def test_mdn_language_mapping():
    assert mdn_language("javascript") == "javascript"
    assert mdn_language("css") == "css"
    assert mdn_language("html") == "html"
    assert mdn_language("api") == "javascript"


def test_markdown_chunking_headings_not_in_fences():
    _, body = parse_mdn_page(_MDN_PAGE)
    chunks = chunk_code_doc(body, markdown=True, min_chars=5)
    sections = {sec for _, sec in chunks}
    assert "Syntax" in sections
    assert "Examples" in sections
    # Le `# commentaire` dans la fence n'est PAS devenu une section.
    assert not any(sec.startswith("pas un titre") for sec in sections)
    # Le code de la fence est conservé tel quel dans un chunk.
    assert any("map(callbackFn)" in c for c, _ in chunks)


def test_iter_mdn_docs_tmp(tmp_path):
    page_dir = tmp_path / "files" / "en-us" / "web" / "javascript" / "ref" / "map"
    page_dir.mkdir(parents=True)
    (page_dir / "index.md").write_text(_MDN_PAGE, encoding="utf-8")
    out = list(iter_mdn_docs(tmp_path, ("javascript", "css")))  # css absente → warning
    assert len(out) == 1
    origin, text, language = out[0]
    assert origin == "mdn/javascript/ref/map"
    assert language == "javascript"
    assert text.startswith("Array.prototype.map()")  # titre préfixé au corps


def test_iter_source_mdn_wiring(tmp_path):
    page_dir = tmp_path / "files" / "en-us" / "web" / "css" / "color"
    page_dir.mkdir(parents=True)
    (page_dir / "index.md").write_text(
        "---\ntitle: color\n---\n\nThe **color** CSS property sets text color.\n",
        encoding="utf-8",
    )
    out = list(iter_source("mdn", bundle_dir=tmp_path, categories=(),
                           pydoc_modules=(), man_pages=(),
                           mdn_dir=tmp_path, mdn_areas=("css",)))
    assert len(out) == 1
    origin, text, source, language = out[0]
    assert source == "mdn"
    assert language == "css"
    assert origin == "mdn/css/color"


# ─── Source Docker (markdown Hugo, fixtures tmp, zéro réseau) ─────────


_DOCKER_PAGE = """---
title: Docker volumes
description: Persist data with volumes
---

{{< summary-bar feature_name="Volumes" >}}

Volumes are the preferred mechanism for persisting data.

## Create a volume

```console
$ docker volume create my-vol
# liste les volumes
$ docker volume ls
```

{{% include "volumes.md" %}}

See the section.
"""


def test_parse_docker_page_front_matter_and_shortcodes():
    title, body = parse_docker_page(_DOCKER_PAGE)
    assert title == "Docker volumes"
    assert "description:" not in body        # front-matter retiré
    assert "summary-bar" not in body         # shortcode {{< … >}} droppé
    assert "include" not in body             # shortcode {{% … %}} droppé
    assert "docker volume create" in body    # le contenu reste


def test_docker_markdown_chunking():
    _, body = parse_docker_page(_DOCKER_PAGE)
    chunks = chunk_code_doc(body, markdown=True, min_chars=5)
    sections = {sec for _, sec in chunks}
    assert "Create a volume" in sections
    # Le `# liste…` dans la fence console n'est pas devenu une section.
    assert not any(sec.startswith("liste") for sec in sections)
    assert any("docker volume create my-vol" in c for c, _ in chunks)


def test_iter_docker_docs_tmp(tmp_path):
    page_dir = tmp_path / "content" / "manuals" / "storage" / "volumes"
    page_dir.mkdir(parents=True)
    (page_dir / "_index.md").write_text(_DOCKER_PAGE, encoding="utf-8")
    out = list(iter_docker_docs(tmp_path, ("manuals", "reference")))  # reference absente
    assert len(out) == 1
    origin, text, language = out[0]
    assert origin == "docker/manuals/storage/volumes/_index.md"
    assert language == "docker"
    assert text.startswith("Docker volumes")  # titre préfixé au corps


def test_iter_source_docker_wiring(tmp_path):
    page_dir = tmp_path / "content" / "get-started"
    page_dir.mkdir(parents=True)
    (page_dir / "intro.md").write_text(
        "---\ntitle: Intro\n---\n\nDocker runs containers on a shared kernel.\n",
        encoding="utf-8",
    )
    out = list(iter_source("docker", bundle_dir=tmp_path, categories=(),
                           pydoc_modules=(), man_pages=(),
                           docker_dir=tmp_path, docker_areas=("get-started",)))
    assert len(out) == 1
    _, _, source, language = out[0]
    assert source == "docker"
    assert language == "docker"


# ─── Source PostgreSQL (HTML → pseudo-markdown, fixtures tmp) ─────────


_PG_PAGE = """<html><head><title>SELECT</title><style>p {color: red}</style></head>
<body>
<div class="navheader"><table><tr><td><a href="x.html">Prev</a></td></tr></table></div>
<h1>SELECT</h1>
<p>SELECT retrieves rows   from a table.</p>
<h2>Examples</h2>
<pre class="screen">
SELECT id, name
  FROM users
 WHERE age &gt; 21;
</pre>
<div class="navfooter"><table><tr><td>Next</td></tr></table></div>
</body></html>
"""


def test_html_to_text_headings_pre_and_nav():
    text = html_to_text(_PG_PAGE)
    assert "# SELECT" in text                  # <h1> → titre markdown
    assert "## Examples" in text               # <h2> → titre markdown
    assert "Prev" not in text and "Next" not in text   # nav skippée
    assert "color: red" not in text            # <style> skippé
    assert "  FROM users" in text              # indentation du <pre> préservée
    assert "WHERE age > 21" in text            # entités décodées
    assert "rows from a table" in text         # espaces prose collapsés


def test_html_to_text_compatible_markdown_chunker():
    chunks = chunk_code_doc(html_to_text(_PG_PAGE), markdown=True, min_chars=5)
    sections = {sec for _, sec in chunks}
    assert "Examples" in sections
    assert any("SELECT id, name" in c for c, _ in chunks)


def test_iter_postgres_docs_prefix_filter(tmp_path):
    (tmp_path / "sql-select.html").write_text(_PG_PAGE, encoding="utf-8")
    (tmp_path / "protocol-flow.html").write_text(_PG_PAGE, encoding="utf-8")
    out = list(iter_postgres_docs(tmp_path, ("sql", "tutorial")))
    assert [o for o, _, _ in out] == ["postgres/sql-select"]
    _, text, language = out[0]
    assert language == "sql"
    assert "# SELECT" in text


def test_resolve_pg_tarball_url_picks_latest_stable(monkeypatch):
    listing = '<a href="v16.9/">v16.9</a> <a href="v17.5/">v17.5</a> <a href="v17.4/">x</a>'

    class _Resp:
        def __init__(self, data): self._data = data
        def read(self): return self._data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    import scripts.ingest_code_docs as mod
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda url, timeout=30: _Resp(listing.encode()))
    url = resolve_pg_tarball_url("https://example.invalid/source/")
    assert url == "https://example.invalid/source/v17.5/postgresql-17.5-docs.tar.gz"


def test_fetch_postgres_html_extracts_only_doc_pages(tmp_path, monkeypatch):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload in [
            ("postgresql-17.5/doc/src/sgml/html/sql-select.html", b"<h1>SELECT</h1>"),
            ("postgresql-17.5/doc/src/sgml/html/stylesheet.css", b"nope"),
            ("postgresql-17.5/src/backend/main.c", b"int main(){}"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    class _Resp:
        def __init__(self, data): self._data = data
        def read(self): return self._data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    import scripts.ingest_code_docs as mod
    monkeypatch.setattr(mod, "resolve_pg_tarball_url", lambda: "https://example.invalid/pg.tar.gz")
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda url, timeout=300: _Resp(buf.getvalue()))
    dest = tmp_path / "pg"
    fetch_postgres_html(dest)
    assert sorted(p.name for p in dest.iterdir()) == ["sql-select.html"]
    # Idempotent : un 2e appel ne re-télécharge pas (urlopen exploserait sinon).
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-download")))
    fetch_postgres_html(dest)
