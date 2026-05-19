"""Tests pour Morrigan-Code — agent specialise code."""

import asyncio
import shutil
import sys

import pytest

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.morrigan_code import MorriganCode
from modules.morrigan_code.verifier import (
    BashVerifier,
    CssVerifier,
    HtmlVerifier,
    JavaScriptVerifier,
    PythonVerifier,
    SqlVerifier,
    get_verifier,
)
from modules.morrigan_code.module import extract_code_blocks


# Skip les tests subprocess si le binaire n'est pas installé sur le runner.
_BASH_AVAILABLE = shutil.which("bash") is not None
_NODE_AVAILABLE = shutil.which("node") is not None


# ─── Verifier unitaire ─────────────────────────────────────────────


def test_python_verifier_valid_code():
    v = PythonVerifier()
    code = "def add(a, b):\n    return a + b\n"
    result = v.verify(code)
    assert result.valid is True
    assert result.errors == []
    assert "add" in result.structure["functions"]


def test_python_verifier_syntax_error():
    v = PythonVerifier()
    code = "def broken(:\n    return 1\n"
    result = v.verify(code)
    assert result.valid is False
    assert len(result.errors) == 1
    assert "Ligne" in result.errors[0]


def test_python_verifier_indent_error():
    v = PythonVerifier()
    code = "def f():\nreturn 1\n"  # corps non indente
    result = v.verify(code)
    assert result.valid is False


def test_python_verifier_empty():
    v = PythonVerifier()
    result = v.verify("")
    assert result.valid is False
    assert "vide" in result.errors[0]


def test_python_verifier_extracts_imports():
    v = PythonVerifier()
    code = "import os\nfrom typing import List\n"
    result = v.verify(code)
    assert "os" in result.structure["imports"]
    assert "typing.List" in result.structure["imports"]


def test_python_verifier_detects_main_guard():
    v = PythonVerifier()
    code = (
        "def main():\n"
        "    pass\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    result = v.verify(code)
    assert result.structure["has_main_guard"] is True


def test_python_verifier_async_function():
    v = PythonVerifier()
    code = "async def fetch():\n    return 1\n"
    result = v.verify(code)
    assert result.valid is True
    assert "async fetch" in result.structure["functions"]


# ─── Bash ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash non installé")
def test_bash_verifier_valid():
    v = BashVerifier()
    code = "#!/bin/bash\nhello() {\n  echo hi\n}\nhello\n"
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["has_shebang"] is True
    assert "hello" in result.structure["functions"]


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash non installé")
def test_bash_verifier_syntax_error():
    v = BashVerifier()
    # `if [` jamais fermé : bash -n détecte.
    code = "if [\n  echo hi\nfi\n"
    result = v.verify(code)
    assert result.valid is False
    assert result.errors  # message non vide


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash non installé")
def test_bash_verifier_function_alt_syntax():
    v = BashVerifier()
    code = "function world() {\n  echo world\n}\n"
    result = v.verify(code)
    assert result.valid is True
    assert "world" in result.structure["functions"]


def test_bash_verifier_empty():
    v = BashVerifier()
    result = v.verify("   \n")
    assert result.valid is False
    assert "vide" in result.errors[0]


# ─── JavaScript ────────────────────────────────────────────────────


@pytest.mark.skipif(not _NODE_AVAILABLE, reason="node non installé")
def test_javascript_verifier_valid():
    v = JavaScriptVerifier()
    code = (
        "import path from 'path';\n"
        "class Foo {\n  bar() { return 1; }\n}\n"
        "function baz(x) { return x + 1; }\n"
    )
    result = v.verify(code)
    assert result.valid is True
    assert "Foo" in result.structure["classes"]
    assert "baz" in result.structure["functions"]
    assert "path" in result.structure["imports"]


@pytest.mark.skipif(not _NODE_AVAILABLE, reason="node non installé")
def test_javascript_verifier_syntax_error():
    v = JavaScriptVerifier()
    code = "function f( {\n  return 1;\n}\n"  # `)` manquant
    result = v.verify(code)
    assert result.valid is False
    assert result.errors


@pytest.mark.skipif(not _NODE_AVAILABLE, reason="node non installé")
def test_javascript_verifier_require_imports():
    v = JavaScriptVerifier()
    code = "const fs = require('fs');\nconst x = require('./local');\n"
    result = v.verify(code)
    assert result.valid is True
    assert "fs" in result.structure["imports"]
    assert "./local" in result.structure["imports"]


# ─── SQL ───────────────────────────────────────────────────────────


def test_sql_verifier_valid_select():
    pytest.importorskip("sqlparse")
    v = SqlVerifier()
    code = "SELECT id, name FROM users WHERE active = 1;"
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["statement_count"] == 1
    assert "SELECT" in result.structure["statement_types"]


def test_sql_verifier_multiple_statements():
    pytest.importorskip("sqlparse")
    v = SqlVerifier()
    code = (
        "INSERT INTO logs (msg) VALUES ('hello');\n"
        "SELECT count(*) FROM logs;\n"
    )
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["statement_count"] == 2
    types = result.structure["statement_types"]
    assert types.get("INSERT") == 1
    assert types.get("SELECT") == 1


def test_sql_verifier_garbage_input():
    pytest.importorskip("sqlparse")
    v = SqlVerifier()
    # sqlparse est très permissif — du texte non-SQL passe en UNKNOWN.
    result = v.verify("ceci n'est pas du sql")
    assert result.valid is False
    assert "Aucune statement" in result.errors[0]


# ─── HTML ──────────────────────────────────────────────────────────


def test_html_verifier_valid():
    v = HtmlVerifier()
    code = (
        "<!DOCTYPE html>\n"
        "<html><head><title>t</title></head>"
        "<body><p>Salut</p></body></html>\n"
    )
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["has_html_root"] is True
    assert result.structure["has_body"] is True


def test_html_verifier_unclosed_tag():
    v = HtmlVerifier()
    code = "<html><body><p>Hello</body></html>"  # <p> jamais fermé
    result = v.verify(code)
    assert result.valid is False
    assert any("p" in err for err in result.errors)


def test_html_verifier_void_tags_no_error():
    v = HtmlVerifier()
    code = "<html><body>line1<br>line2<img src='x' alt=''></body></html>"
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["tag_counts"].get("br") == 1


def test_html_verifier_orphan_close():
    v = HtmlVerifier()
    code = "<div>x</span>"
    result = v.verify(code)
    assert result.valid is False
    assert any("</span>" in err or "span" in err for err in result.errors)


# ─── CSS ───────────────────────────────────────────────────────────


def test_css_verifier_valid():
    pytest.importorskip("tinycss2")
    v = CssVerifier()
    code = ".foo { color: red; } #bar { padding: 1em; }"
    result = v.verify(code)
    assert result.valid is True
    assert result.structure["qualified_rules"] == 2


def test_css_verifier_at_rule():
    pytest.importorskip("tinycss2")
    v = CssVerifier()
    code = "@media (max-width: 600px) { .small { display: none; } }"
    result = v.verify(code)
    assert result.valid is True
    assert "media" in result.structure["at_rules"]


def test_css_verifier_malformed_declaration():
    pytest.importorskip("tinycss2")
    v = CssVerifier()
    # `}` orphelin sans bloc — tinycss2 remonte une erreur.
    code = ".foo { color: red; @@@; }"
    result = v.verify(code)
    # `@@@` est un at-rule mal formé : on attend au moins un warning ou
    # un statut invalid. tinycss2 peut tolérer selon version → on vérifie
    # juste qu'on ne crash pas et qu'on a une structure cohérente.
    assert result.language == "css"


# ─── Registry / alias ──────────────────────────────────────────────


def test_registry_aliases():
    """Les alias usuels doivent pointer vers le bon vérifieur canonique."""
    assert get_verifier("py").language == "python"
    assert get_verifier("sh").language == "bash"
    assert get_verifier("shell").language == "bash"
    assert get_verifier("js").language == "javascript"
    assert get_verifier("node").language == "javascript"


def test_registry_unknown_returns_none():
    assert get_verifier("brainfuck") is None
    assert get_verifier("") is None


# ─── Extraction de blocs markdown ──────────────────────────────────


def test_extract_single_block():
    text = "Voici un exemple :\n```python\nprint('hello')\n```\nVoila."
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == "python"
    assert "print" in blocks[0][1]


def test_extract_multiple_blocks():
    text = (
        "```python\nx = 1\n```\n"
        "Texte intermediaire.\n"
        "```bash\necho hi\n```"
    )
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0][0] == "python"
    assert blocks[1][0] == "bash"


def test_extract_block_no_language():
    text = "```\nplain text\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == ""


def test_extract_no_blocks():
    text = "Pas de bloc ici."
    blocks = extract_code_blocks(text)
    assert blocks == []


# ─── Module MorriganCode (pipeline) ────────────────────────────────


def test_module_no_blocks():
    mod = MorriganCode()
    out = asyncio.run(mod.process(ModuleInput(query="Salut")))
    assert out.metadata["blocks_found"] == 0
    assert out.confidence == 0.0


def test_module_valid_python_block():
    mod = MorriganCode()
    query = "Verifie ce code :\n```python\ndef f(): return 1\n```"
    out = asyncio.run(mod.process(ModuleInput(query=query)))
    assert out.confidence == 1.0
    assert out.metadata["blocks_verified"] == 1
    assert out.result["all_valid"] is True


def test_module_invalid_python_block():
    mod = MorriganCode()
    query = "Code casse :\n```python\ndef f(:\nreturn 1\n```"
    out = asyncio.run(mod.process(ModuleInput(query=query)))
    assert out.confidence == 0.0
    assert out.result["all_valid"] is False


def test_module_unsupported_language_skipped():
    mod = MorriganCode()
    query = "```rust\nfn main() {}\n```"
    out = asyncio.run(mod.process(ModuleInput(query=query)))
    # Bloc detecte mais skip car rust non supporte
    assert out.metadata["blocks_found"] == 1
    assert out.metadata["blocks_verified"] == 0
    assert out.result["verified"][0]["skipped"] is True


def test_module_explicit_blocks_via_parameters():
    mod = MorriganCode()
    inp = ModuleInput(
        query="anything",
        parameters={
            "code_blocks": [
                {"language": "python", "code": "x = 1\n"},
            ]
        },
    )
    out = asyncio.run(mod.process(inp))
    assert out.confidence == 1.0
    assert out.metadata["blocks_verified"] == 1


def test_module_health_check_and_capabilities():
    mod = MorriganCode()
    assert asyncio.run(mod.health_check()) is True
    caps = mod.get_capabilities()
    assert caps["domain"] == "code"
    assert "python" in caps["languages"]


# ─── Integration An Dagda + Scathach ───────────────────────────────


def test_dagda_routes_code_query_to_morrigan_code():
    """Une query avec fence markdown doit etre routee vers QueryType.CODE."""
    from core.dagda import AnDagda
    from core.types import QueryType

    dagda = AnDagda()
    query = "Verifie ce code :\n```python\ndef f(): return 1\n```"
    routing = dagda.classify_query(query)
    assert routing.query_type == QueryType.CODE
    assert routing.modules == ["morrigan_code", "scathach"]
    assert routing.domain_hint == "code"


def test_dagda_no_code_fence_no_code_routing():
    """Sans fence markdown, pas de routing CODE."""
    from core.dagda import AnDagda
    from core.types import QueryType

    dagda = AnDagda()
    routing = dagda.classify_query("Qui est Alan Turing ?")
    assert routing.query_type != QueryType.CODE


def test_pipeline_code_query_end_to_end():
    """Pipeline Dagda → MorriganCode → Scathach sur une query code valide."""
    from core.dagda import AnDagda
    from modules.scathach.generator import Scathach

    dagda = AnDagda()
    dagda.register_module("morrigan_code", MorriganCode())
    dagda.register_module("scathach", Scathach())

    query = "Verifie ce code :\n```python\ndef hello(): return 'hi'\n```"
    response = asyncio.run(dagda.process(query))

    assert response is not None
    assert len(response) > 0
    assert "hello" in response or "valide" in response.lower()


def test_pipeline_code_query_with_syntax_error():
    """Pipeline sur du code casse : doit signaler une erreur."""
    from core.dagda import AnDagda
    from modules.scathach.generator import Scathach

    dagda = AnDagda()
    dagda.register_module("morrigan_code", MorriganCode())
    dagda.register_module("scathach", Scathach())

    query = "Verifie :\n```python\ndef f(:\nreturn 1\n```"
    response = asyncio.run(dagda.process(query))

    assert "erreur" in response.lower() or "erreurs" in response.lower()
