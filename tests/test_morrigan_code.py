"""Tests pour Morrigan-Code — agent specialise code."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.morrigan_code import MorriganCode
from modules.morrigan_code.verifier import PythonVerifier
from modules.morrigan_code.module import extract_code_blocks


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
