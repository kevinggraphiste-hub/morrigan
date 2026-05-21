"""Tests du streaming de génération (RWKVBackend / Scáthach / An Dagda).

Backends factices, pas de modèle réel ni llama_cpp.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Iterator, List, Optional

sys.path.insert(0, ".")

from core.dagda import AnDagda
from core.types import ModuleInput, ModuleOutput
from modules.scathach.generator import Scathach
from modules.scathach.rwkv_backend import RWKVBackend


async def _collect(agen) -> List[str]:
    return [x async for x in agen]


# ─── RWKVBackend.generate_stream / answer_stream ──────────────────


class _FakeLlm:
    """Imite llama_cpp.Llama(...) en mode stream=True."""

    def __init__(self, pieces: List[str]) -> None:
        self.pieces = pieces
        self.last_kwargs: dict = {}

    def __call__(self, prompt, **kw):
        self.last_kwargs = kw
        assert kw.get("stream") is True
        for p in self.pieces:
            yield {"choices": [{"text": p}]}


def _backend_with_fake(pieces: List[str]) -> RWKVBackend:
    b = RWKVBackend(model_path=Path("/nonexistent.gguf"))
    b._llm = _FakeLlm(pieces)  # _try_load() renverra True car _llm != None
    return b


def test_generate_stream_yields_pieces():
    b = _backend_with_fake(["Bon", "jour", " !"])
    assert list(b.generate_stream("prompt")) == ["Bon", "jour", " !"]


def test_generate_stream_skips_empty_pieces():
    b = _backend_with_fake(["a", "", "b"])
    assert list(b.generate_stream("p")) == ["a", "b"]


def test_answer_stream_uses_format_prompt():
    b = _backend_with_fake(["ok"])
    out = list(b.answer_stream("Q ?", context=["info"], strict=True))
    assert out == ["ok"]
    # Le prompt strict a bien été construit (passé à _llm).
    # (on vérifie indirectement : pas d'exception + sortie attendue)


def test_generate_stream_raises_when_unavailable():
    b = RWKVBackend(model_path=Path("/nonexistent.gguf"))
    with __import__("pytest").raises(RuntimeError):
        list(b.generate_stream("p"))


# ─── Scáthach.stream ──────────────────────────────────────────────


class FakeStreamRWKV:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: List[dict] = []

    def is_available(self) -> bool:
        return self._available

    def answer_stream(self, query, context=None, strict=True, **kw) -> Iterator[str]:
        self.calls.append({"query": query, "context": context, "strict": strict})
        for piece in ["[RWKV] ", "réponse ", "streamée"]:
            yield piece


def _danann(chunks):
    return ModuleOutput(
        result={"chunks": [{"text": c, "score": 0.9, "metadata": {"score_cosine": 0.9}} for c in chunks]},
        confidence=0.8,
    )


def test_scathach_stream_yields_tokens_with_context():
    fake = FakeStreamRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    inp = ModuleInput(
        query="Explique TCP",
        context={"previous_results": {"danann": _danann(["TCP est fiable et ordonné."])}},
    )
    pieces = asyncio.run(_collect(sc.stream(inp)))
    assert pieces == ["[RWKV] ", "réponse ", "streamée"]
    assert len(fake.calls) == 1


def test_scathach_stream_strict_no_context_yields_template_once():
    fake = FakeStreamRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    inp = ModuleInput(query="recette du cassoulet ?", context={"previous_results": {}})
    pieces = asyncio.run(_collect(sc.stream(inp)))
    # Un seul bloc (template not_found), pas de streaming RWKV.
    assert len(pieces) == 1
    assert "Morrigan" in pieces[0]
    assert fake.calls == []  # LLM jamais appelé (0 hallucination)


def test_scathach_stream_template_backend_yields_once():
    sc = Scathach(backend="template")
    inp = ModuleInput(query="Bonjour", context={"previous_results": {}})
    pieces = asyncio.run(_collect(sc.stream(inp)))
    assert len(pieces) == 1
    assert "Morrigan" in pieces[0]


def test_scathach_stream_code_verification_yields_template():
    fake = FakeStreamRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake)
    code_out = ModuleOutput(
        result={"verified": [{"language": "python", "valid": True, "skipped": False,
                              "errors": [], "warnings": [], "structure": {}}],
                "all_valid": True},
        confidence=1.0, metadata={"blocks_verified": 1},
    )
    inp = ModuleInput(query="vérifie", context={"previous_results": {"morrigan_code": code_out}})
    pieces = asyncio.run(_collect(sc.stream(inp)))
    assert len(pieces) == 1  # template code, pas RWKV
    assert fake.calls == []


# ─── AnDagda.process_stream ───────────────────────────────────────


def test_dagda_process_stream_streams_last_module():
    dagda = AnDagda()
    fake = FakeStreamRWKV()
    # Scáthach streamable + un faux Danann qui fournit du contexte.
    dagda.register_module("scathach", Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=False))

    pieces = asyncio.run(_collect(dagda.process_stream("Salut")))
    # La sortie est streamée morceau par morceau.
    assert len(pieces) >= 1
    assert "".join(pieces) != ""


def test_dagda_process_stream_without_scathach_graceful():
    dagda = AnDagda()
    # Aucun module enregistré → message déterministe, pas de crash.
    pieces = asyncio.run(_collect(dagda.process_stream("Qui est Turing ?")))
    assert len(pieces) == 1
    assert "Morrigan" in pieces[0] or "module" in pieces[0].lower()


def test_dagda_process_stream_fence_routes_to_code():
    """Fence markdown → routing CODE ; le dernier module (scathach) streame."""
    dagda = AnDagda()
    fake = FakeStreamRWKV()
    from modules.morrigan_code.module import MorriganCode
    dagda.register_module("morrigan_code", MorriganCode())
    dagda.register_module("scathach", Scathach(backend="rwkv", rwkv_backend=fake))

    query = "Vérifie :\n```python\nprint('hi')\n```"
    pieces = asyncio.run(_collect(dagda.process_stream(query)))
    # Code verification → template (1 bloc), pas de RWKV.
    assert len(pieces) == 1
    assert fake.calls == []
