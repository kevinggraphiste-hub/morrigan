"""Tests du RAG strict de Scáthach (PR C Phase 3).

Le RAG strict est le cœur du "0 hallucination" de Morrigan :
  - sans contexte fiable → on NE génère PAS (réponse "je ne sais pas"
    déterministe via template, pas d'appel LLM)
  - avec contexte → génération instruite de s'appuyer uniquement dessus
  - le contexte combine chunks Danann + faits structurés du KG Ogham

Backend RWKV factice (déterministe) pour tester la logique sans modèle.
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

sys.path.insert(0, ".")

from core.types import ModuleInput, ModuleOutput
from modules.scathach.generator import Scathach
from modules.scathach.rwkv_backend import RWKVBackend


class RecordingRWKV:
    """Backend factice qui enregistre query/context/strict reçus."""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def is_available(self) -> bool:
        return True

    def answer(self, query, context=None, strict=True, **kw) -> str:
        self.calls.append({"query": query, "context": context, "strict": strict})
        return "[RWKV] réponse ancrée"


def _danann(chunks):
    return ModuleOutput(result={"chunks": chunks}, confidence=0.8)


def _ogham(result):
    return ModuleOutput(result=result, confidence=0.85)


# ─── Gating : pas de contexte → pas de génération ─────────────────


def test_strict_no_context_does_not_call_llm():
    """Mode strict + aucun contexte → fallback template, LLM jamais appelé."""
    fake = RecordingRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    out = asyncio.run(sc.process(ModuleInput(
        query="Quelle est la recette du cassoulet ?",
        context={"previous_results": {}},
    )))
    assert out.metadata["generated_by"] == "template"
    assert "Morrigan" in out.result  # not_found.j2 = "je ne sais pas"
    assert fake.calls == []  # 0 hallucination : pas de génération sans contexte


def test_strict_with_chunks_generates():
    fake = RecordingRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    chunks = [{"text": "Le cassoulet est un plat de haricots.",
               "score": 0.9, "metadata": {"score_cosine": 0.9}}]
    out = asyncio.run(sc.process(ModuleInput(
        query="Qu'est-ce que le cassoulet ?",
        context={"previous_results": {"danann": _danann(chunks)}},
    )))
    assert out.metadata["generated_by"] == "rwkv"
    assert fake.calls[0]["strict"] is True
    assert any("cassoulet" in c for c in fake.calls[0]["context"])


def test_non_strict_generates_without_context():
    """strict_rag=False → génère même sans contexte (mode libre)."""
    fake = RecordingRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=False)
    out = asyncio.run(sc.process(ModuleInput(
        query="Raconte une blague.", context={"previous_results": {}},
    )))
    assert out.metadata["generated_by"] == "rwkv"
    assert fake.calls[0]["strict"] is False


# ─── Contexte Ogham (faits KG) ────────────────────────────────────


def test_ogham_compare_injected_as_context():
    fake = RecordingRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    ogham_result = {
        "type": "structured_response",
        "structure_type": "comparison",
        "compare": {
            "a": "tcp", "b": "udp",
            "common_neighbors": [{"id": "protocole", "label": "Protocole", "predicates": ["is_a"]}],
            "a_only": [{"id": "fiable", "label": "Fiable", "predicate": "is_a"}],
            "b_only": [{"id": "rapide", "label": "Rapide", "predicate": "is_a"}],
            "direct_relations": [],
        },
        "facts": None,
    }
    out = asyncio.run(sc.process(ModuleInput(
        query="Compare TCP et UDP",
        context={"previous_results": {"ogham": _ogham(ogham_result)}},
    )))
    assert out.metadata["generated_by"] == "rwkv"
    ctx = fake.calls[0]["context"]
    # Les faits du KG sont passés en contexte.
    joined = " ".join(ctx)
    assert "Protocole" in joined          # commun
    assert "Fiable" in joined             # spécifique TCP
    assert "Rapide" in joined             # spécifique UDP


def test_ogham_facts_injected_as_context():
    fake = RecordingRWKV()
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=True)
    ogham_result = {
        "type": "structured_response",
        "structure_type": "explanation",
        "compare": None,
        "facts": {
            "entity": "brigid",
            "relations": [
                {"subject": "brigid", "predicate": "is_a", "object": "deesse", "confidence": 0.8},
            ],
        },
    }
    out = asyncio.run(sc.process(ModuleInput(
        query="Qui est Brigid ?",
        context={"previous_results": {"ogham": _ogham(ogham_result)}},
    )))
    assert out.metadata["generated_by"] == "rwkv"
    ctx = " ".join(fake.calls[0]["context"])
    # Prédicat humanisé : is_a → "est".
    assert "brigid est deesse" in ctx


def test_ogham_context_helper_humanizes_predicates():
    sc = Scathach(backend="template")
    ogham_result = {
        "facts": {
            "entity": "x",
            "relations": [
                {"subject": "a", "predicate": "uses", "object": "b", "confidence": 1.0},
                {"subject": "a", "predicate": "co_occurs_with", "object": "c", "confidence": 0.3},
            ],
        },
    }
    lines = sc._ogham_context({"ogham": _ogham(ogham_result)})
    assert "a utilise b." in lines
    assert "a est lié à c." in lines


# ─── format_prompt strict (RWKVBackend) ───────────────────────────


def test_format_prompt_strict_has_refusal_instruction():
    p = RWKVBackend.format_prompt("Q ?", context=["fait 1"], strict=True)
    assert "UNIQUEMENT" in p
    assert "Je ne sais pas" in p
    assert "fait 1" in p


def test_format_prompt_non_strict_no_refusal():
    p = RWKVBackend.format_prompt("Q ?", context=["fait 1"], strict=False)
    assert "Je ne sais pas" not in p
    assert "fait 1" in p


def test_answer_passes_strict_to_format(monkeypatch):
    """answer(strict=...) doit propager au format_prompt + generate."""
    captured = {}

    class Spy(RWKVBackend):
        def generate(self, prompt, **kw):  # type: ignore[override]
            captured["prompt"] = prompt
            return "ok"

        def _try_load(self):  # court-circuite le chargement réel
            return True

    spy = Spy(model_path=__import__("pathlib").Path("/nonexistent.gguf"))
    spy.answer("Q ?", context=["info"], strict=True)
    assert "Je ne sais pas" in captured["prompt"]
