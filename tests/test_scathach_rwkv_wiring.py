"""Tests du wiring RWKV dans Scáthach (PR B Phase 3).

On injecte un backend RWKV **factice** (pas de modèle réel, pas de
llama_cpp requis) pour tester la logique de bascule template ↔ RWKV
de façon déterministe :
  - mode template → jamais de RWKV
  - mode rwkv dispo → utilise RWKV, contexte = chunks Danann
  - mode rwkv indispo → fallback template (zéro régression)
  - morrigan_code présent → toujours template (sortie structurée)
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

sys.path.insert(0, ".")

from core.types import ModuleInput, ModuleOutput
from modules.scathach.generator import Scathach


class FakeRWKV:
    """Backend RWKV factice : enregistre les appels, renvoie du canned text."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: List[dict] = []

    def is_available(self) -> bool:
        return self._available

    def answer(self, query: str, context: Optional[List[str]] = None, **kw) -> str:
        self.calls.append({"query": query, "context": context})
        return f"[RWKV] réponse à: {query}"


def _danann_output(chunks):
    return ModuleOutput(result={"chunks": chunks}, confidence=0.8)


# ─── Mode template (défaut) : jamais de RWKV ───────────────────────


def test_template_backend_never_uses_rwkv():
    fake = FakeRWKV(available=True)
    # backend template → même si un rwkv est injecté, on ne l'utilise pas.
    sc = Scathach(backend="template", rwkv_backend=fake)
    out = asyncio.run(sc.process(ModuleInput(query="Bonjour", context={"previous_results": {}})))
    assert out.metadata["generated_by"] == "template"
    assert fake.calls == []


# ─── Mode rwkv disponible ──────────────────────────────────────────


def test_rwkv_backend_used_when_available():
    # strict_rag=False : ce test cible le wiring pur (sans contexte).
    # Le gating RAG strict (sans contexte → pas de génération) est
    # couvert dans test_scathach_rag_strict.py.
    fake = FakeRWKV(available=True)
    sc = Scathach(backend="rwkv", rwkv_backend=fake, strict_rag=False)
    out = asyncio.run(sc.process(ModuleInput(query="Qu'est-ce que TCP ?", context={"previous_results": {}})))
    assert out.metadata["generated_by"] == "rwkv"
    assert out.result.startswith("[RWKV]")
    assert len(fake.calls) == 1


def test_rwkv_receives_danann_chunks_as_context():
    fake = FakeRWKV(available=True)
    sc = Scathach(backend="rwkv", rwkv_backend=fake)
    chunks = [
        {"text": "TCP garantit la livraison ordonnée.",
         "score": 0.9, "metadata": {"score_cosine": 0.9}},
        {"text": "TCP établit une connexion avant transfert.",
         "score": 0.8, "metadata": {"score_cosine": 0.8}},
    ]
    previous = {"danann": _danann_output(chunks)}
    out = asyncio.run(sc.process(ModuleInput(
        query="Explique TCP", context={"previous_results": previous},
    )))
    assert out.metadata["generated_by"] == "rwkv"
    # Le contexte transmis à RWKV contient le texte des chunks pertinents.
    ctx = fake.calls[0]["context"]
    assert ctx is not None
    assert any("livraison" in c for c in ctx)


def test_auto_backend_is_alias_for_rwkv():
    fake = FakeRWKV(available=True)
    sc = Scathach(backend="auto", rwkv_backend=fake, strict_rag=False)
    out = asyncio.run(sc.process(ModuleInput(query="Salut", context={"previous_results": {}})))
    assert out.metadata["generated_by"] == "rwkv"


# ─── Mode rwkv indisponible → fallback template ───────────────────


def test_rwkv_unavailable_falls_back_to_template():
    fake = FakeRWKV(available=False)
    sc = Scathach(backend="rwkv", rwkv_backend=fake)
    out = asyncio.run(sc.process(ModuleInput(query="Bonjour", context={"previous_results": {}})))
    assert out.metadata["generated_by"] == "template"
    assert "Morrigan" in out.result  # rendu not_found.j2
    assert fake.calls == []  # answer() jamais appelé si indispo


def test_rwkv_exception_falls_back_to_template():
    class BoomRWKV(FakeRWKV):
        def answer(self, *a, **k):
            raise RuntimeError("llama.cpp a planté")

    # strict_rag=False pour que answer() soit bien appelé même sans
    # contexte (sinon le gating strict renverrait None avant l'appel).
    sc = Scathach(backend="rwkv", rwkv_backend=BoomRWKV(available=True), strict_rag=False)
    out = asyncio.run(sc.process(ModuleInput(query="Bonjour", context={"previous_results": {}})))
    assert out.metadata["generated_by"] == "template"
    assert "Morrigan" in out.result


# ─── morrigan_code → toujours template ────────────────────────────


def test_code_verification_stays_template_even_in_rwkv_mode():
    fake = FakeRWKV(available=True)
    sc = Scathach(backend="rwkv", rwkv_backend=fake)
    code_out = ModuleOutput(
        result={
            "verified": [{"language": "python", "valid": True, "skipped": False,
                          "errors": [], "warnings": [], "structure": {}}],
            "all_valid": True,
        },
        confidence=1.0,
        metadata={"blocks_verified": 1},
    )
    previous = {"morrigan_code": code_out}
    out = asyncio.run(sc.process(ModuleInput(
        query="Vérifie ce code", context={"previous_results": previous},
    )))
    # Code verification = template, jamais RWKV.
    assert out.metadata["generated_by"] == "template"
    assert fake.calls == []


# ─── _extract_structure_type : régression PR4 KG corrigée ─────────


def test_structure_type_reads_new_kg_field():
    """Ogham post-KG renvoie type=structured_response + structure_type=comparison."""
    ogham_out = ModuleOutput(
        result={"type": "structured_response", "structure_type": "comparison"},
        confidence=0.85,
    )
    assert Scathach._extract_structure_type({"ogham": ogham_out}) == "comparison"


def test_structure_type_legacy_fallback():
    """Ancien format Ogham (type = sous-type directement) reste géré."""
    ogham_out = ModuleOutput(result={"type": "comparison"}, confidence=0.7)
    assert Scathach._extract_structure_type({"ogham": ogham_out}) == "comparison"


def test_structure_type_marker_only_defaults_to_explanation():
    """type=structured_response sans structure_type → défaut explanation."""
    ogham_out = ModuleOutput(result={"type": "structured_response"}, confidence=0.7)
    assert Scathach._extract_structure_type({"ogham": ogham_out}) == "explanation"
