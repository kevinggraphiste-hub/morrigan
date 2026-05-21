"""Tests du harnais de benchmark (PR D Phase 3).

On teste la logique d'agrégation / rapport / ancrage sans le vrai
modèle (backend RWKV factice injecté dans Scáthach pour run_case).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from modules.scathach.generator import Scathach
from scripts.benchmark import (
    BenchCase,
    CaseResult,
    _is_grounded,
    format_report,
    run_case,
    summarize,
)


class FakeRWKV:
    def is_available(self) -> bool:
        return True

    def answer(self, query, context=None, strict=True, **kw) -> str:
        # Réponse qui reprend un mot du contexte (pour tester l'ancrage).
        if context:
            return f"D'après le contexte : {context[0]}"
        return "réponse libre"


# ─── _is_grounded ──────────────────────────────────────────────────


def test_is_grounded_true_on_overlap():
    assert _is_grounded(
        "Le protocole garantit la livraison",
        ["TCP garantit la livraison ordonnée"],
    ) is True


def test_is_grounded_false_without_overlap():
    assert _is_grounded(
        "bonjour tout le monde",
        ["TCP garantit la livraison ordonnée"],
    ) is False


def test_is_grounded_ignores_short_words():
    # "le", "la" (<= 4 lettres) ne comptent pas comme ancrage.
    assert _is_grounded("le la les de", ["le la les de protocole"]) is False


# ─── summarize ─────────────────────────────────────────────────────


def _result(refused, expect_refusal, latency, grounded):
    return CaseResult(
        query="q", expect_refusal=expect_refusal, refused=refused,
        generated_by="template" if refused else "rwkv",
        latency_s=latency, response_chars=10, grounded=grounded,
    )


def test_summarize_refusal_rate():
    results = [
        _result(refused=True, expect_refusal=True, latency=0.0, grounded=None),
        _result(refused=True, expect_refusal=True, latency=0.0, grounded=None),
        _result(refused=False, expect_refusal=False, latency=5.0, grounded=True),
    ]
    s = summarize(results)
    assert s["refusal_rate"] == 1.0  # 2/2 hors-corpus refusés
    assert s["n_out_of_corpus"] == 2
    assert s["n_generated"] == 1


def test_summarize_grounding_rate():
    results = [
        _result(refused=False, expect_refusal=False, latency=5.0, grounded=True),
        _result(refused=False, expect_refusal=False, latency=6.0, grounded=False),
    ]
    s = summarize(results)
    assert s["grounding_rate"] == 0.5


def test_summarize_latency_stats():
    results = [
        _result(refused=False, expect_refusal=False, latency=2.0, grounded=True),
        _result(refused=False, expect_refusal=False, latency=4.0, grounded=True),
        _result(refused=False, expect_refusal=False, latency=6.0, grounded=True),
    ]
    s = summarize(results)
    assert s["latency_p50"] == 4.0
    assert s["latency_max"] == 6.0
    assert s["target_under_1s"] is False


def test_summarize_target_under_1s_true():
    results = [_result(refused=False, expect_refusal=False, latency=0.5, grounded=True)]
    s = summarize(results)
    assert s["target_under_1s"] is True


# ─── format_report ─────────────────────────────────────────────────


def test_format_report_has_key_sections():
    results = [_result(refused=True, expect_refusal=True, latency=0.0, grounded=None)]
    s = summarize(results)
    meta = {"date": "2026-05-21", "machine": "X", "model": "rwkv.gguf",
            "backend": "rwkv", "strict": "True"}
    report = format_report(s, results, meta)
    assert "# Benchmarks de génération" in report
    assert "Taux de refus" in report
    assert "Lecture honnête" in report
    assert "rwkv.gguf" in report


# ─── run_case (avec Scáthach + backend factice) ───────────────────


def test_run_case_refusal_out_of_corpus():
    sc = Scathach(backend="rwkv", rwkv_backend=FakeRWKV(), strict_rag=True)
    case = BenchCase(query="hors corpus ?", chunks=[], expect_refusal=True)
    r = run_case(sc, case)
    assert r.refused is True
    assert r.generated_by == "template"
    assert r.grounded is None


def test_run_case_generates_with_context():
    sc = Scathach(backend="rwkv", rwkv_backend=FakeRWKV(), strict_rag=True)
    case = BenchCase(
        query="Qu'est-ce que TCP ?",
        chunks=["TCP garantit la livraison ordonnée des paquets."],
    )
    r = run_case(sc, case)
    assert r.refused is False
    assert r.generated_by == "rwkv"
    assert r.grounded is True  # FakeRWKV reprend le contexte
    assert r.latency_s >= 0.0
