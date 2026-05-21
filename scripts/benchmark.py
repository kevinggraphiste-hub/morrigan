"""
Benchmarks de génération Scáthach (Phase 3, PR D).

Mesure honnêtement, sur la machine courante :
  - Latence de génération (p50 / p95 / moyenne / max) en mode RWKV.
  - Taux de refus sur des queries hors-corpus (le « 0 hallucination »
    de Morrigan doit refuser à 100 %, sans appeler le LLM).
  - Taux d'ancrage sur des queries avec contexte (la réponse partage-
    t-elle du vocabulaire avec le contexte fourni ?).
  - Comparaison à la cible README : < 1 s par réponse complète sur CPU.

Le modèle GGUF étant gitignoré (option B), ce benchmark est un outil
*local* : il ne tourne pas en CI. Lance-le après
`scripts/fetch_rwkv_model.py`.

Usage :
    .venv-uv/bin/python scripts/benchmark.py
    .venv-uv/bin/python scripts/benchmark.py --output docs/benchmarks.md
    .venv-uv/bin/python scripts/benchmark.py --max-tokens 80 --repeat 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import ModuleInput, ModuleOutput  # noqa: E402
from modules.scathach.generator import Scathach  # noqa: E402

logger = logging.getLogger("morrigan.benchmark")


# ─── Jeu de benchmark ──────────────────────────────────────────────


@dataclass
class BenchCase:
    """Un cas de benchmark : query + contexte simulé + attendu."""

    query: str
    chunks: List[str] = field(default_factory=list)
    expect_refusal: bool = False  # True = hors-corpus, doit refuser


# Cas "in-corpus" : on simule des chunks Danann pertinents.
# Cas "out-of-corpus" : pas de chunks → doit refuser en mode strict.
DEFAULT_CASES: List[BenchCase] = [
    BenchCase(
        query="Qu'est-ce que le protocole TCP ?",
        chunks=[
            "TCP est un protocole de transport fiable qui garantit la "
            "livraison ordonnée des paquets via une connexion établie.",
        ],
    ),
    BenchCase(
        query="Quelle est la différence entre TCP et UDP ?",
        chunks=[
            "TCP garantit la fiabilité et l'ordre des paquets.",
            "UDP est plus rapide mais ne garantit pas la livraison.",
        ],
    ),
    BenchCase(
        query="Qu'est-ce qu'un réseau neuronal liquide ?",
        chunks=[
            "Un réseau neuronal liquide (LNN) utilise des dynamiques à "
            "temps continu pour une inférence efficiente et compacte.",
        ],
    ),
    BenchCase(
        query="Qui est la déesse Brigid ?",
        chunks=[
            "Brigid est une déesse celtique associée à la forge, la "
            "poésie et la guérison.",
        ],
    ),
    BenchCase(
        query="Comment trier une liste en Python ?",
        chunks=[
            "En Python, sorted(liste) renvoie une nouvelle liste triée ; "
            "liste.sort() trie en place.",
        ],
    ),
    # Hors-corpus : doivent refuser (0 hallucination).
    BenchCase(query="Quelle est la recette du cassoulet toulousain ?", expect_refusal=True),
    BenchCase(query="Quel temps fera-t-il demain à Tokyo ?", expect_refusal=True),
    BenchCase(query="Qui a gagné la coupe du monde 2074 ?", expect_refusal=True),
]


# ─── Exécution ─────────────────────────────────────────────────────


@dataclass
class CaseResult:
    query: str
    expect_refusal: bool
    refused: bool
    generated_by: str
    latency_s: float
    response_chars: int
    grounded: Optional[bool]  # None si non applicable (refus)


def _danann_output(chunks: List[str]) -> ModuleOutput:
    """Simule un output Danann (chunks avec score cosine élevé)."""
    return ModuleOutput(
        result={
            "chunks": [
                {"text": c, "score": 0.9, "metadata": {"score_cosine": 0.9}}
                for c in chunks
            ]
        },
        confidence=0.8,
    )


def _sig_words(text: str) -> set:
    """Mots significatifs (> 4 lettres) d'un texte, lowercase."""
    return {
        w for w in re.findall(r"[a-zàâäéèêëïîôöùûüç]+", text.lower())
        if len(w) > 4
    }


def _is_grounded(response: str, chunks: List[str]) -> bool:
    """Heuristique d'ancrage : recouvrement lexical réponse ↔ contexte."""
    ctx_words: set = set()
    for c in chunks:
        ctx_words |= _sig_words(c)
    return bool(_sig_words(response) & ctx_words)


def run_case(scathach: Scathach, case: BenchCase) -> CaseResult:
    previous: Dict[str, Any] = {}
    if case.chunks:
        previous["danann"] = _danann_output(case.chunks)

    inp = ModuleInput(query=case.query, context={"previous_results": previous})

    t0 = time.time()
    out = asyncio.run(scathach.process(inp))
    latency = time.time() - t0

    response = str(out.result)
    generated_by = out.metadata.get("generated_by", "?")
    refused = generated_by == "template"
    grounded = None if refused else _is_grounded(response, case.chunks)

    return CaseResult(
        query=case.query,
        expect_refusal=case.expect_refusal,
        refused=refused,
        generated_by=generated_by,
        latency_s=latency,
        response_chars=len(response),
        grounded=grounded,
    )


# ─── Agrégation + rapport ──────────────────────────────────────────


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    gen = [r for r in results if not r.refused]
    out_of_corpus = [r for r in results if r.expect_refusal]
    gen_latencies = [r.latency_s for r in gen]

    refusals_correct = sum(1 for r in out_of_corpus if r.refused)
    grounded_cases = [r for r in gen if r.grounded is not None]
    grounded_ok = sum(1 for r in grounded_cases if r.grounded)

    return {
        "n_cases": len(results),
        "n_generated": len(gen),
        "n_out_of_corpus": len(out_of_corpus),
        "refusal_rate": (refusals_correct / len(out_of_corpus)) if out_of_corpus else 1.0,
        "grounding_rate": (grounded_ok / len(grounded_cases)) if grounded_cases else 0.0,
        "latency_p50": statistics.median(gen_latencies) if gen_latencies else 0.0,
        "latency_p95": _percentile(gen_latencies, 95),
        "latency_mean": statistics.fmean(gen_latencies) if gen_latencies else 0.0,
        "latency_max": max(gen_latencies) if gen_latencies else 0.0,
        "target_under_1s": bool(gen_latencies) and all(l < 1.0 for l in gen_latencies),
    }


def format_report(
    summary: Dict[str, Any], results: List[CaseResult], meta: Dict[str, str]
) -> str:
    lines = [
        "# Benchmarks de génération — Morrigan / Scáthach RWKV",
        "",
        f"- **Date** : {meta['date']}",
        f"- **Machine** : {meta['machine']}",
        f"- **Modèle** : {meta['model']}",
        f"- **Backend** : {meta['backend']} (strict_rag={meta['strict']})",
        "",
        "## Synthèse",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Cas testés | {summary['n_cases']} ({summary['n_generated']} générés, {summary['n_out_of_corpus']} hors-corpus) |",
        f"| **Taux de refus (hors-corpus)** | **{summary['refusal_rate']*100:.0f}%** (cible 100% — 0 hallucination) |",
        f"| **Taux d'ancrage (générés)** | **{summary['grounding_rate']*100:.0f}%** |",
        f"| Latence génération p50 | {summary['latency_p50']:.2f} s |",
        f"| Latence génération p95 | {summary['latency_p95']:.2f} s |",
        f"| Latence génération moyenne | {summary['latency_mean']:.2f} s |",
        f"| Latence génération max | {summary['latency_max']:.2f} s |",
        f"| Cible README < 1 s | {'✅ atteinte' if summary['target_under_1s'] else '❌ non atteinte (voir note)'} |",
        "",
        "## Détail par cas",
        "",
        "| Query | Type | generated_by | Latence | Ancré |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        typ = "hors-corpus" if r.expect_refusal else "in-corpus"
        grounded = "—" if r.grounded is None else ("oui" if r.grounded else "non")
        q = r.query if len(r.query) <= 45 else r.query[:42] + "…"
        lines.append(
            f"| {q} | {typ} | {r.generated_by} | {r.latency_s:.2f}s | {grounded} |"
        )

    lines += [
        "",
        "## Lecture honnête",
        "",
        "- Le **refus hors-corpus est déterministe** : sans contexte fiable, "
        "Scáthach n'appelle pas le LLM (latence quasi nulle) et renvoie un "
        "« je ne sais pas ». C'est le cœur du « 0 hallucination ».",
        "- La **cible < 1 s n'est pas atteinte** pour une génération complète "
        "sur ce CPU contraint : un RWKV-6 1.6B Q4_K génère à ~10-12 tok/s, "
        "soit plusieurs secondes pour une réponse de quelques phrases. "
        "Pistes : quantization plus agressive (Q3_K), modèle plus petit "
        "(0.4B), streaming (afficher au fil de l'eau), ou réponses plus "
        "courtes. À traiter en optimisation (Phase 4).",
        "- L'**ancrage** est mesuré par une heuristique de recouvrement "
        "lexical réponse↔contexte — indicatif, pas une preuve d'absence "
        "d'hallucination.",
        "",
        "_Régénérer : `.venv-uv/bin/python scripts/benchmark.py --output docs/benchmarks.md`_",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Écrit le rapport markdown ici.")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=1, help="Répète chaque cas (latence stable).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    scathach = Scathach(backend="rwkv", strict_rag=True)
    rwkv = scathach._get_rwkv()
    if rwkv is None or not rwkv.is_available():
        sys.exit(
            "Backend RWKV indisponible. Lance scripts/fetch_rwkv_model.py "
            "et installe llama-cpp-python."
        )

    print(f"Benchmark sur {len(DEFAULT_CASES)} cas (repeat={args.repeat})…\n")
    results: List[CaseResult] = []
    for case in DEFAULT_CASES:
        last: Optional[CaseResult] = None
        for _ in range(args.repeat):
            last = run_case(scathach, case)
        assert last is not None
        results.append(last)
        flag = "REFUS" if last.refused else f"{last.latency_s:.2f}s"
        print(f"  [{flag:>7}] {case.query}")

    summary = summarize(results)

    from modules.scathach.rwkv_backend import DEFAULT_FILENAME  # noqa: PLC0415
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "model": DEFAULT_FILENAME,
        "backend": "rwkv (llama.cpp)",
        "strict": "True",
    }
    report = format_report(summary, results, meta)

    print("\n" + "=" * 60)
    print(
        f"Refus hors-corpus : {summary['refusal_rate']*100:.0f}%  |  "
        f"Ancrage : {summary['grounding_rate']*100:.0f}%  |  "
        f"Latence p50 : {summary['latency_p50']:.2f}s  p95 : {summary['latency_p95']:.2f}s"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nRapport écrit : {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
