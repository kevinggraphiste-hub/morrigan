"""Tests post-audit retrieval (docs/audit-retrieval-2026-06-12.md).

Couvre les 3 chantiers issus de l'audit :
- reranker : device CPU par défaut (fix du plantage CUDA silencieux),
  troncature des passages, fenêtre de candidats bornée ;
- IVF : n_probe configurable, exposé au runtime via MORRIGAN_ANN /
  MORRIGAN_IVF_PROBES ;
- build_danann : reranker OFF par défaut (MORRIGAN_RERANKER pour réactiver).

CI-safe : embedder stubé, index int8 fabriqués à la main, zéro réseau.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import numpy as np

from core.knowledge import _retrieval_opts, build_danann
from modules.danann.quantization import Int8Index
from modules.danann.reranker import CrossEncoderReranker
from modules.danann.store import Danann


# ─── Stubs ────────────────────────────────────────────────────────────


class _FakeCEModel:
    """Faux CrossEncoder : capture les paires, score décroissant."""

    def __init__(self):
        self.seen_pairs = []

    def predict(self, pairs):
        self.seen_pairs.append(list(pairs))
        return [float(len(pairs) - i) for i in range(len(pairs))]


class _FakeEngine:
    """Faux EmbeddingEngine : vecteur constant normalisé, jamais de réseau."""

    def __init__(self, dim=8):
        self.dim = dim
        self.model = object()  # non-None → pas de load()
        self.model_name = "fake"

    def encode(self, texts, kind="passage"):
        v = np.ones(self.dim, dtype=np.float32)
        return [(v / np.linalg.norm(v)).tolist() for _ in texts]


def _synthetic_danann(n=12, dim=8, **kwargs) -> Danann:
    """Danann int8 synthétique sans embedder réel ni réseau."""
    d = Danann(compression="int8", use_reranker=False, **kwargs)
    d.embedding_engine = _FakeEngine(dim)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    d._int8 = Int8Index.build(vecs, per_vector=True)
    d.chunks = [f"chunk numero {i} contenu" for i in range(n)]
    d.metadata = [{"domain": "code", "origin": f"doc/{i}"} for i in range(n)]
    return d


# ─── Reranker : device, troncature ────────────────────────────────────


def test_reranker_defaults_cpu_and_truncation():
    ce = CrossEncoderReranker()
    assert ce.device == "cpu"          # sans device explicite → CUDA error
    assert ce.max_passage_chars == 512


def test_reranker_truncates_passages():
    ce = CrossEncoderReranker(max_passage_chars=10)
    ce.model = _FakeCEModel()
    long_text = "x" * 1000
    out = ce.rerank("ma question", [(long_text, 0.5, {})], top_k=1)
    assert len(out) == 1
    assert out[0][0] == long_text       # le texte RENDU reste complet
    (pairs,) = ce.model.seen_pairs
    assert pairs[0][1] == "x" * 10      # le texte ENVOYÉ au CE est tronqué


def test_reranker_no_truncation_when_none():
    ce = CrossEncoderReranker(max_passage_chars=None)
    ce.model = _FakeCEModel()
    ce.rerank("q", [("y" * 999, 0.5, {})])
    (pairs,) = ce.model.seen_pairs
    assert len(pairs[0][1]) == 999


# ─── Store : fenêtre de rerank bornée ─────────────────────────────────


def test_search_bounds_rerank_window():
    d = _synthetic_danann(rerank_window=2)
    captured = {}

    class _Recorder:
        def rerank(self, query, candidates, top_k=None):
            captured["n"] = len(candidates)
            return candidates[:top_k]

    d.reranker = _Recorder()
    res = d.search("une question", top_k=5)
    assert captured["n"] == 2           # fenêtre bornée, pas pre_k=15
    assert len(res) <= 5


# ─── IVF : probes configurables + câblage runtime ─────────────────────


def test_ivf_probes_param_wired():
    d = _synthetic_danann(n=20, ann="ivf", ivf_probes=3)
    d._ensure_ann()
    assert d._ivf is not None
    assert d._ivf.n_probe == 3


def test_retrieval_opts_defaults(monkeypatch):
    for var in ("MORRIGAN_RERANKER", "MORRIGAN_ANN", "MORRIGAN_IVF_PROBES"):
        monkeypatch.delenv(var, raising=False)
    opts = _retrieval_opts(None)
    assert opts == {"use_reranker": False, "ann": "flat", "ivf_probes": None}
    # Valeurs invalides → replis silencieux, jamais d'exception au boot.
    monkeypatch.setenv("MORRIGAN_ANN", "nimporte")
    monkeypatch.setenv("MORRIGAN_IVF_PROBES", "pas-un-nombre")
    opts = _retrieval_opts(None)
    assert opts["ann"] == "flat"
    assert opts["ivf_probes"] is None
    # L'argument explicite garde la priorité sur l'env.
    monkeypatch.setenv("MORRIGAN_RERANKER", "on")
    assert _retrieval_opts(False)["use_reranker"] is False


def _write_fake_index(path, n=10, dim=8):
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = Int8Index.build(vecs, per_vector=True)
    path.mkdir(parents=True)
    (path / "corpus.json").write_text(json.dumps({
        "compression": "int8",
        "embedding_model": "fake",
        "chunks": [f"c{i}" for i in range(n)],
        "metadata": [{} for _ in range(n)],
    }), encoding="utf-8")
    np.savez_compressed(path / "vectors.npz",
                        int8_codes=idx.codes, int8_scale=np.asarray(idx.scale))


def test_build_danann_env_wiring(tmp_path, monkeypatch):
    idx = tmp_path / "index"
    _write_fake_index(idx)
    monkeypatch.setenv("MORRIGAN_ANN", "ivf")
    monkeypatch.setenv("MORRIGAN_IVF_PROBES", "2")
    monkeypatch.delenv("MORRIGAN_RERANKER", raising=False)
    d = build_danann(index_path=str(idx))
    assert d.ann == "ivf"
    assert d.ivf_probes == 2
    assert d.reranker is None            # défaut post-audit : OFF
    assert d.count() == 10

    monkeypatch.setenv("MORRIGAN_RERANKER", "on")
    d = build_danann(index_path=str(idx))
    assert d.reranker is not None
