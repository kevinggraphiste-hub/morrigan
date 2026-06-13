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
    """Faux EmbeddingEngine : vecteur contrôlable via `.vec`, jamais de réseau."""

    def __init__(self, dim=8):
        self.dim = dim
        self.model = object()  # non-None → pas de load()
        self.model_name = "fake"
        self.vec = None        # requête simulée (défaut : vecteur constant)

    def encode(self, texts, kind="passage"):
        v = (np.asarray(self.vec, dtype=np.float32) if self.vec is not None
             else np.ones(self.dim, dtype=np.float32))
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
    # Défauts 2D : modèle multilingue + troncature 1000 (cut 500 mange le
    # gain mesuré, cf. eval_rag.py).
    assert ce.max_passage_chars == 1000
    assert "mmarco" in ce.model_name


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
    for var in ("MORRIGAN_RERANKER", "MORRIGAN_ANN", "MORRIGAN_IVF_PROBES",
                "MORRIGAN_SHARD_BY"):
        monkeypatch.delenv(var, raising=False)
    opts = _retrieval_opts(None)
    assert opts == {
        "use_reranker": True, "ann": "flat",
        "ivf_probes": None, "shard_by": "language",
    }
    # Reranker : ON par défaut (Phase 2D), opt-out explicite via "off".
    monkeypatch.setenv("MORRIGAN_RERANKER", "off")
    assert _retrieval_opts(None)["use_reranker"] is False
    monkeypatch.setenv("MORRIGAN_RERANKER", "")
    assert _retrieval_opts(None)["use_reranker"] is True
    monkeypatch.delenv("MORRIGAN_RERANKER")
    # Valeurs invalides → replis silencieux, jamais d'exception au boot.
    monkeypatch.setenv("MORRIGAN_ANN", "nimporte")
    monkeypatch.setenv("MORRIGAN_IVF_PROBES", "pas-un-nombre")
    opts = _retrieval_opts(None)
    assert opts["ann"] == "flat"
    assert opts["ivf_probes"] is None
    # Shards : ON par défaut (language), opt-out explicite via off/none.
    monkeypatch.setenv("MORRIGAN_SHARD_BY", "off")
    assert _retrieval_opts(None)["shard_by"] is None
    monkeypatch.setenv("MORRIGAN_SHARD_BY", "none")
    assert _retrieval_opts(None)["shard_by"] is None
    monkeypatch.setenv("MORRIGAN_SHARD_BY", "domain")
    assert _retrieval_opts(None)["shard_by"] == "domain"
    monkeypatch.setenv("MORRIGAN_SHARD_BY", "")
    assert _retrieval_opts(None)["shard_by"] == "language"
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
    assert d.reranker is not None        # défaut 2D : ON (mmarco multilingue)
    assert d.shard_by == "language"      # défaut : shards ON (dégrade seul)
    assert d.count() == 10

    monkeypatch.setenv("MORRIGAN_RERANKER", "off")
    d = build_danann(index_path=str(idx))
    assert d.reranker is None


# ─── Mini-RAG fragmenté (shards par métadonnée) ───────────────────────


def _unit(*coords):
    v = np.zeros(8, dtype=np.float32)
    for i, c in coords:
        v[i] = c
    return v / np.linalg.norm(v)


def _danann_from(vectors, metadatas, **kwargs) -> Danann:
    d = Danann(compression="int8", use_reranker=False, **kwargs)
    d.embedding_engine = _FakeEngine(8)
    d._int8 = Int8Index.build(np.stack(vectors), per_vector=True)
    d.chunks = [f"texte {i}" for i in range(len(vectors))]
    d.metadata = metadatas
    return d


def _two_shards_with_trap(**kwargs) -> Danann:
    """5 chunks python ≈ e1, 5 html ≈ e2 + 1 PIÈGE html = e1 pile.

    Une requête e1 : en monolithique le piège html gagne (dot 1.0 vs ~0.96) ;
    routée sur le shard python, il est exclu — reproduit le faux ami
    « tableau » array/table de l'audit.
    """
    vecs = [_unit((0, 1.0), (1, 0.3)) for _ in range(5)]          # python
    metas = [{"language": "python"} for _ in range(5)]
    vecs += [_unit((1, 1.0), (2, 0.1)) for _ in range(5)]         # html
    metas += [{"language": "html"} for _ in range(5)]
    vecs.append(_unit((0, 1.0)))                                  # piège html
    metas.append({"language": "html", "origin": "piege"})
    return _danann_from(vecs, metas, **kwargs)


def test_shard_routing_excludes_cross_shard_trap():
    d = _two_shards_with_trap(shard_by="language")
    d.embedding_engine.vec = _unit((0, 1.0))                      # requête ≈ e1
    res = d.search("requete python", top_k=3)
    assert all(meta["language"] == "python" for _, _, meta in res)

    # Référence : le même index en monolithique remonte le piège en tête.
    mono = _two_shards_with_trap()
    mono.embedding_engine.vec = _unit((0, 1.0))
    top = mono.search("requete python", top_k=3)
    assert top[0][2].get("origin") == "piege"


def test_shard_router_abstains_on_tie():
    d = _two_shards_with_trap(shard_by="language")
    d._ensure_shards()
    _, cents, _, _ = d._shards
    tie = (cents[0] + cents[1]).astype(np.float32)
    d.embedding_engine.vec = tie / np.linalg.norm(tie)            # pile entre les 2
    res = d.search("requete ambigue", top_k=10)
    langs = {meta["language"] for _, _, meta in res}
    assert langs == {"python", "html"}                            # monolithique


def test_shard_keyless_rows_always_searched():
    vecs = [_unit((0, 1.0), (1, 0.3)) for _ in range(4)]
    metas = [{"language": "python"} for _ in range(4)]
    vecs += [_unit((1, 1.0)) for _ in range(4)]
    metas += [{"language": "html"} for _ in range(4)]
    vecs.append(_unit((0, 1.0), (2, 0.1)))                        # sans clé, ≈ e1
    metas.append({"origin": "keyless"})
    d = _danann_from(vecs, metas, shard_by="language")
    d.embedding_engine.vec = _unit((0, 1.0))
    res = d.search("requete", top_k=5)
    assert any(meta.get("origin") == "keyless" for _, _, meta in res)


def test_shards_disabled_without_int8():
    d = Danann(compression="none", use_reranker=False, shard_by="language")
    d.embedding_engine = _FakeEngine(8)
    d.embeddings = np.stack([_unit((0, 1.0)), _unit((1, 1.0))])
    d.chunks = ["a", "b"]
    d.metadata = [{"language": "python"}, {"language": "html"}]
    d.embedding_engine.vec = _unit((0, 1.0))
    res = d.search("q", top_k=2)
    assert d.shard_by is None                                     # désactivé proprement
    assert len(res) == 2


def test_shards_disabled_single_value():
    vecs = [_unit((0, 1.0)) for _ in range(4)]
    metas = [{"language": "python"} for _ in range(4)]
    d = _danann_from(vecs, metas, shard_by="language")
    d.embedding_engine.vec = _unit((0, 1.0))
    d.search("q", top_k=2)
    assert d.shard_by is None


def test_index_invalidates_shards():
    d = _two_shards_with_trap(shard_by="language")
    d.embedding_engine.vec = _unit((0, 1.0))
    d.search("q", top_k=2)
    assert d._shards is not None
    d.index(["nouveau"], [{"language": "python"}])
    assert d._shards is None                                      # rebâti au prochain search


def test_build_danann_shard_env(tmp_path, monkeypatch):
    idx = tmp_path / "index"
    _write_fake_index(idx)
    monkeypatch.setenv("MORRIGAN_SHARD_BY", "language")
    for var in ("MORRIGAN_RERANKER", "MORRIGAN_ANN", "MORRIGAN_IVF_PROBES"):
        monkeypatch.delenv(var, raising=False)
    d = build_danann(index_path=str(idx))
    assert d.shard_by == "language"
