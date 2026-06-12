"""Tests du gate RAG strict recalibré (Phase 2D).

Couvre :
- Danann expose le cosinus PUR dans metadata["score_cosine"] sur tous les
  chemins de recherche (int8, mode "none"), sans polluer l'index partagé ;
- le reranker préserve ce cosinus pur (ne l'écrase pas avec le score boosté) ;
- le gate Scáthach lit le cosinus pur : un chunk dont seul le score boosté
  dépasse le seuil est refusé ;
- seuil surchargeable via MORRIGAN_MIN_RELEVANCE (invalide → ignoré).

CI-safe : embedder stubé, zéro réseau.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import numpy as np

from modules.danann.quantization import Int8Index
from modules.danann.reranker import CrossEncoderReranker
from modules.danann.store import Danann
from modules.scathach.generator import Scathach


class _FakeEngine:
    def __init__(self, dim=8):
        self.dim = dim
        self.model = object()
        self.model_name = "fake"
        self.vec = None

    def encode(self, texts, kind="passage"):
        v = (np.asarray(self.vec, dtype=np.float32) if self.vec is not None
             else np.ones(self.dim, dtype=np.float32))
        return [(v / np.linalg.norm(v)).tolist() for _ in texts]


def _danann(compression="int8", n=6, dim=8, **kwargs):
    d = Danann(compression=compression, use_reranker=False, **kwargs)
    d.embedding_engine = _FakeEngine(dim)
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    if compression == "none":
        d.embeddings = vecs
    else:
        d._int8 = Int8Index.build(vecs, per_vector=True)
    d.chunks = [f"contenu numero {i}" for i in range(n)]
    d.metadata = [{"origin": f"doc/{i}"} for i in range(n)]
    return d


# ─── score_cosine exposé partout, sans boost ni pollution ──────────────


def test_score_cosine_in_metadata_int8():
    d = _danann("int8")
    res = d.search("contenu numero", top_k=3)
    for text, score, meta in res:
        assert "score_cosine" in meta
        # cosinus pur ≤ score (boost lexical ≥ 0), et jamais > 1 + epsilon
        assert meta["score_cosine"] <= score + 1e-6
        assert meta["score_cosine"] <= 1.01


def test_score_cosine_in_metadata_none_mode():
    d = _danann("none")
    res = d.search("contenu numero", top_k=3)
    assert res and all("score_cosine" in meta for _, _, meta in res)


def test_index_metadata_not_polluted():
    d = _danann("int8")
    d.search("contenu numero", top_k=3)
    assert all("score_cosine" not in m for m in d.metadata)


def test_reranker_preserves_pure_cosine():
    ce = CrossEncoderReranker()
    ce.model = SimpleNamespace(predict=lambda pairs: [9.0] * len(pairs))
    candidates = [("texte", 1.05, {"score_cosine": 0.81})]  # score boosté
    out = ce.rerank("q", candidates, top_k=1)
    assert out[0][2]["score_cosine"] == 0.81       # pas écrasé par 1.05
    assert out[0][2]["score_reranker"] == 9.0


# ─── Gate Scáthach : cosinus pur, pas le score boosté ──────────────────


def _previous(chunks):
    return {"danann": SimpleNamespace(result={"chunks": chunks})}


def test_gate_reads_pure_cosine_not_boosted():
    s = Scathach(backend="template")
    s.MIN_RELEVANCE_SCORE = 0.84
    # Score boosté 1.02 au-dessus du seuil, cosinus pur 0.79 en dessous :
    # hors-corpus à recouvrement lexical accidentel → doit être refusé.
    chunks = [{
        "text": "des elephants au zoo",
        "score": 1.02,
        "metadata": {"score_cosine": 0.79},
    }]
    assert s._relevant_chunks("les elephants", _previous(chunks)) == []


def test_gate_passes_in_corpus():
    s = Scathach(backend="template")
    s.MIN_RELEVANCE_SCORE = 0.84
    chunks = [{
        "text": "git commit enregistre les modifications",
        "score": 1.10,
        "metadata": {"score_cosine": 0.92},
    }]
    assert s._relevant_chunks("annuler un commit git", _previous(chunks))


def test_gate_default_threshold_calibrated():
    # 0.84 = calibration e5 (scripts/eval_rag.py) ; garde-fou contre un
    # retour accidentel au 0.42 de l'ère MiniLM.
    assert Scathach.MIN_RELEVANCE_SCORE == 0.84


def test_gate_env_override(monkeypatch):
    monkeypatch.setenv("MORRIGAN_MIN_RELEVANCE", "0.5")
    assert Scathach(backend="template").MIN_RELEVANCE_SCORE == 0.5
    monkeypatch.setenv("MORRIGAN_MIN_RELEVANCE", "pas-un-nombre")
    assert Scathach(backend="template").MIN_RELEVANCE_SCORE == 0.84
    monkeypatch.delenv("MORRIGAN_MIN_RELEVANCE")
    assert Scathach(backend="template").MIN_RELEVANCE_SCORE == 0.84
