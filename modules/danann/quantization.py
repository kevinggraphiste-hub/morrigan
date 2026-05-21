"""
DANANN — Quantization vectorielle (Phase 4).

Compresse l'index d'embeddings pour tenir un gros corpus sur PC modeste.
Pure NumPy, zéro dépendance native.

Deux schémas :
  - Int8Index  : quantization scalaire symétrique → 4× plus compact que
    float32, recall quasi-identique. Recherche par produit scalaire sur
    les codes int8 déquantizés à la volée.
  - BinaryIndex : 1 bit par dimension (signe) → 32× plus compact.
    Recherche par distance de Hamming (popcount). Approximatif : pensé
    comme *filtre grossier* à 2 étages — on récupère un large top-N en
    binaire, puis on re-score finement (int8/float ou cross-encoder).

Les embeddings MiniLM sont L2-normalisés ; le produit scalaire ≈ cosine.

Recherche à 2 étages (recommandée pour binary) :
    coarse = BinaryIndex.search(q, k*RERANK_FACTOR)   # rapide, large
    fine   = re-score des `coarse` avec une mesure précise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Facteur d'élargissement du top-N grossier avant re-score fin.
# 16 : sur un filtre binaire (lossy), élargir la fenêtre de candidats
# récupère un recall élevé après re-score float. Coût négligeable pour
# un gros corpus (on re-score k*16 vecteurs, pas N).
RERANK_FACTOR = 16


def _as_2d_f32(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings doit être 2D (N, D), reçu {arr.shape}")
    return arr


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices des k plus grands scores, triés décroissant."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    # argpartition O(N) pour les k meilleurs, puis tri local.
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(scores[part])[::-1]]


# ─── Int8 (scalaire, 4×) ───────────────────────────────────────────


@dataclass
class Int8Index:
    """Index quantizé int8 (scalaire symétrique global)."""

    codes: np.ndarray   # (N, D) int8
    scale: float        # facteur de déquantization

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "Int8Index":
        arr = _as_2d_f32(embeddings)
        max_abs = float(np.max(np.abs(arr))) or 1.0
        scale = max_abs / 127.0
        codes = np.round(arr / scale).clip(-127, 127).astype(np.int8)
        return cls(codes=codes, scale=scale)

    def __len__(self) -> int:
        return self.codes.shape[0]

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(np.float32) * self.scale

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k par produit scalaire (≈ cosine si vecteurs normalisés).

        Renvoie (indices, scores). `query` est un vecteur float (D,).
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        # codes (N,D) @ q (D,) puis * scale → produit scalaire approx.
        scores = (self.codes.astype(np.float32) @ q) * self.scale
        idx = _topk(scores, k)
        return idx, scores[idx]

    def memory_bytes(self) -> int:
        return int(self.codes.nbytes)


# ─── Binary (signe, 32×) ───────────────────────────────────────────


@dataclass
class BinaryIndex:
    """Index binaire (signe par dimension, bits packés)."""

    bits: np.ndarray    # (N, ceil(D/8)) uint8
    dim: int            # D original

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "BinaryIndex":
        arr = _as_2d_f32(embeddings)
        bools = arr > 0.0
        packed = np.packbits(bools, axis=1)
        return cls(bits=packed, dim=arr.shape[1])

    def __len__(self) -> int:
        return self.bits.shape[0]

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k par similarité de Hamming (plus de bits communs = mieux).

        Renvoie (indices, scores) où score = D - distance_hamming
        (donc plus grand = plus proche). `query` est un vecteur float (D,).
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        q_bits = np.packbits(q > 0.0)
        # XOR puis popcount par ligne → distance de Hamming.
        xor = np.bitwise_xor(self.bits, q_bits)
        hamming = np.unpackbits(xor, axis=1).sum(axis=1)
        scores = (self.dim - hamming).astype(np.int32)
        idx = _topk(scores.astype(np.float32), k)
        return idx, scores[idx]

    def memory_bytes(self) -> int:
        return int(self.bits.nbytes)


# ─── Recherche exacte de référence ─────────────────────────────────


def exact_search(
    embeddings: np.ndarray, query: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Recherche float32 brute-force (référence pour mesurer le recall)."""
    arr = _as_2d_f32(embeddings)
    q = np.asarray(query, dtype=np.float32).ravel()
    scores = arr @ q
    idx = _topk(scores, k)
    return idx, scores[idx]


def recall_at_k(
    approx_idx: np.ndarray, exact_idx: np.ndarray
) -> float:
    """Fraction du top-k exact retrouvée dans le top-k approximatif."""
    if exact_idx.size == 0:
        return 1.0
    return len(set(approx_idx.tolist()) & set(exact_idx.tolist())) / len(exact_idx)


def two_stage_search(
    coarse: "BinaryIndex",
    fine_embeddings: np.ndarray,
    query: np.ndarray,
    k: int,
    rerank_factor: int = RERANK_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recherche 2 étages : filtre binaire grossier → re-score float fin.

    1. BinaryIndex.search élargi (k * rerank_factor) → candidats.
    2. Re-score exact (float) des candidats → top-k final.

    `fine_embeddings` est la matrice float32 complète (ou un sous-ensemble
    aligné sur les indices du BinaryIndex).
    """
    arr = _as_2d_f32(fine_embeddings)
    q = np.asarray(query, dtype=np.float32).ravel()
    cand_idx, _ = coarse.search(query, k * rerank_factor)
    if cand_idx.size == 0:
        return cand_idx, np.empty(0, dtype=np.float32)
    cand_scores = arr[cand_idx] @ q
    order = np.argsort(cand_scores)[::-1][:k]
    final_idx = cand_idx[order]
    return final_idx, cand_scores[order]
