"""
DANANN — Index ANN approximatif (IVF, pure NumPy).

Recherche vectorielle **sous-linéaire** sans dépendance native :
on partitionne le corpus en `n_clusters` cellules (k-means), et à la
requête on ne sonde (`n_probe`) que les cellules dont le centroïde est
le plus proche — au lieu de scanner tout le corpus.

Choix vs DiskANN/SPANN : ces graph-ANN nécessitent un build C++/Rust
lourd, incompatible avec la philo « PC modeste, deps minimales » de
Morrigan. L'IVF pur-NumPy couvre le même besoin (retrieval scalable
sur CPU) sans build natif, et se combine avec la quantization (PR1-3)
pour la RAM. Compromis recall/vitesse réglable par `n_probe`.

Les embeddings MiniLM étant L2-normalisés, on utilise le produit
scalaire (≈ cosine) comme mesure de proximité.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _as_2d_f32(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings doit être 2D (N, D), reçu {arr.shape}")
    return arr


def _kmeans(
    data: np.ndarray, n_clusters: int, n_iter: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """k-means Lloyd (pure NumPy), centroïdes renormalisés (cosine).

    Renvoie (centroids (C, D), assignments (N,)).
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    # Init : échantillon aléatoire de points distincts.
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = data[init_idx].copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # Assignation : centroïde le plus proche (max produit scalaire).
        sims = data @ centroids.T              # (N, C)
        new_assign = np.argmax(sims, axis=1)
        if np.array_equal(new_assign, assignments) and _ > 0:
            assignments = new_assign
            break
        assignments = new_assign
        # Mise à jour : moyenne des points assignés, renormalisée.
        for c in range(n_clusters):
            members = data[assignments == c]
            if members.shape[0] > 0:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[c] = centroid / norm if norm > 0 else centroid
            # cluster vide → on garde l'ancien centroïde.
    return centroids, assignments


@dataclass
class IVFIndex:
    """Index IVF (inverted file) : centroïdes + listes inversées."""

    centroids: np.ndarray            # (C, D) float32
    vectors: np.ndarray              # (N, D) float32 (pour re-score exact)
    lists: List[np.ndarray]          # cluster id -> indices des vecteurs
    n_probe: int                     # nb de cellules sondées par défaut

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
        n_probe: Optional[int] = None,
        n_iter: int = 10,
        seed: int = 0,
    ) -> "IVFIndex":
        arr = _as_2d_f32(embeddings)
        n = arr.shape[0]
        # Heuristique standard : ~sqrt(N) cellules, sonder ~1/8 d'entre elles.
        if n_clusters is None:
            n_clusters = max(1, min(n, int(np.sqrt(n))))
        n_clusters = min(n_clusters, n)
        if n_probe is None:
            # Petit index → sonde tout (exact, pas de perte de recall ;
            # l'IVF n'a d'intérêt qu'à grande échelle). Sinon ~1/8.
            n_probe = n_clusters if n_clusters <= 4 else max(1, n_clusters // 8)

        centroids, assignments = _kmeans(arr, n_clusters, n_iter, seed)
        lists = [np.where(assignments == c)[0] for c in range(n_clusters)]
        return cls(centroids=centroids, vectors=arr, lists=lists, n_probe=n_probe)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    @property
    def n_clusters(self) -> int:
        return self.centroids.shape[0]

    def search(
        self, query: np.ndarray, k: int, n_probe: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recherche IVF : sonde les `n_probe` cellules les plus proches,
        re-score exact les candidats, renvoie top-k (indices, scores)."""
        q = np.asarray(query, dtype=np.float32).ravel()
        probe = n_probe or self.n_probe
        probe = min(probe, self.n_clusters)

        # Cellules les plus proches du query.
        centroid_scores = self.centroids @ q
        probe_cells = np.argpartition(centroid_scores, -probe)[-probe:]

        # Candidats = union des listes inversées sondées.
        parts = [self.lists[c] for c in probe_cells if self.lists[c].size > 0]
        if not parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        cand_idx = np.concatenate(parts)

        # Re-score exact des candidats.
        scores = self.vectors[cand_idx] @ q
        kk = min(k, cand_idx.shape[0])
        order = np.argpartition(scores, -kk)[-kk:]
        order = order[np.argsort(scores[order])[::-1]]
        return cand_idx[order], scores[order]

    def candidates_scanned(self, n_probe: Optional[int] = None) -> int:
        """Nombre moyen de candidats scannés (pour mesurer la sous-linéarité)."""
        probe = min(n_probe or self.n_probe, self.n_clusters)
        sizes = sorted((lst.size for lst in self.lists), reverse=True)
        return int(sum(sizes[:probe]))
