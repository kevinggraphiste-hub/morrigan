"""
DANANN — Interface de memoire vectorielle.

Gere le stockage et la recherche de connaissances.
Backends supportes :
- "memory" : store en RAM (Phase 0, dev)
- "supabase" : Supabase pgvector (Phase 1+, persistant)

Phase 2 : metadonnees riches, reranker cross-encoder, filtrage par domaine/type.
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.types import ModuleInput, ModuleOutput, MorriganModule
from modules.danann.ann import IVFIndex
from modules.danann.embeddings import EmbeddingEngine
from modules.danann.quantization import RERANK_FACTOR, BinaryIndex, Int8Index
from modules.danann.reranker import CrossEncoderReranker
from modules.danann.supabase_backend import SupabaseVectorStore

logger = logging.getLogger("morrigan.danann")

# Stopwords FR/EN pour le boost lexical — mots trop courants pour discriminer
_STOPWORDS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "est",
    "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "ce", "ces",
    "cette", "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
    "au", "aux", "en", "dans", "sur", "par", "pour", "avec", "sans",
    "a", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "pas", "ne", "ni", "si", "plus", "moins", "tres", "peu", "bien",
    "tout", "tous", "toute", "toutes", "comme", "ainsi", "alors", "mais",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were",
    "qu", "c", "n", "l", "d", "s", "t", "m",
    "entre", "chez", "vers", "sous", "deux", "trois",
    "explique", "compare", "difference", "qu'est-ce", "quelle",
}


def _tokenize(text: str) -> Set[str]:
    """Lowercase + strip accents + split en tokens significatifs."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", no_accents)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


class Danann(MorriganModule):
    """
    Memoire vectorielle de Morrigan.

    Backends :
    - memory   : store en RAM avec embeddings locaux
    - supabase : Supabase pgvector pour la persistence
    """

    def __init__(
        self,
        backend: str = "memory",
        supabase_url: str = "",
        supabase_key: str = "",
        embedding_model: str = "intfloat/multilingual-e5-small",
        top_k: int = 5,
        use_reranker: bool = True,
        reranker_top_k: int = 3,
        rerank_window: int = 8,
        compression: str = "none",
        ann: str = "flat",
        ivf_probes: Optional[int] = None,
    ):
        self.backend = backend
        self.top_k = top_k
        self.reranker_top_k = reranker_top_k
        # Nb max de candidats passés au cross-encoder (post-audit : son coût
        # est ~linéaire en paires, ~117 ms/paire CPU ; au-delà de ~8 le gain
        # qualité mesuré est nul voire négatif — cf. audit 2026-06-12).
        self.rerank_window = rerank_window

        # Phase 4 : compression de l'index mémoire.
        #   "none"   : float32 (exact, défaut historique)
        #   "int8"   : codes int8 par-vecteur (~4× moins de RAM)
        #   "binary" : bits (coarse Hamming) + int8 (rerank) (~4.5× moins)
        if compression not in ("none", "int8", "binary"):
            raise ValueError(f"compression inconnue : {compression!r}")
        self.compression = compression

        # Phase 4-5 : index ANN sous-linéaire (IVF). "flat" = scan complet
        # (défaut). "ivf" = recherche par cellules (k-means + probes).
        # Combinable avec la compression : re-score int8 des candidats IVF
        # (cf. IVFIndex.build_from_int8), sans matérialiser de float32.
        if ann not in ("flat", "ivf"):
            raise ValueError(f"ann inconnu : {ann!r}")
        self.ann = ann
        # n_probe explicite (None = heuristique IVFIndex ~C/8). L'audit
        # 2026-06-12 mesure recall@5 0.925 à C/8 et 0.988 à 64 probes.
        self.ivf_probes = ivf_probes
        self._ivf: Optional[IVFIndex] = None

        # Store en memoire (toujours disponible en fallback)
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # float32, mode "none"
        self.metadata: List[Dict[str, Any]] = []
        # Index compressés (modes int8 / binary) — float32 jamais conservé.
        self._int8: Optional[Int8Index] = None
        self._binary: Optional[BinaryIndex] = None

        # Moteur d'embeddings (lazy load)
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)

        # Reranker cross-encoder (Phase 2, lazy load)
        self.reranker: Optional[CrossEncoderReranker] = None
        if use_reranker:
            self.reranker = CrossEncoderReranker()

        # Backend Supabase (lazy init)
        self.supabase: Optional[SupabaseVectorStore] = None
        if backend == "supabase":
            url = supabase_url or os.getenv("SUPABASE_URL", "")
            key = supabase_key or os.getenv("SUPABASE_KEY", "")
            self.supabase = SupabaseVectorStore(url, key)
            if not self.supabase.connect():
                logger.warning("Bascule en mode memoire (Supabase indisponible)")
                self.backend = "memory"
                self.supabase = None

        logger.info(
            "Danann initialisee (backend=%s, reranker=%s)",
            self.backend,
            "on" if use_reranker else "off",
        )

    def _ensure_embeddings_loaded(self) -> None:
        """Charge le modele d'embeddings si pas encore fait."""
        if self.embedding_engine.model is None:
            self.embedding_engine.load()

    def _compressed_coarse(
        self, query_emb: Any, pre_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recherche grossiere sur l'index compresse → (indices, scores).

        - int8  : produit scalaire direct sur les codes int8.
        - binary: filtre Hamming large (pre_k * RERANK_FACTOR) puis
          re-score des candidats avec les codes int8 (étage fin).
        Les embeddings MiniLM etant L2-normalises, le produit scalaire
        approxime le cosine (comparable au mode "none").
        """
        q = np.asarray(query_emb, dtype=np.float32).ravel()
        if self.compression == "binary" and self._binary is not None:
            n = len(self.chunks)
            cand_idx, _ = self._binary.search(q, min(n, pre_k * RERANK_FACTOR))
            if cand_idx.size == 0:
                return cand_idx, np.empty(0, dtype=np.float32)
            # Re-score fin avec int8 (sans materialiser de float32).
            codes = self._int8.codes[cand_idx].astype(np.float32)  # type: ignore[union-attr]
            sc = self._int8.scale  # type: ignore[union-attr]
            scale = sc[cand_idx] if isinstance(sc, np.ndarray) else sc
            scores = (codes @ q) * scale
            order = np.argsort(scores)[::-1][:pre_k]
            return cand_idx[order], scores[order]
        # int8
        if self._int8 is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return self._int8.search(q, pre_k)

    def _ensure_ann(self) -> None:
        """Construit l'index IVF (lazy) — float si non compressé, sinon int8."""
        if self.ann != "ivf" or self._ivf is not None:
            return
        if self.compression == "none" and self.embeddings is not None:
            self._ivf = IVFIndex.build(self.embeddings, n_probe=self.ivf_probes)
        elif self.compression != "none" and self._int8 is not None:
            self._ivf = IVFIndex.build_from_int8(self._int8, n_probe=self.ivf_probes)
        else:
            return
        logger.info(
            "Danann IVF construit : %d cellules sur %d vecteurs",
            self._ivf.n_clusters, len(self._ivf),
        )

    def _candidates_from(
        self, idx: np.ndarray, base_scores: np.ndarray, query_tokens: Set[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Construit la liste (texte, score+boost lexical, meta) triée."""
        candidates: List[Tuple[str, float, Dict]] = []
        for j, i in enumerate(idx):
            score = float(base_scores[j])
            if query_tokens:
                score += min(0.30, 0.08 * len(query_tokens & _tokenize(self.chunks[i])))
            candidates.append((self.chunks[i], score, self.metadata[i]))
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates

    def memory_bytes(self) -> int:
        """Empreinte mémoire de l'index vectoriel (octets)."""
        if self.compression == "none":
            return int(self.embeddings.nbytes) if self.embeddings is not None else 0
        total = self._int8.memory_bytes() if self._int8 else 0
        total += self._binary.memory_bytes() if self._binary else 0
        return total

    def index(
        self, texts: List[str], metadata: Optional[List[Dict]] = None
    ) -> int:
        """
        Indexe une liste de textes.

        Retourne le nombre de chunks effectivement indexes.
        """
        if not texts:
            return 0

        self._ensure_embeddings_loaded()
        # kind="passage" (défaut) : documents indexés → préfixe e5 `passage:`.
        new_embeddings = self.embedding_engine.encode(texts, kind="passage")

        if metadata is None:
            metadata = [{} for _ in texts]

        # Backend Supabase
        if self.backend == "supabase" and self.supabase:
            inserted = self.supabase.insert_batch(
                texts, new_embeddings, metadata
            )
            return inserted

        # Backend memoire
        new_arr = np.asarray(new_embeddings, dtype=np.float32)

        if self.compression == "none":
            if self.embeddings is None:
                self.embeddings = new_arr
            else:
                self.embeddings = np.vstack([self.embeddings, new_arr])
        else:
            # Modes compressés : on quantize le lot et on JETTE le float32.
            # int8 sert de représentation fine (mode int8 + rerank binary).
            if self._int8 is None:
                self._int8 = Int8Index.build(new_arr, per_vector=True)
            else:
                self._int8.extend(new_arr)
            if self.compression == "binary":
                if self._binary is None:
                    self._binary = BinaryIndex.build(new_arr)
                else:
                    self._binary.extend(new_arr)

        # Nouveau contenu → l'IVF (s'il existe) est périmé, rebâti au search.
        self._ivf = None
        self.chunks.extend(texts)
        self.metadata.extend(metadata)

        logger.info(
            "Danann [memory/%s] — %d chunks indexes (total: %d)",
            self.compression, len(texts), len(self.chunks),
        )
        return len(texts)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        domain: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Recherche les chunks les plus proches.

        Phase 2 : filtrage optionnel par domaine/type + reranking cross-encoder.

        Args:
            query: requete utilisateur
            top_k: nombre de resultats finaux
            domain: filtrer par domaine (reseau, ia, mythologie, projet, code)
            chunk_type: filtrer par type (definition, comparison, explanation, fact)
        """
        k = top_k or self.top_k
        self._ensure_embeddings_loaded()
        # kind="query" : préfixe e5 `query:` (asymétrique vs `passage:` à l'index).
        query_emb = self.embedding_engine.encode([query], kind="query")[0]

        # Backend Supabase
        if self.backend == "supabase" and self.supabase:
            return self.supabase.search(query_emb, top_k=k)

        # Backend memoire
        if not self.chunks:
            return []

        # Fenetre de candidats (avant filtrage + rerank). On recupere plus
        # de candidats pour compenser le filtrage et nourrir le reranker.
        pre_k = k * 3 if (domain or chunk_type) else k
        if self.reranker:
            pre_k = max(pre_k, k * 3)

        # Boost lexical : +0.08 par token rare de la query present dans le
        # chunk (plafonne a +0.30). Corrige les cas ou deux chunks ont un
        # score cosine proche mais un seul mentionne explicitement le sujet.
        query_tokens = _tokenize(query)

        if self.ann == "ivf":
            # Recherche sous-linéaire : IVF gather candidats → boost lexical.
            # Re-score float (non compressé) ou int8 (compressé) selon le mode.
            self._ensure_ann()
            if self._ivf is None:
                return []
            q = np.asarray(query_emb, dtype=np.float32)
            cand_idx, base_scores = self._ivf.search(q, pre_k)
            candidates = self._candidates_from(cand_idx, base_scores, query_tokens)
        elif self.compression == "none":
            if self.embeddings is None:
                return []
            query_vec = np.asarray(query_emb, dtype=np.float32)
            # Embeddings et query sont L2-normalises (cf. EmbeddingEngine.
            # encode) → le produit scalaire EST le cosinus. Plus besoin de
            # recalculer les normes du corpus a chaque requete. Coherent avec
            # les chemins int8/binary/IVF qui font deja un dot brut.
            scores = self.embeddings @ query_vec
            if query_tokens:
                boost = np.zeros(len(self.chunks), dtype=np.float32)
                for i, chunk in enumerate(self.chunks):
                    boost[i] = min(0.30, 0.08 * len(query_tokens & _tokenize(chunk)))
                scores = scores + boost
            top_indices = np.argsort(scores)[::-1][:pre_k]
            candidates = [
                (self.chunks[i], float(scores[i]), self.metadata[i])
                for i in top_indices
            ]
        else:
            # Coarse compresse (int8 ou binary→int8) → fenetre de candidats,
            # puis boost lexical sur cette fenetre uniquement (on ne tokenize
            # pas tout le corpus, ce qui ne tiendrait pas a grande echelle).
            cand_idx, base_scores = self._compressed_coarse(query_emb, pre_k)
            candidates = self._candidates_from(cand_idx, base_scores, query_tokens)

        # Filtrage par domaine / type — en **best-effort** : le domain_hint
        # de Dagda est une heuristique mots-clés, pas une vérité. Si le filtre
        # vide entièrement la fenêtre de candidats (corpus sans chunk de ce
        # domaine, ou hint erroné), on **retombe sur les candidats non filtrés**
        # plutôt que de provoquer un faux « je ne sais pas » en RAG strict.
        # Le filtre n'améliore donc la précision que quand il reste pertinent ;
        # il ne peut jamais dégrader le rappel à zéro.
        if domain:
            filtered = [
                (text, score, meta)
                for text, score, meta in candidates
                if meta.get("domain") == domain
            ]
            if filtered:
                candidates = filtered
            else:
                logger.debug(
                    "Filtre domaine '%s' sans candidat → repli sur non-filtré", domain
                )

        if chunk_type:
            filtered = [
                (text, score, meta)
                for text, score, meta in candidates
                if meta.get("type") == chunk_type
            ]
            if filtered:
                candidates = filtered
            else:
                logger.debug(
                    "Filtre type '%s' sans candidat → repli sur non-filtré", chunk_type
                )

        # Phase 2 : reranking cross-encoder — fenêtre bornée (rerank_window),
        # le coût du cross-encoder étant ~linéaire en nombre de paires.
        if self.reranker and candidates:
            candidates = self.reranker.rerank(
                query, candidates[: self.rerank_window], top_k=k
            )
            logger.info(
                "Danann: reranker applique sur %d candidats",
                len(candidates),
            )
        else:
            candidates = candidates[:k]

        return candidates

    def count(self) -> int:
        """Nombre de chunks indexes."""
        if self.backend == "supabase" and self.supabase:
            return self.supabase.count()
        return len(self.chunks)

    # ─── Persistance disque (Phase 4) ───────────────────────────

    def save_index(self, path: Any) -> Path:
        """Sauve l'index mémoire sur disque (dossier).

        Écrit `corpus.json` (compression, chunks, metadata, modèle) et
        `vectors.npz` (arrays numériques). En mode compressé, seuls les
        codes quantizés sont sauvés — pas de float32. Permet de
        reconstruire l'index sans réembedder (gros corpus → chargement
        rapide, RAM réduite).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        corpus = {
            "compression": self.compression,
            "embedding_model": self.embedding_engine.model_name,
            "chunks": self.chunks,
            "metadata": self.metadata,
        }
        (path / "corpus.json").write_text(
            json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
        )

        arrays: Dict[str, np.ndarray] = {}
        if self.compression == "none":
            if self.embeddings is not None:
                arrays["embeddings"] = self.embeddings
        else:
            assert self._int8 is not None
            arrays["int8_codes"] = self._int8.codes
            arrays["int8_scale"] = np.asarray(self._int8.scale)
            if self.compression == "binary":
                assert self._binary is not None
                arrays["binary_bits"] = self._binary.bits
                arrays["binary_dim"] = np.asarray(self._binary.dim)
        np.savez_compressed(path / "vectors.npz", **arrays)

        logger.info(
            "Danann index sauvé : %d chunks (%s) → %s",
            len(self.chunks), self.compression, path,
        )
        return path

    @classmethod
    def load_index(cls, path: Any, **kwargs: Any) -> "Danann":
        """Recharge un index sauvé par save_index (sans réembedder).

        Le modèle d'embeddings n'est chargé que paresseusement à la
        première requête (pour encoder la query) — pas pour l'index.
        """
        path = Path(path)
        corpus = json.loads((path / "corpus.json").read_text(encoding="utf-8"))

        d = cls(
            compression=corpus["compression"],
            embedding_model=corpus.get(
                "embedding_model", "intfloat/multilingual-e5-small"
            ),
            **kwargs,
        )
        d.chunks = list(corpus["chunks"])
        d.metadata = list(corpus["metadata"])

        data = np.load(path / "vectors.npz")
        if d.compression == "none":
            d.embeddings = data["embeddings"] if "embeddings" in data else None
        else:
            scale_arr = data["int8_scale"]
            scale: Any = float(scale_arr) if scale_arr.ndim == 0 else scale_arr
            d._int8 = Int8Index(codes=data["int8_codes"], scale=scale)
            if d.compression == "binary":
                d._binary = BinaryIndex(
                    bits=data["binary_bits"], dim=int(data["binary_dim"])
                )

        logger.info(
            "Danann index chargé : %d chunks (%s) depuis %s",
            len(d.chunks), d.compression, path,
        )
        return d

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Recherche dans la memoire vectorielle."""
        logger.info("Danann cherche: %s", input.query[:60])

        # Phase 2 : filtrage optionnel via parametres
        domain = input.parameters.get("domain")
        chunk_type = input.parameters.get("chunk_type")

        results = self.search(
            input.query, domain=domain, chunk_type=chunk_type
        )

        if not results:
            return ModuleOutput(
                result={"chunks": [], "query": input.query},
                confidence=0.0,
                metadata={
                    "backend": self.backend,
                    "note": "Memoire vide — indexez des connaissances",
                },
            )

        chunks_data = [
            {"text": text, "score": score, "metadata": meta}
            for text, score, meta in results
        ]

        top_score = results[0][1] if results else 0.0

        return ModuleOutput(
            result={"chunks": chunks_data, "query": input.query},
            confidence=float(top_score),
            metadata={
                "backend": self.backend,
                "reranker": "on" if self.reranker else "off",
                "top_k": len(results),
                "total_indexed": self.count(),
                "filter_domain": domain,
                "filter_type": chunk_type,
            },
        )

    async def health_check(self) -> bool:
        if self.backend == "supabase" and self.supabase:
            return self.supabase.client is not None
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Danann",
            "type": "vector_memory",
            "backend": self.backend,
            "compression": self.compression,
            "ann": self.ann,
            "reranker": "on" if self.reranker else "off",
            "capabilities": [
                "semantic_search",
                "knowledge_retrieval",
                "fact_storage",
                "metadata_filtering",
                "cross_encoder_reranking",
                "vector_quantization",
            ],
            "indexed_chunks": self.count(),
            "index_memory_bytes": self.memory_bytes(),
        }
