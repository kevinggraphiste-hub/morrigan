"""Étape 1+2 — instrumentation du pipeline retrieval Danann (perf_counter).

Reproduit fidèlement le chemin store.search() mode int8/flat, en chronométrant
séparément : embed query / recherche vectorielle / rerank cross-encoder.
Puis vérifie l'IVF : build, clusters/probes effectifs, sweep probes
(recall@5 vs latence).
"""
import sys, time, statistics as st
sys.path.insert(0, ".")
import numpy as np
from modules.danann.store import Danann, _tokenize

QUERIES = [
    ("python", "comment lire un fichier en Python ?"),
    ("python", "comprendre les list comprehensions en Python"),
    ("git", "comment annuler un commit git ?"),
    ("git", "comment créer une branche git ?"),
    ("bash", "variable d'environnement dans un script bash"),
    ("shell", "comment chercher un motif dans des fichiers avec grep ?"),
    ("javascript", "comment trier un tableau en JavaScript ?"),
    ("javascript", "qu'est-ce qu'une promesse en JavaScript ?"),
    ("css", "centrer un élément horizontalement en CSS"),
    ("css", "comment faire une grille CSS ?"),
    ("html", "quelle balise HTML pour un lien hypertexte ?"),
    ("html", "comment insérer une image en HTML ?"),
    ("docker", "comment monter un volume dans un conteneur Docker ?"),
    ("docker", "à quoi sert un Dockerfile ?"),
    ("sql", "comment faire une jointure entre deux tables en SQL ?"),
    ("sql", "comment créer un index sur une colonne PostgreSQL ?"),
]
REPS = 5
K = 5
PRE_K = K * 3  # fenêtre reranker, comme store.search

def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

def report(name, ms):
    print(f"  {name:28s} p50={st.median(ms):8.1f} ms   p95={pct(ms, 95):8.1f} ms   (n={len(ms)})")

if __name__ == "__main__":
    d = Danann.load_index("data/models/index_code", use_reranker=False)
    print(f"Index : {d.count()} chunks, compression={d.compression}, ann par défaut={d.ann}")
    d._ensure_embeddings_loaded()
    eng = d.embedding_engine

    # Warmup embed
    eng.encode(["query: warmup"], kind="query")

    # Reranker forcé CPU (le code actuel n'a pas de device → CUDA error sur MX330)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    ce.predict([("warmup", "warmup")])

    t_embed, t_search, t_rerank, t_total = [], [], [], []
    for _ in range(REPS):
        for _, q in QUERIES:
            t0 = time.perf_counter()
            emb = eng.encode([q], kind="query")[0]
            t1 = time.perf_counter()
            qv = np.asarray(emb, dtype=np.float32)
            idx, scores = d._int8.search(qv, PRE_K)
            cands = d._candidates_from(idx, scores, _tokenize(q))
            t2 = time.perf_counter()
            pairs = [(q, text) for text, _, _ in cands]
            ce.predict(pairs)
            t3 = time.perf_counter()
            t_embed.append((t1 - t0) * 1000)
            t_search.append((t2 - t1) * 1000)
            t_rerank.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)

    print(f"\n── ÉTAPE 1 : latence par étage (int8 FLAT, {d.count()} chunks, pre_k={PRE_K}) ──")
    report("embedding requête", t_embed)
    report("recherche vectorielle", t_search)
    report(f"rerank CPU ({PRE_K} paires)", t_rerank)
    report("TOTAL pipeline", t_total)

    # ── ÉTAPE 2 : IVF ──
    from modules.danann.ann import IVFIndex
    t0 = time.perf_counter()
    ivf = IVFIndex.build_from_int8(d._int8)
    build_s = time.perf_counter() - t0
    n = d.count()
    print(f"\n── ÉTAPE 2 : IVF (build {build_s:.1f}s) ──")
    print(f"  clusters={ivf.n_clusters} (√N={int(np.sqrt(n))}), n_probe défaut={ivf.n_probe}, "
          f"candidats scannés ≈ {ivf.candidates_scanned()} / {n} ({100*ivf.candidates_scanned()/n:.0f}%)")

    # Référence exacte = top-5 du flat int8
    q_embs = {q: np.asarray(eng.encode([q], kind="query")[0], dtype=np.float32) for _, q in QUERIES}
    exact = {q: set(d._int8.search(qv, K)[0].tolist()) for q, qv in q_embs.items()}

    print(f"\n  {'probes':>7s} {'recall@5':>9s} {'p50 ms':>8s} {'p95 ms':>8s} {'scannés':>9s}")
    for probes in (4, 8, 16, ivf.n_probe, 32, 64):
        lat, rec = [], []
        for _ in range(REPS):
            for _, q in QUERIES:
                qv = q_embs[q]
                t0 = time.perf_counter()
                idx, _ = ivf.search(qv, K, n_probe=probes)
                lat.append((time.perf_counter() - t0) * 1000)
                rec.append(len(set(idx.tolist()) & exact[q]) / K)
        print(f"  {probes:>7d} {st.mean(rec):>9.3f} {st.median(lat):>8.1f} {pct(lat,95):>8.1f} "
              f"{ivf.candidates_scanned(probes):>9d}")
