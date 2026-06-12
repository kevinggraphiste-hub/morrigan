"""Étape 3 — A/B retrieval monolithique vs fragmenté par langage (mini-RAG).

Shards = partition de l'index int8 par métadonnée `language` (slices des
codes, zéro re-embedding). Routeur réaliste sans nouveau modèle : centroïde
de shard (le routage par Brigid est impossible : Brigid classifie
l'INTENTION — factual/code/… — pas le langage/domaine).

Métriques : taux de bonne réponse (proxy = top-5 contient un chunk dont
l'origine matche un mot-clé attendu), erreur de routage, latence.
+ mesures complémentaires reranker (paires/longueur).
"""
import sys, time, statistics as st
sys.path.insert(0, ".")
import numpy as np
from modules.danann.store import Danann, _tokenize
from modules.danann.quantization import Int8Index

# (langage attendu, requête, mots-clés d'origine acceptés comme "bonne réponse")
QUERIES = [
    ("python", "comment lire un fichier en Python ?", ["inputoutput", "io", "open", "tutorial"]),
    ("python", "comprendre les list comprehensions en Python", ["datastructures", "tutorial", "ast", "functional"]),
    ("git", "comment annuler un commit git ?", ["git-commit", "git-reset", "git-revert"]),
    ("git", "comment créer une branche git ?", ["git-branch", "git-checkout"]),
    ("bash", "variable d'environnement dans un script bash", ["bash"]),
    ("shell", "comment chercher un motif dans des fichiers avec grep ?", ["grep"]),
    ("javascript", "comment trier un tableau en JavaScript ?", ["array/sort", "array/tosorted", "array"]),
    ("javascript", "qu'est-ce qu'une promesse en JavaScript ?", ["promise", "using_promises", "asynchronous"]),
    ("css", "centrer un élément horizontalement en CSS", ["margin", "center", "box_alignment", "flexbox"]),
    ("css", "comment faire une grille CSS ?", ["grid"]),
    ("html", "quelle balise HTML pour un lien hypertexte ?", ["elements/a", "links", "createlinks"]),
    ("html", "comment insérer une image en HTML ?", ["img", "image"]),
    ("docker", "comment monter un volume dans un conteneur Docker ?", ["volume", "bind-mount", "storage"]),
    ("docker", "à quoi sert un Dockerfile ?", ["dockerfile"]),
    ("sql", "comment faire une jointure entre deux tables en SQL ?", ["join", "tutorial-join", "queries-table"]),
    ("sql", "comment créer un index sur une colonne PostgreSQL ?", ["indexes", "sql-createindex"]),
]
K = 5
REPS = 3

def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

def hit(origins, keywords):
    """Vrai si une origine de résultat matche un des mots-clés attendus."""
    return any(kw in o for o in origins for kw in keywords)

if __name__ == "__main__":
    d = Danann.load_index("data/models/index_code", use_reranker=False)
    d._ensure_embeddings_loaded()
    eng = d.embedding_engine
    eng.encode(["query: warmup"], kind="query")

    langs = sorted({m.get("language") for m in d.metadata})
    shard_idx = {L: np.array([i for i, m in enumerate(d.metadata) if m.get("language") == L]) for L in langs}
    codes, scale = d._int8.codes, d._int8.scale
    shards = {L: Int8Index(codes=codes[ix], scale=scale[ix]) for L, ix in shard_idx.items()}
    print("Shards :", {L: len(ix) for L, ix in shard_idx.items()})

    # Centroïdes de shard (normalisés) pour le routage
    cents = []
    for L in langs:
        ix = shard_idx[L]
        v = (codes[ix].astype(np.float32) * scale[ix][:, None]).mean(axis=0)
        cents.append(v / np.linalg.norm(v))
    cents = np.stack(cents)

    q_embs = {q: np.asarray(eng.encode([q], kind="query")[0], dtype=np.float32) for _, q, _ in QUERIES}

    def eval_mode(name, search_fn, routed_fn=None):
        lat, hits, lang_ok, route_err = [], 0, 0, 0
        for L, q, kws in QUERIES:
            qv = q_embs[q]
            t0 = time.perf_counter()
            for _ in range(REPS):
                res_idx = search_fn(qv)
            lat.append((time.perf_counter() - t0) * 1000 / REPS)
            origins = [d.metadata[i].get("origin", "").lower() for i in res_idx]
            rlangs = {d.metadata[i].get("language") for i in res_idx}
            hits += hit(origins, kws)
            lang_ok += (L in rlangs)
            if routed_fn is not None and L not in routed_fn(qv):
                route_err += 1
        n = len(QUERIES)
        extra = f"  erreurs routage={route_err}/{n}" if routed_fn else ""
        print(f"  {name:32s} bonne réponse {hits:2d}/{n}  bon langage top5 {lang_ok:2d}/{n}  "
              f"p50={st.median(lat):6.1f} ms{extra}")

    def mono_flat(qv):
        idx, _ = d._int8.search(qv, K)
        return idx

    from modules.danann.ann import IVFIndex
    ivf = IVFIndex.build_from_int8(d._int8)
    def mono_ivf(qv):
        idx, _ = ivf.search(qv, K)
        return idx

    def make_sharded(top_n):
        def routed(qv):
            order = np.argsort(cents @ qv)[::-1][:top_n]
            return {langs[i] for i in order}
        def search(qv):
            order = np.argsort(cents @ qv)[::-1][:top_n]
            parts = []
            for ci in order:
                L = langs[ci]
                li, ls = shards[L].search(qv, K)
                parts += [(shard_idx[L][j], s) for j, s in zip(li, ls)]
            parts.sort(key=lambda t: t[1], reverse=True)
            return np.array([i for i, _ in parts[:K]])
        return search, routed

    print(f"\n── ÉTAPE 3 : A/B monolithique vs fragmenté ({len(QUERIES)} requêtes FR) ──")
    eval_mode("monolithique int8 flat", mono_flat)
    eval_mode(f"monolithique IVF (probes={ivf.n_probe})", mono_ivf)
    for n in (1, 2, 3):
        s, r = make_sharded(n)
        eval_mode(f"fragmenté top-{n} shard(s)", s, r)

    # ── Complément reranker : coût selon nb de paires et troncature ──
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    ce.predict([("w", "w")])
    print("\n── Complément : reranker CPU, leviers de coût ──")
    for npairs, trunc in [(15, None), (15, 512), (15, 256), (5, None), (5, 512)]:
        lat = []
        for _, q, _ in QUERIES[:8]:
            qv = q_embs[q]
            idx, sc = d._int8.search(qv, npairs)
            cands = [(d.chunks[i][:trunc] if trunc else d.chunks[i]) for i in idx]
            t0 = time.perf_counter()
            ce.predict([(q, c) for c in cands])
            lat.append((time.perf_counter() - t0) * 1000)
        print(f"  {npairs:2d} paires, passage {'complet' if not trunc else f'{trunc} chars':>12s} : "
              f"p50={st.median(lat):7.1f} ms")
    print("\nLongueur des chunks : p50 =", st.median(len(c) for c in d.chunks), "chars")
