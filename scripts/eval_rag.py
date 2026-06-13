#!/usr/bin/env python3
"""Évaluation du RAG strict sur l'index code — Phase 2D.

Deux questions, un seul jeu de données committé :
1. **Qualité retrieval** : les requêtes FR in-corpus trouvent-elles le bon
   document (hit@k sur l'origine attendue) ?
2. **Calibration du gate** : quel seuil `MIN_RELEVANCE_SCORE` sépare le
   mieux in-corpus (doit passer) et hors-corpus (doit être refusé) ?
   Le seuil historique 0.42 datait de MiniLM ; l'embedder e5 (Phase 2A)
   produit des cosinus resserrés (~0.75-0.95) qui le rendent inopérant.

Le verdict de refus est rendu par le VRAI gate runtime
(`Scathach._relevant_chunks`) — pas une réplique — pour que la calibration
mesure exactement ce que la prod fera (seuil cosinus pur + garde « token
rare »).

Usage :
    .venv-uv/bin/python scripts/eval_rag.py [--index data/models/index_code]
        [--sweep] [--threshold 0.86] [--k 3]

`--sweep` balaie une grille de seuils et affiche recall in-corpus /
refus hors-corpus pour chacun. Sans `--sweep`, évalue au seuil runtime
actuel (ou `--threshold`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Jeu d'éval committé ────────────────────────────────────────────────
# IN_CORPUS : (langage attendu, requête FR, mots-clés d'origine attendus).
# hit = un résultat du top-k a la bonne `language` ET une origine contenant
# un des mots-clés. Les mots-clés sont des sous-chaînes des chemins
# d'origine de l'index (ex. "mdn/javascript/…/array", "man/git-commit",
# "postgres/indexes-intro").

IN_CORPUS = [
    # ── python (docs officielles + pydoc) ──
    ("python", "comment lire un fichier en Python ?", ["inputoutput", "io", "open"]),
    ("python", "comprendre les list comprehensions en Python", ["datastructures", "tutorial"]),
    ("python", "gérer les exceptions avec try except en Python", ["errors", "exception"]),
    ("python", "définir une fonction avec des arguments par défaut en Python", ["controlflow", "function"]),
    ("python", "utiliser les dictionnaires en Python", ["datastructures", "dict", "stdtypes"]),
    ("python", "travailler avec les dates et heures en Python", ["datetime"]),
    ("python", "expressions régulières en Python", ["re", "regex"]),
    ("python", "lancer un sous-processus en Python", ["subprocess"]),
    ("python", "manipuler des chemins de fichiers en Python", ["pathlib", "os.path", "os"]),
    ("python", "sérialiser un objet en JSON en Python", ["json"]),
    ("python", "créer une classe et hériter en Python", ["classes"]),
    ("python", "les générateurs et yield en Python", ["classes", "functional", "generator"]),
    # ── javascript (MDN) ──
    ("javascript", "trier un tableau en JavaScript", ["array", "sort"]),
    ("javascript", "utiliser les promesses en JavaScript", ["promise", "asynchronous", "using_promises"]),
    ("javascript", "la méthode map sur les tableaux JavaScript", ["array", "map"]),
    ("javascript", "déclarer une variable avec let ou const en JavaScript", ["let", "const", "declarations", "grammar", "variables"]),
    ("javascript", "les fonctions fléchées en JavaScript", ["arrow", "functions"]),
    ("javascript", "boucler sur un objet en JavaScript", ["for...in", "object", "loops", "iteration"]),
    ("javascript", "async await en JavaScript", ["async", "await"]),
    ("javascript", "destructurer un objet en JavaScript", ["destructuring"]),
    # ── css (MDN) ──
    ("css", "centrer un élément horizontalement en CSS", ["center", "margin", "box_alignment", "flexbox"]),
    ("css", "utiliser flexbox en CSS", ["flex"]),
    ("css", "créer une grille avec CSS grid", ["grid"]),
    ("css", "les media queries pour le responsive en CSS", ["media_quer", "@media"]),
    ("css", "animer un élément en CSS", ["animation", "transition"]),
    ("css", "les sélecteurs CSS", ["selector"]),
    ("css", "positionner un élément en absolu en CSS", ["position"]),
    # ── html (MDN) ──
    ("html", "créer un lien hypertexte en HTML", ["/a", "anchor", "link"]),
    ("html", "structurer un formulaire en HTML", ["form"]),
    ("html", "insérer une image en HTML", ["img", "image"]),
    ("html", "créer un tableau de données en HTML", ["table"]),
    ("html", "les balises sémantiques HTML", ["semantic", "section", "article", "element"]),
    # ── git (man) ──
    ("git", "comment annuler un commit git ?", ["git-commit", "git-reset", "git-revert"]),
    ("git", "créer une branche avec git", ["git-branch", "git-checkout", "git-switch"]),
    ("git", "fusionner deux branches avec git", ["git-merge"]),
    ("git", "voir l'historique des commits git", ["git-log"]),
    ("git", "remiser des modifications avec git stash", ["git-stash"]),
    ("git", "récupérer les changements distants avec git", ["git-pull", "git-fetch"]),
    # ── bash / shell (man) ──
    ("bash", "écrire une boucle for en bash", ["bash"]),
    ("bash", "les variables d'environnement dans un script bash", ["bash"]),
    ("shell", "chercher un motif dans des fichiers avec grep", ["grep"]),
    ("shell", "remplacer du texte avec sed", ["sed"]),
    ("shell", "trouver des fichiers avec find", ["find"]),
    ("shell", "traiter des colonnes de texte avec awk", ["awk"]),
    # ── docker ──
    ("docker", "monter un volume Docker", ["volume"]),
    ("docker", "écrire un Dockerfile", ["dockerfile"]),
    ("docker", "lancer un conteneur avec docker run", ["run", "container"]),
    ("docker", "docker compose pour plusieurs services", ["compose"]),
    ("docker", "construire une image Docker", ["build", "image"]),
    ("docker", "les réseaux Docker", ["network"]),
    # ── sql (PostgreSQL) ──
    ("sql", "créer un index sur une colonne PostgreSQL", ["indexes"]),
    ("sql", "faire une jointure entre deux tables SQL", ["join", "tutorial"]),
    ("sql", "filtrer des lignes avec WHERE en SQL", ["select", "queries", "tutorial", "dml"]),
    # queries-table-expressions couvre GROUP BY/HAVING → hit légitime
    ("sql", "agréger avec GROUP BY en SQL", ["agg", "group", "tutorial", "select", "queries"]),
    ("sql", "mettre à jour des lignes avec UPDATE en SQL", ["update", "dml"]),
    ("sql", "les types de données PostgreSQL", ["datatype"]),
]

# HORS_CORPUS : requêtes FR sans réponse dans l'index code — le RAG strict
# doit les REFUSER (c'est le « 0 hallucination »). Mélange volontaire :
# culture générale, vie quotidienne, et tech-adjacent absent du corpus.
HORS_CORPUS = [
    "quelle est la capitale de l'Australie ?",
    "recette de la tarte tatin",
    "qui a gagné la coupe du monde de football 1998 ?",
    "comment soigner une angine ?",
    "résume-moi la révolution française",
    "quel est le meilleur film de science-fiction ?",
    "combien de calories dans une pomme ?",
    "comment méditer pour débutant ?",
    "quelle voiture électrique acheter en 2026 ?",
    "histoire de l'empire romain",
    "comment faire pousser des tomates ?",
    "les paroles de la Marseillaise",
    "quel vin servir avec du poisson ?",
    "comment calculer son IMC ?",
    "itinéraire de randonnée dans les Alpes",
    "qui a peint la Joconde ?",
    # tech-adjacent mais ABSENT du corpus (pas de doc Rust/Kubernetes/ML)
    "comment déclarer une variable en Rust ?",
    "déployer un pod Kubernetes",
    "entraîner un réseau de neurones avec PyTorch",
    "configurer un serveur nginx",
    "programmer un microcontrôleur Arduino",
    "développer une app iOS en Swift",
    "requête GraphQL avec Apollo",
    "installer Windows 11",
]


def _hit(results, language, keywords):
    for _, _, meta in results:
        origin = str(meta.get("origin", "")).lower()
        if meta.get("language") == language and any(k in origin for k in keywords):
            return True
    return False


def _gate_passes(scathach, query, results):
    """Verdict du VRAI gate runtime (seuil cosinus + garde token rare)."""
    chunks = [
        {"text": text, "score": score, "metadata": meta}
        for text, score, meta in results
    ]
    previous = {"danann": SimpleNamespace(result={"chunks": chunks})}
    return bool(scathach._relevant_chunks(query, previous))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default="data/models/index_code")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=None,
                    help="seuil MIN_RELEVANCE_SCORE à évaluer (défaut : runtime)")
    ap.add_argument("--sweep", action="store_true",
                    help="balayer une grille de seuils (calibration)")
    ap.add_argument("--no-reranker", action="store_true",
                    help="désactiver le cross-encoder (défaut : config "
                         "runtime complète, mmarco fenêtre 16 cut 1000)")
    args = ap.parse_args()

    from modules.danann.store import Danann
    from modules.scathach.generator import Scathach

    d = Danann.load_index(args.index, use_reranker=not args.no_reranker,
                          shard_by="language")
    d._ensure_embeddings_loaded()
    scathach = Scathach(backend="template")

    # 1 seule passe retrieval (coûteuse), réutilisée pour tous les seuils.
    runs_in = [(lang, q, kws, d.search(q, top_k=args.k)) for lang, q, kws in IN_CORPUS]
    runs_out = [(q, d.search(q, top_k=args.k)) for q in HORS_CORPUS]

    # ── Qualité retrieval (indépendante du gate) ──
    hits = [(lang, q, _hit(res, lang, kws)) for lang, q, kws, res in runs_in]
    n_hit = sum(1 for _, _, h in hits if h)
    print(f"── Retrieval : hit@{args.k} = {n_hit}/{len(hits)} ──")
    for lang, q, h in hits:
        if not h:
            print(f"  MISS [{lang}] {q}")

    def eval_threshold(t):
        scathach.MIN_RELEVANCE_SCORE = t
        ok_in = sum(1 for _, q, _, res in runs_in if _gate_passes(scathach, q, res))
        refused_out = sum(1 for q, res in runs_out if not _gate_passes(scathach, q, res))
        return ok_in, refused_out

    if args.sweep:
        print(f"\n── Calibration du gate (n_in={len(runs_in)}, n_out={len(runs_out)}) ──")
        print(f"{'seuil':>6} | {'in passent':>10} | {'out refusés':>11}")
        for t in [0.42, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]:
            ok_in, refused_out = eval_threshold(t)
            print(f"{t:>6.2f} | {ok_in:>7}/{len(runs_in)} | {refused_out:>8}/{len(runs_out)}")
    else:
        t = args.threshold if args.threshold is not None else Scathach.MIN_RELEVANCE_SCORE
        ok_in, refused_out = eval_threshold(t)
        print(f"\n── Gate @ seuil {t} ──")
        print(f"  in-corpus passants : {ok_in}/{len(runs_in)}")
        print(f"  hors-corpus refusés : {refused_out}/{len(runs_out)}")
        scathach.MIN_RELEVANCE_SCORE = t
        for q, res in runs_out:
            if _gate_passes(scathach, q, res):
                top = res[0] if res else None
                cos = top[2].get("score_cosine") if top else None
                print(f"  FUITE : {q!r} (top cosine={cos:.3f}, "
                      f"origin={top[2].get('origin', '?')})" if top else f"  FUITE : {q!r}")


if __name__ == "__main__":
    main()
