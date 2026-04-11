"""
DANANN — Prototype d'embeddings locaux et recherche semantique.

Demontre que sentence-transformers local peut etre utilise
pour encoder des chunks de connaissances et les retrouver
par similarite cosinus, sans API externe.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    """Similarite cosinus entre deux vecteurs."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("=" * 60)
    print("  DANANN — Prototype Memoire Vectorielle Locale")
    print("=" * 60)
    print()

    # 1. Charger le modele d'embeddings local
    print("Chargement du modele all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Modele charge ({model.get_sentence_embedding_dimension()} dimensions)")
    print()

    # 2. Base de connaissances (chunks)
    knowledge_base = [
        "TCP est un protocole fiable qui garantit la livraison des paquets dans l'ordre.",
        "UDP est un protocole rapide mais sans garantie de livraison.",
        "HTTP utilise TCP pour la communication web fiable.",
        "DNS utilise UDP pour la resolution de noms de domaine rapide.",
        "La mythologie celtique comprend les Tuatha De Danann, peuple des dieux.",
        "Brigid est la deesse celtique de la forge, de la poesie et de la guerison.",
        "Le Dagda possede un chaudron inepuisable dans la mythologie celtique.",
        "Les reseaux de neurones liquides sont inspires du C. elegans.",
        "Les LNN peuvent tourner sur CPU avec peu de parametres.",
        "PyTorch est une librairie d'apprentissage automatique open source.",
    ]

    print(f"Indexation de {len(knowledge_base)} chunks de connaissances...")
    embeddings = model.encode(knowledge_base, show_progress_bar=False)
    print(f"  Shape: {embeddings.shape}")
    print()

    # 3. Requetes de test
    queries = [
        "Comment fonctionne TCP ?",
        "Qui est Brigid ?",
        "Parle-moi des reseaux de neurones efficaces",
        "Quelle difference entre TCP et UDP ?",
    ]

    print("--- Recherche semantique ---")
    print()

    for query in queries:
        query_emb = model.encode([query], show_progress_bar=False)[0]

        # Calculer les similarites
        scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        print(f"Requete: {query}")
        for i, (idx, score) in enumerate(ranked[:3]):
            print(f"  [{i+1}] ({score:.3f}) {knowledge_base[idx]}")
        print()

    print("Danann peut stocker et retrouver la connaissance.")
    print("(Prochaine etape: brancher Supabase pgvector pour le stockage persistant)")


if __name__ == "__main__":
    main()
