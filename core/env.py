"""
Chargement optionnel des variables d'environnement depuis un fichier .env.

Évite d'avoir à coller le token Telegram (ou les clés Supabase / HF) à
chaque lancement. Dégradation gracieuse :
  - si `python-dotenv` n'est pas installé → no-op (les variables peuvent
    toujours venir de l'environnement réel) ;
  - si le fichier .env est absent → no-op.

Le .env est gitignoré (jamais commité). Voir `.env.example` pour les
clés attendues.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("morrigan.env")

# Racine du repo (core/ → ..).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path = ".env") -> bool:
    """Charge le .env dans os.environ s'il existe et que dotenv est dispo.

    Les variables déjà présentes dans l'environnement ne sont PAS
    écrasées (override=False) : l'env réel l'emporte sur le fichier.

    Renvoie True si un fichier a été chargé, False sinon.
    """
    p = Path(path)
    if not p.is_absolute():
        # Cherche relativement à la racine du repo (robuste au cwd).
        candidate = _REPO_ROOT / p
        p = candidate if candidate.exists() else p

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv absent — .env non chargé (env réel utilisé)")
        return False

    if not p.exists():
        logger.debug(".env introuvable (%s) — env réel utilisé", p)
        return False

    load_dotenv(p, override=False)
    logger.info("Variables d'environnement chargées depuis %s", p)
    return True
