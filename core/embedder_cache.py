"""
Cache partagé de modèles SentenceTransformer.

Mutualise le bi-encodeur MiniLM entre Danann (mémoire vectorielle) et Brigid
(classification d'intent) : sans ce cache, chacun chargeait sa propre instance
→ **deux fois le même modèle en RAM**. Sur une machine RAM-saturée c'est le
levier #1 (cf. mémoire perf Morrigan).

Module *leaf* : il n'importe que `sentence_transformers` (en lazy), jamais
`modules.*` ni `core.*` → pas de cycle d'import même si `modules/` l'importe.

Partage l'objet modèle (les poids torch + tokenizer), pas la logique d'`encode`
de chaque appelant : Danann renvoie des listes, Brigid des tensors. L'inférence
PyTorch (forward read-only) est sûre en accès concurrent → pas de lock à l'usage,
seulement au chargement (double-checked locking, l'API offload en threads).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("morrigan.embedder_cache")

# (nom canonique, device) -> instance SentenceTransformer
_CACHE: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()


def _canonical(model_name: str) -> str:
    """Unifie les alias d'un même modèle pour ne charger qu'une instance.

    `all-MiniLM-L6-v2` et `sentence-transformers/all-MiniLM-L6-v2` désignent le
    MÊME modèle (sentence-transformers préfixe l'org quand le nom est nu). On
    préfixe ici pour que Danann (`all-MiniLM-L6-v2`) et Brigid
    (`sentence-transformers/all-MiniLM-L6-v2`) tombent sur la même entrée.
    """
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def text_prompt_prefix(model_name: str, kind: str) -> str:
    """Préfixe d'instruction exigé par certains modèles avant l'encodage.

    La famille **e5** (intfloat/*-e5-*) est entraînée avec des préfixes
    asymétriques : `query: ` pour une requête, `passage: ` pour un document
    indexé. Les omettre dégrade fortement le retrieval. Les modèles sans
    convention de préfixe (MiniLM…) renvoient une chaîne vide → comportement
    inchangé. `kind` ∈ {"query", "passage"}.

    Helper partagé Danann/Brigid (module *leaf*) : la cohérence du préfixe entre
    indexation, recherche ET entraînement Brigid passe par ce point unique.
    """
    base = _canonical(model_name).split("/")[-1].lower()
    is_e5 = base.startswith("e5-") or "-e5-" in base
    if not is_e5:
        return ""
    return "query: " if kind == "query" else "passage: "


def get_sentence_transformer(model_name: str, device: str = "cpu"):
    """Renvoie une instance SentenceTransformer partagée (chargée une seule fois
    par couple nom canonique/device). Lève si le chargement échoue — c'est à
    l'appelant de gérer la dégradation gracieuse s'il en veut une."""
    key = (_canonical(model_name), device)
    model = _CACHE.get(key)
    if model is not None:
        return model
    with _LOCK:
        # Re-check sous le lock : un autre thread a pu charger entre-temps.
        model = _CACHE.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Chargement SentenceTransformer %s (device=%s)", key[0], device)
            model = SentenceTransformer(key[0], device=device)
            _CACHE[key] = model
    return model
