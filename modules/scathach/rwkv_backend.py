"""
SCÁTHACH — Backend RWKV.

Intégration de RWKV (0.19B-1.5B) pour la génération de texte
en Phase 2+. Mémoire constante, inférence linéaire, CPU-friendly.
"""

import logging

logger = logging.getLogger("morrigan.scathach.rwkv")


class RWKVBackend:
    """
    Backend RWKV pour la génération de texte.

    Phase 2 : À implémenter avec rwkv.cpp.
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None
        logger.info("RWKV Backend — non implémenté (Phase 2)")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Génère du texte via RWKV."""
        raise NotImplementedError("RWKV Backend sera implémenté en Phase 2")
