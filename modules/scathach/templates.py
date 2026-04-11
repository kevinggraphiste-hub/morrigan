"""
SCÁTHACH — Templates Jinja2 pour la génération.

Templates paramétriques avec slots et variantes stylistiques.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger("morrigan.scathach.templates")

# Templates Phase 1 — à migrer vers fichiers Jinja2 séparés
TEMPLATES = {
    "factual_answer": (
        "{{ introduction }}\n\n"
        "{{ content }}\n\n"
        "{% if sources %}Sources : {{ sources }}{% endif %}"
    ),
    "comparison": (
        "**{{ item_a }}** vs **{{ item_b }}**\n\n"
        "Points communs :\n{{ common }}\n\n"
        "Différences :\n{{ differences }}\n\n"
        "{% if recommendation %}Recommandation : {{ recommendation }}{% endif %}"
    ),
    "creative": (
        "{{ content }}"
    ),
    "conversation": (
        "{{ response }}"
    ),
}
