"""
MORRIGAN-CODE — Module agent specialise code.

Implemente l'interface MorriganModule.

Detecte les blocs de code dans une query (markdown fences ```lang ... ```)
ou dans des chunks fournis via context, et lance le verifieur approprie.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from core.types import ModuleInput, ModuleOutput, MorriganModule
from modules.morrigan_code.verifier import (
    VERIFIERS,
    VerificationResult,
    get_verifier,
)

logger = logging.getLogger("morrigan.code")

# Markdown fence : ```python ... ``` ou ``` ... ```
_FENCE_PATTERN = re.compile(
    r"```(\w+)?\s*\n(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extrait les blocs de code d'un texte markdown.

    Retourne une liste de (language, code).
    Si aucun langage n'est specifie dans la fence, language vaut "".
    """
    blocks: List[Tuple[str, str]] = []
    for match in _FENCE_PATTERN.finditer(text):
        lang = (match.group(1) or "").lower().strip()
        code = match.group(2)
        blocks.append((lang, code))
    return blocks


class MorriganCode(MorriganModule):
    """
    Module agent specialise pour le domaine code.

    Phase 2 (initial) : verification syntaxique Python via AST.
    Phase 2+ : JS/TS, Bash, SQL, HTML/CSS.

    Capacites :
      - Verifier les blocs de code dans une query (fences markdown)
      - Verifier les chunks de code recuperes via Danann (domain=code)
      - Enrichir les metadonnees avec un flag `syntax_valid` et la
        structure extraite (imports, fonctions, classes)
    """

    def __init__(self) -> None:
        self.languages = list(VERIFIERS.keys())
        logger.info(
            "Morrigan-Code initialisee (langages: %s)",
            ", ".join(self.languages),
        )

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """
        Traite un input : extrait les blocs de code, les verifie.

        Sources des blocs (par ordre de priorite) :
          1. `input.parameters["code_blocks"]` : liste explicite de
             {language, code} fournie par l'orchestrateur.
          2. `input.query` : extraction des fences markdown.
        """
        explicit = input.parameters.get("code_blocks") or []

        if explicit:
            blocks: List[Tuple[str, str]] = [
                (b.get("language", ""), b.get("code", ""))
                for b in explicit
            ]
        else:
            blocks = extract_code_blocks(input.query)

        if not blocks:
            return ModuleOutput(
                result={
                    "verified": [],
                    "summary": "Aucun bloc de code detecte.",
                },
                confidence=0.0,
                metadata={
                    "languages_supported": self.languages,
                    "blocks_found": 0,
                },
            )

        verified: List[Dict[str, Any]] = []
        all_valid = True

        for lang, code in blocks:
            verifier = get_verifier(lang) if lang else None
            if verifier is None:
                verified.append({
                    "language": lang or "unknown",
                    "valid": None,
                    "skipped": True,
                    "reason": (
                        f"Langage '{lang}' non supporte"
                        if lang
                        else "Pas de langage specifie dans la fence"
                    ),
                })
                continue

            result: VerificationResult = verifier.verify(code)
            verified.append({
                "language": result.language,
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "structure": result.structure,
            })
            if not result.valid:
                all_valid = False

        # Confiance : 1.0 si tout valide, 0.0 si tout invalide,
        # ratio si mixte. On ne compte que les blocs verifies.
        verified_blocks = [v for v in verified if not v.get("skipped")]
        if verified_blocks:
            valid_count = sum(1 for v in verified_blocks if v["valid"])
            confidence = valid_count / len(verified_blocks)
        else:
            confidence = 0.0

        summary = (
            f"{len(verified_blocks)} bloc(s) verifie(s), "
            f"tous valides" if all_valid and verified_blocks
            else f"{len(verified_blocks)} bloc(s) verifie(s), "
                 f"erreurs detectees"
        )

        return ModuleOutput(
            result={
                "verified": verified,
                "summary": summary,
                "all_valid": all_valid,
            },
            confidence=confidence,
            metadata={
                "languages_supported": self.languages,
                "blocks_found": len(blocks),
                "blocks_verified": len(verified_blocks),
            },
        )

    async def health_check(self) -> bool:
        """Verifie que les verifieurs sont charges."""
        return len(VERIFIERS) > 0

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Morrigan-Code",
            "type": "specialized_agent",
            "domain": "code",
            "languages": self.languages,
            "capabilities": [
                "syntax_verification",
                "ast_structure_extraction",
                "markdown_fence_extraction",
            ],
        }
