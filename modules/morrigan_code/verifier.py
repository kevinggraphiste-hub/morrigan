"""
MORRIGAN-CODE — Vérifieurs de code par langage.

Chaque verifieur prend un snippet et retourne un VerificationResult
avec un statut, des erreurs, et des metadonnees structurelles.

Phase 2 : PythonVerifier (stdlib `ast`).
Phase 2+ : JS/TS (`tree-sitter`), Bash (`shellcheck`), SQL (`sqlparse`),
           HTML/CSS (`html.parser`, `tinycss2`).
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("morrigan.code.verifier")


@dataclass
class VerificationResult:
    """Resultat d'une verification de code."""

    valid: bool
    language: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Metadonnees structurelles (deps, fonctions, classes, etc.) extraites a la passe AST.
    structure: dict = field(default_factory=dict)


class BaseVerifier(ABC):
    """Interface commune a tous les verifieurs."""

    language: str = "unknown"

    @abstractmethod
    def verify(self, code: str) -> VerificationResult:
        """Verifie un snippet et retourne un VerificationResult."""
        ...


class PythonVerifier(BaseVerifier):
    """
    Verifieur Python utilisant le module `ast` de la stdlib.

    Detecte :
      - erreurs de syntaxe (SyntaxError, IndentationError)
      - imports utilises
      - fonctions et classes definies
      - presence de `if __name__ == "__main__"`

    Ne fait PAS d'analyse semantique (pas de detection de noms non
    definis, pas de type checking) — c'est une passe purement syntaxique.
    """

    language = "python"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=["Snippet vide"],
            )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # ast.parse leve SyntaxError, qui couvre aussi IndentationError.
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=[f"Ligne {e.lineno}: {e.msg}"],
            )

        # Passe AST pour extraire la structure
        imports: List[str] = []
        functions: List[str] = []
        classes: List[str] = []
        has_main_guard = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(f"async {node.name}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.If):
                # Detection de `if __name__ == "__main__":`
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                ):
                    has_main_guard = True

        warnings: List[str] = []
        if not functions and not classes and len(code.splitlines()) > 5:
            warnings.append("Snippet long sans fonction ni classe definie")

        return VerificationResult(
            valid=True,
            language=self.language,
            warnings=warnings,
            structure={
                "imports": imports,
                "functions": functions,
                "classes": classes,
                "has_main_guard": has_main_guard,
                "lines": len(code.splitlines()),
            },
        )


# Registry des verifieurs disponibles, indexes par nom de langage.
VERIFIERS: dict[str, BaseVerifier] = {
    "python": PythonVerifier(),
}


def get_verifier(language: str) -> Optional[BaseVerifier]:
    """Retourne le verifieur pour un langage, None si non supporte."""
    return VERIFIERS.get(language.lower())
