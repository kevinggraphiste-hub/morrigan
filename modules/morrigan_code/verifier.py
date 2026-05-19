"""
MORRIGAN-CODE — Vérifieurs de code par langage.

Chaque vérifieur prend un snippet et retourne un VerificationResult
avec un statut, des erreurs, et des métadonnées structurelles.

Phase 2 :
  - PythonVerifier      → stdlib `ast`
  - BashVerifier        → subprocess `bash -n`
  - JavaScriptVerifier  → subprocess `node --check`
  - SqlVerifier         → `sqlparse` (structure ; cf. limitations en docstring)
  - HtmlVerifier        → stdlib `html.parser` + suivi de pile des balises
  - CssVerifier         → `tinycss2`

Phase 2+ : TypeScript (tsc), Rust (rustc --emit=metadata), …
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Optional

logger = logging.getLogger("morrigan.code.verifier")

# Timeout pour les vérifieurs basés sur subprocess (bash, node).
# Volontairement court : un check de syntaxe ne doit pas tourner > 1 s.
_SUBPROCESS_TIMEOUT = 5


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


# ─── Helpers ─────────────────────────────────────────────────────────────


def _empty_result(language: str) -> VerificationResult:
    return VerificationResult(
        valid=False,
        language=language,
        errors=["Snippet vide"],
    )


def _run_syntax_check(
    argv: List[str], code: str, language: str
) -> Optional[VerificationResult]:
    """
    Lance un sous-processus de check de syntaxe (bash -n, node --check, …).

    Renvoie un VerificationResult si le check s'est exécuté (valide ou non),
    ou None s'il s'est passé sans erreur — auquel cas l'appelant doit faire
    sa propre extraction de structure.

    Le code est passé en stdin pour éviter tout fichier temporaire et toute
    interpolation shell.
    """
    try:
        proc = subprocess.run(
            argv,
            input=code,
            text=True,
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        return VerificationResult(
            valid=False,
            language=language,
            errors=[f"Binaire '{argv[0]}' non disponible sur ce système"],
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            valid=False,
            language=language,
            errors=[
                f"Timeout ({_SUBPROCESS_TIMEOUT}s) sur le check {argv[0]} — "
                "snippet trop volumineux ou parser bloqué"
            ],
        )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or "Erreur de syntaxe (sans message)"
        # Les parsers crachent souvent plusieurs lignes — on garde un résumé compact.
        first_lines = "\n".join(stderr.splitlines()[:5])
        return VerificationResult(
            valid=False,
            language=language,
            errors=[first_lines],
        )

    return None  # Pas d'erreur : à l'appelant d'extraire la structure.


# ─── Python ──────────────────────────────────────────────────────────────


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
            return _empty_result(self.language)

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


# ─── Bash ────────────────────────────────────────────────────────────────


# Fonction bash : `nom() { ... }` ou `function nom { ... }`.
_BASH_FUNC_RE = re.compile(
    r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{",
    re.MULTILINE,
)


class BashVerifier(BaseVerifier):
    """
    Vérifieur Bash via `bash -n` (no-execute syntax check).

    `bash -n` parse le script sans rien exécuter — pas d'effets de bord
    via les commandes du script. Les builtins shell sont parsés selon les
    règles courantes du bash invoqué.

    Détecte :
      - erreurs de syntaxe (`unexpected EOF`, `expected token`, …)
      - shebang présent
      - fonctions définies (deux syntaxes)
    """

    language = "bash"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return _empty_result(self.language)

        error_result = _run_syntax_check(["bash", "-n"], code, self.language)
        if error_result is not None:
            return error_result

        lines = code.splitlines()
        has_shebang = bool(lines and lines[0].startswith("#!"))
        functions = _BASH_FUNC_RE.findall(code)

        return VerificationResult(
            valid=True,
            language=self.language,
            structure={
                "functions": functions,
                "has_shebang": has_shebang,
                "lines": len(lines),
            },
        )


# ─── JavaScript ──────────────────────────────────────────────────────────


# Quelques regex légères pour la structure JS. On reste volontairement simple :
# un vrai parse AST nécessiterait acorn/babel, ce qui sortirait du périmètre
# pure-Python. `node --check` valide déjà la syntaxe en amont.
_JS_FUNCTION_RE = re.compile(
    r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
)
_JS_CLASS_RE = re.compile(
    r"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)\b",
)
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]"""
    r"""|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
)


class JavaScriptVerifier(BaseVerifier):
    """
    Vérifieur JavaScript via `node --check`.

    `node --check` parse le module sans l'exécuter. Couvre la syntaxe
    ES (Node 20 = ES2023+) y compris les modules ESM.

    NB : ne valide PAS TypeScript (syntaxe `: type`, interfaces, generics
    rejetés). Un TypeScriptVerifier dédié reste à faire — il nécessitera
    `tsc` ou un parser TS pur Python.

    On force `--input-type=module` pour accepter à la fois la syntaxe ESM
    (`import x from '...'`) ET les appels `require()` (syntaxiquement
    valides en mode module : `require` est juste un identifiant, l'erreur
    sémantique « require is not defined » n'apparaît qu'à l'exécution).
    En mode script par défaut, `import` top-level est rejeté — c'est ce
    qu'on veut éviter.

    Détecte :
      - erreurs de syntaxe via Node lui-même
      - fonctions nommées (regex légère)
      - classes
      - imports / require()
    """

    language = "javascript"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return _empty_result(self.language)

        error_result = _run_syntax_check(
            ["node", "--input-type=module", "--check"], code, self.language
        )
        if error_result is not None:
            return error_result

        functions = _JS_FUNCTION_RE.findall(code)
        classes = _JS_CLASS_RE.findall(code)
        # Chaque match a 2 groupes (import OR require) ; on garde le non-vide.
        imports = [a or b for a, b in _JS_IMPORT_RE.findall(code)]

        return VerificationResult(
            valid=True,
            language=self.language,
            structure={
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "lines": len(code.splitlines()),
            },
        )


# ─── SQL ─────────────────────────────────────────────────────────────────


class SqlVerifier(BaseVerifier):
    """
    Vérifieur SQL via `sqlparse`.

    LIMITATION HONNÊTE : `sqlparse` est un *parseur de structure*, pas un
    *validateur de syntaxe* strict. Il est très permissif et ne renvoie
    pas d'erreur sur du SQL malformé qui reste tokenisable. On s'en sert
    pour extraire la structure (nombre et types de statements) ; on
    considère le snippet "valide" s'il produit au moins une statement
    non triviale.

    Pour une vraie validation par dialecte, il faudra brancher un moteur
    cible (postgres EXPLAIN, sqlite parse, …) — Phase 2+ ou Phase 3.

    Détecte :
      - statements DML/DDL (SELECT, INSERT, UPDATE, DELETE, CREATE, …)
      - nombre de statements
      - présence de commentaires
    """

    language = "sql"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return _empty_result(self.language)

        try:
            import sqlparse  # noqa: PLC0415
        except ImportError:
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=["Dépendance 'sqlparse' manquante (cf. requirements.txt)"],
            )

        statements = [s for s in sqlparse.parse(code) if s.tokens]
        non_trivial = [
            s for s in statements
            if (s.get_type() or "UNKNOWN") not in ("UNKNOWN",)
        ]

        if not non_trivial:
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=[
                    "Aucune statement SQL reconnaissable détectée "
                    "(SELECT, INSERT, UPDATE, DELETE, CREATE, …)"
                ],
            )

        types = [s.get_type() for s in non_trivial]
        return VerificationResult(
            valid=True,
            language=self.language,
            warnings=[
                "sqlparse ne valide pas strictement la syntaxe — "
                "ce résultat confirme la structure, pas l'exécutabilité"
            ],
            structure={
                "statement_count": len(non_trivial),
                "statement_types": dict(Counter(types)),
                "lines": len(code.splitlines()),
            },
        )


# ─── HTML ────────────────────────────────────────────────────────────────


# Balises HTML auto-fermantes (void elements) — pas besoin de fermeture.
_HTML_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


class _StackHTMLParser(HTMLParser):
    """HTMLParser instrumenté pour détecter les balises non fermées."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[tuple[str, int]] = []
        self.tag_counts: Counter = Counter()
        self.unmatched_close: List[str] = []
        self.parse_errors: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.tag_counts[tag] += 1
        if tag not in _HTML_VOID_TAGS:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        # `<br/>` etc. — ne pas empiler.
        self.tag_counts[tag] += 1

    def handle_endtag(self, tag: str) -> None:
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        else:
            self.unmatched_close.append(tag)

    def error(self, message: str) -> None:  # pragma: no cover
        # html.parser appelle rarement error() en Python 3.5+ ; pour sûreté.
        self.parse_errors.append(message)


class HtmlVerifier(BaseVerifier):
    """
    Vérifieur HTML via `html.parser` (stdlib) + suivi de pile.

    `html.parser` est très permissif (HTML5 tolérant) — il ne lève
    pratiquement jamais d'erreur. On compense en suivant la pile des
    balises ouvertes/fermées pour détecter :
      - balises non fermées (à la fin du parse, pile non vide)
      - balises de fermeture sans ouverture correspondante
      - balises auto-fermantes correctement gérées (void elements)

    Détecte aussi :
      - présence de <html>, <head>, <body>
      - nombre total de balises
    """

    language = "html"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return _empty_result(self.language)

        parser = _StackHTMLParser()
        try:
            parser.feed(code)
            parser.close()
        except Exception as exc:  # pragma: no cover - html.parser ne lève quasi jamais
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=[f"Erreur de parse : {exc}"],
            )

        errors: List[str] = list(parser.parse_errors)
        for tag in parser.unmatched_close:
            errors.append(f"Balise </{tag}> sans ouverture correspondante")
        for tag, line in parser.stack:
            errors.append(f"Balise <{tag}> (ligne {line}) jamais fermée")

        valid = not errors
        return VerificationResult(
            valid=valid,
            language=self.language,
            errors=errors,
            structure={
                "tag_counts": dict(parser.tag_counts),
                "has_html_root": "html" in parser.tag_counts,
                "has_head": "head" in parser.tag_counts,
                "has_body": "body" in parser.tag_counts,
                "lines": len(code.splitlines()),
            },
        )


# ─── CSS ─────────────────────────────────────────────────────────────────


class CssVerifier(BaseVerifier):
    """
    Vérifieur CSS via `tinycss2`.

    Parse la feuille de style et remonte les tokens de type `error`
    produits par le parser. C'est une validation syntaxique CSS3 ; ne
    valide pas la sémantique (propriété inconnue, valeur invalide).

    Détecte :
      - erreurs de tokenisation / parse
      - règles `qualified` (sélecteur + bloc)
      - règles `at` (@media, @import, @keyframes, …)
    """

    language = "css"

    def verify(self, code: str) -> VerificationResult:
        if not code.strip():
            return _empty_result(self.language)

        try:
            import tinycss2  # noqa: PLC0415
        except ImportError:
            return VerificationResult(
                valid=False,
                language=self.language,
                errors=["Dépendance 'tinycss2' manquante (cf. requirements.txt)"],
            )

        rules = tinycss2.parse_stylesheet(code, skip_comments=True, skip_whitespace=True)

        errors: List[str] = []
        qualified = 0
        at_rules: List[str] = []

        for rule in rules:
            if rule.type == "error":
                errors.append(f"Ligne {rule.source_line}: {rule.message}")
            elif rule.type == "qualified-rule":
                qualified += 1
                # Erreurs éventuelles dans le bloc de déclarations.
                decls = tinycss2.parse_declaration_list(rule.content)
                for d in decls:
                    if d.type == "error":
                        errors.append(f"Ligne {d.source_line}: {d.message}")
            elif rule.type == "at-rule":
                at_rules.append(rule.at_keyword)

        valid = not errors
        return VerificationResult(
            valid=valid,
            language=self.language,
            errors=errors,
            structure={
                "qualified_rules": qualified,
                "at_rules": at_rules,
                "lines": len(code.splitlines()),
            },
        )


# ─── Registry ────────────────────────────────────────────────────────────


# Registry des verifieurs disponibles, indexes par nom de langage.
# Les alias (js → javascript, sh → bash, …) sont enregistrés en plus
# du nom canonique, pour matcher ce que les utilisateurs écrivent dans
# les fences markdown.
VERIFIERS: dict[str, BaseVerifier] = {
    "python": PythonVerifier(),
    "py": PythonVerifier(),
    "bash": BashVerifier(),
    "sh": BashVerifier(),
    "shell": BashVerifier(),
    "javascript": JavaScriptVerifier(),
    "js": JavaScriptVerifier(),
    "node": JavaScriptVerifier(),
    "sql": SqlVerifier(),
    "html": HtmlVerifier(),
    "css": CssVerifier(),
}


def get_verifier(language: str) -> Optional[BaseVerifier]:
    """Retourne le verifieur pour un langage, None si non supporte."""
    return VERIFIERS.get(language.lower())
