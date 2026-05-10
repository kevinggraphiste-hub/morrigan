"""
MORRIGAN-CODE — Agent specialise code (Phase 2).

Vérifie la cohérence syntaxique de chunks de code recuperes ou generes,
par langage (Python en premier, puis JS/TS, Bash, SQL, HTML/CSS).

Premier agent specialise : il s'attache a un domaine et lui apporte
des outils de verification (AST par langage) que le pipeline generaliste
de Morrigan ne peut pas offrir tout seul.
"""

from modules.morrigan_code.module import MorriganCode

__all__ = ["MorriganCode"]
