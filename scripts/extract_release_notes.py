"""Extrait la section CHANGELOG correspondant à une version donnée.

Utilisé par `.github/workflows/release.yml` pour fournir le corps de la
GitHub Release. Lit `CHANGELOG.md`, trouve la section `## [X.Y.Z]`
correspondant au tag/version passé, et imprime son contenu (sans la
ligne d'en-tête) sur la sortie standard.

Usage :
    python scripts/extract_release_notes.py v0.2.0
    python scripts/extract_release_notes.py 0.2.0

Sort en erreur (code 1) si la section n'existe pas — un tag posé sans
section CHANGELOG correspondante ne doit pas produire une release vide.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"


def normalize(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def extract_section(version: str) -> str:
    """Renvoie le contenu de la section `## [version]` jusqu'au prochain `##`."""
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    header_re = re.compile(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = header_re.search(text)
    if not match:
        sys.exit(
            f"ERREUR : aucune section `## [{version}]` trouvée dans "
            f"{CHANGELOG_FILE.relative_to(REPO_ROOT)}. Ajoute-la avant "
            f"de retaguer (voir la mémoire `gungnir-release-changelog-gotcha`)."
        )
    return match.group(1).strip()


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("Usage : python scripts/extract_release_notes.py <vX.Y.Z|X.Y.Z>")
    version = normalize(sys.argv[1])
    print(extract_section(version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
