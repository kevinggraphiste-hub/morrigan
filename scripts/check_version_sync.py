"""Vérifie la cohérence version Morrigan ↔ CHANGELOG ↔ tag git.

Source de vérité : `core/__version__.py` (la variable `__version__`).
Première section versionnée du `CHANGELOG.md` (la plus haute `## [X.Y.Z]`,
en ignorant `## [Non publié]` / `## [Unreleased]`).

Usage :
    python scripts/check_version_sync.py
    python scripts/check_version_sync.py --tag v0.2.0

Sortie : code 0 si tout matche, code 1 avec un message clair sinon.
Utilisé par `.github/workflows/version-sync-check.yml` (sans `--tag`) et
par `.github/workflows/release.yml` (avec `--tag ${{ github.ref_name }}`).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "core" / "__version__.py"
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"

# Versions "non publiées" à ignorer quand on cherche la première
# entrée versionnée du changelog.
UNRELEASED_LABELS = {"non publié", "non publie", "unreleased"}

VERSION_RE = re.compile(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE)
CHANGELOG_HEADER_RE = re.compile(r"^## \[([^\]]+)\](?:\s*-\s*(\S+))?", re.MULTILINE)


def read_module_version() -> str:
    """Extrait la chaîne `__version__` de `core/__version__.py`."""
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        sys.exit(
            f"ERREUR : aucune ligne `__version__ = \"...\"` trouvée dans "
            f"{VERSION_FILE.relative_to(REPO_ROOT)}."
        )
    return match.group(1)


def read_first_changelog_version() -> str:
    """Renvoie la première section versionnée du CHANGELOG, hors `[Non publié]`."""
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    for match in CHANGELOG_HEADER_RE.finditer(text):
        label = match.group(1).strip()
        if label.lower() in UNRELEASED_LABELS:
            continue
        return label
    sys.exit(
        f"ERREUR : aucune section `## [X.Y.Z] - YYYY-MM-DD` trouvée dans "
        f"{CHANGELOG_FILE.relative_to(REPO_ROOT)}."
    )


def normalize_tag(tag: str) -> str:
    """Enlève un éventuel préfixe `v` du tag pour comparaison."""
    return tag[1:] if tag.startswith("v") else tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help=(
            "Tag git à valider (ex: v0.2.0). Si fourni, doit correspondre à "
            "`__version__` ET à la première section versionnée du CHANGELOG."
        ),
    )
    args = parser.parse_args()

    module_version = read_module_version()
    changelog_version = read_first_changelog_version()

    print(f"core/__version__.py    : {module_version}")
    print(f"CHANGELOG.md (1re ver.) : {changelog_version}")
    if args.tag:
        print(f"git tag                : {args.tag}")

    errors: list[str] = []

    if module_version != changelog_version:
        errors.append(
            f"`__version__` ({module_version}) ne correspond pas à la "
            f"première section versionnée du CHANGELOG ({changelog_version}). "
            f"Avant de tagger, ajoute une section "
            f"`## [{module_version}] - YYYY-MM-DD` au CHANGELOG, "
            f"ou bump `core/__version__.py`."
        )

    if args.tag:
        tag_version = normalize_tag(args.tag)
        if tag_version != module_version:
            errors.append(
                f"Le tag git ({args.tag}) ne correspond pas à `__version__` "
                f"({module_version}). Bump `core/__version__.py` à "
                f"{tag_version} avant de retaguer."
            )
        if tag_version != changelog_version:
            errors.append(
                f"Le tag git ({args.tag}) ne correspond pas à la première "
                f"section versionnée du CHANGELOG ({changelog_version})."
            )

    if errors:
        print()
        for err in errors:
            print(f"  ✗ {err}")
        return 1

    print()
    print("✓ Tout est aligné.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
