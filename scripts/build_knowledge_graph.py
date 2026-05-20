"""
Construit le Knowledge Graph d'Ogham à partir d'un répertoire de
documents (Markdown par défaut). Lit chaque fichier, applique
l'extracteur FR (PR 2), agrège dans un seul `KnowledgeGraph`, et
sauve en JSON versionné dans `data/models/knowledge_graph.json`.

Usage typique :
    .venv-uv/bin/python scripts/build_knowledge_graph.py

Usage CI : voir `.github/workflows/kg-build.yml`.

Le `.json` produit est gitignoré (option B : artefact reproductible
plutôt que checkpoint commité). Régénérable à tout moment depuis le
corpus tracké dans `data/knowledge/`.

Sortie code :
  - 0 si le KG produit dépasse `--min-entities` et `--min-relations`
  - 1 sinon (utilisé par la CI `kg-build.yml` comme garde-fou)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Sans cet ajout, le script ne peut pas importer `modules.*` quand on
# le lance depuis la racine du repo (cas typique en CI et en local).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ogham.extractor import populate_graph  # noqa: E402
from modules.ogham.knowledge_graph import KnowledgeGraph  # noqa: E402

logger = logging.getLogger("morrigan.ogham.build_kg")

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "knowledge_graph.json"
)

# Patterns markdown qu'on ne veut pas voir l'extracteur traiter comme
# du texte normal. On les retire AVANT extraction (sinon `# Titre`
# fait croire que `Titre` est une entité acronyme à cause du `#`
# attaché). Simple et conservateur.
_MD_HEADER_RE = re.compile(r"^#+\s*", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [texte](url) → texte
_MD_CODE_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")  # `code` → code
_MD_BULLET_RE = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)


def _clean_markdown(text: str) -> str:
    """Nettoie le markdown pour donner du texte à l'extracteur.

    - Retire les blocs de code (on n'y cherche pas d'entités).
    - Aplatit les links [texte](url) en gardant le texte.
    - Retire les marqueurs de headers, bullets, et l'inline code.
    """
    text = _MD_CODE_FENCE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    return text


def iter_source_files(source: Path, patterns: Iterable[str]) -> list[Path]:
    """Liste les fichiers matchant les patterns sous `source` (récursif)."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(source.rglob(pattern)))
    # Déduplique en gardant l'ordre.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def build(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        sys.exit(f"Source introuvable : {source}")

    files = iter_source_files(source, args.patterns)
    if not files:
        sys.exit(
            f"Aucun fichier matchant {args.patterns} sous {source}. "
            "Vérifie --source et --patterns."
        )

    kg = KnowledgeGraph()
    t0 = time.time()
    total_e, total_r = 0, 0

    for f in files:
        text = _clean_markdown(f.read_text(encoding="utf-8"))
        # source = chemin relatif à la racine source (lisible dans le KG)
        rel = f.relative_to(source) if f.is_relative_to(source) else f
        n_e, n_r = populate_graph(kg, text, source=str(rel))
        total_e += n_e
        total_r += n_r
        logger.info("%s : +%d entités, +%d relations", rel, n_e, n_r)

    elapsed = time.time() - t0

    logger.info("─" * 60)
    logger.info(
        "Fichiers traités : %d en %.1fs", len(files), elapsed,
    )
    logger.info(
        "Extractions cumulées : %d entités, %d relations "
        "(dédoublonnées dans le KG : %d entités, %d triplets)",
        total_e, total_r, len(kg), kg.relation_count,
    )

    # Top entités pour visibilité humaine — utile pour valider à l'œil
    # que le corpus est bien capturé.
    top = sorted(
        kg.entities(), key=lambda e: -len(kg.facts_about(e.id))
    )[: args.top]
    if top:
        logger.info("Top %d entités (par nombre de relations) :", args.top)
        for ent in top:
            n_facts = len(kg.facts_about(ent.id))
            logger.info("  %-30s (%s) : %d relations", ent.label, ent.type, n_facts)

    # Sauvegarde
    out = Path(args.output)
    kg.save(out)
    size_kb = out.stat().st_size / 1024
    logger.info("KG sauvé : %s (%.1f KB)", out, size_kb)

    # Garde-fous (utilisés par la CI pour catch un corpus vidé / un
    # extracteur cassé).
    if len(kg) < args.min_entities or kg.relation_count < args.min_relations:
        logger.error(
            "ÉCHEC : KG sous les seuils (%d entités < %d ou %d triplets < %d). "
            "Corpus vidé ou extracteur cassé ?",
            len(kg), args.min_entities, kg.relation_count, args.min_relations,
        )
        return 1

    logger.info(
        "OK : %d entités >= %d, %d triplets >= %d",
        len(kg), args.min_entities, kg.relation_count, args.min_relations,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--patterns", nargs="+", default=["*.md", "*.txt"],
        help="Globs des fichiers à ingérer (récursif). Défaut : *.md *.txt",
    )
    parser.add_argument(
        "--min-entities", type=int, default=50,
        help="Échec CI si moins d'entités produites (défaut 50)",
    )
    parser.add_argument(
        "--min-relations", type=int, default=100,
        help="Échec CI si moins de triplets produits (défaut 100)",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
