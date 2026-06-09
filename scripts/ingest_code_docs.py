"""
Ingestion d'un corpus de documentation CODE dans un index Danann persisté (Phase 2B).

Deux sources (décisions brainstorm 2026-06-09) :
  1. Bundle texte officiel Python (docs.python.org) — tutorial / howto / faq /
     (library en option). Riche, conversationnel, licence PSF. Téléchargé une
     fois puis réutilisé hors-ligne.
  2. Introspection `pydoc` de modules stdlib curatés — couverture API exhaustive.

Chunker **CODE-AWARE** : préserve l'indentation et les sauts de ligne. Le chunker
markdown générique de `ingest_knowledge.py` écrase les espaces (`\\s+`→` `), ce
qui détruirait le code ; on ne le réutilise donc pas ici. On découpe par
section (titres soulignés du format Sphinx-text) en gardant les blocs entiers,
sous la limite ~512 tokens de l'embedder e5-small.

Sortie : index Danann compressé **int8** persisté (défaut `data/models/index_code/`),
servi au runtime via `MORRIGAN_INDEX`. Corpus EN interrogeable en FR grâce à
l'embedder multilingue (Phase 2A).

Usage :
    .venv-uv/bin/python scripts/ingest_code_docs.py
    .venv-uv/bin/python scripts/ingest_code_docs.py --categories tutorial,library,howto,faq
    .venv-uv/bin/python scripts/ingest_code_docs.py --no-fetch --pydoc-modules none
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import re
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("morrigan.ingest_code")

DEFAULT_BUNDLE_DIR = Path("data/code_docs/python-text")
DEFAULT_OUTPUT = Path("data/models/index_code")
DOWNLOAD_PAGE = "https://docs.python.org/3/download.html"

# library exclu par défaut (énorme — dizaines de milliers de chunks) : on
# démarre sur le prose riche + pydoc, on ajoute `library` explicitement ensuite.
DEFAULT_CATEGORIES = ("tutorial", "howto", "faq")

# Modules stdlib à fort intérêt Q/R, pour la couverture API via pydoc.
DEFAULT_PYDOC_MODULES = (
    "os", "sys", "json", "re", "datetime", "collections", "itertools",
    "functools", "pathlib", "subprocess", "typing", "math", "random",
    "string", "io", "argparse", "logging", "asyncio", "dataclasses", "enum",
)

MAX_CHARS = 1500          # ~370 tokens, sous la limite 512 d'e5-small
MIN_CHUNK_CHARS = 40

# Souligné de titre dans le format texte Sphinx (==, --, ~~, **, etc.).
_UNDERLINE_RE = re.compile(r'^[=\-~^"\'*+#.`]{3,}\s*$')


# ─── Fetch du bundle texte officiel ───────────────────────────────────


def resolve_text_bundle_url(page: str = DOWNLOAD_PAGE) -> str:
    """Trouve l'URL de l'archive `*-docs-text.tar.bz2` sur la page de download
    (auto-suit la version stable courante, pas de version à coder en dur)."""
    with urllib.request.urlopen(page, timeout=30) as r:  # noqa: S310
        html = r.read().decode("utf-8", "replace")
    m = re.search(r'href="([^"]*-docs-text\.tar\.bz2)"', html)
    if not m:
        raise RuntimeError("Lien du bundle texte introuvable sur la page de download.")
    return urllib.parse.urljoin(page, m.group(1))


def fetch_python_text_bundle(dest: Path = DEFAULT_BUNDLE_DIR) -> Path:
    """Télécharge + extrait le bundle texte Python dans `dest` (idempotent)."""
    dest = Path(dest)
    if dest.exists() and next(dest.rglob("*.txt"), None) is not None:
        logger.info("Bundle déjà présent (%s) → pas de re-téléchargement", dest)
        return dest
    url = resolve_text_bundle_url()
    logger.info("Téléchargement du bundle texte : %s", url)
    with urllib.request.urlopen(url, timeout=180) as r:  # noqa: S310
        data = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
        tar.extractall(dest, filter="data")  # filter=data : extraction sûre (3.12+)
    logger.info("Bundle extrait dans %s (%.1f Mo)", dest, len(data) / 1e6)
    return dest


def _bundle_root(bundle_dir: Path) -> Path:
    """Le tar extrait sous `python-X.Y.Z-docs-text/` : on retourne ce sous-dossier."""
    roots = [p for p in bundle_dir.iterdir() if p.is_dir() and p.name.startswith("python-")]
    return roots[0] if roots else bundle_dir


def iter_bundle_docs(
    bundle_dir: Path, categories: Tuple[str, ...]
) -> Iterator[Tuple[str, str]]:
    """Itère (origine, texte) sur les .txt des catégories sélectionnées."""
    top = _bundle_root(Path(bundle_dir))
    for cat in categories:
        catdir = top / cat
        if not catdir.exists():
            logger.warning("Catégorie absente du bundle : %s", cat)
            continue
        for txt in sorted(catdir.rglob("*.txt")):
            try:
                content = txt.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            yield f"{cat}/{txt.relative_to(catdir).as_posix()}", content


# ─── Introspection pydoc ──────────────────────────────────────────────


def extract_pydoc(modules: Tuple[str, ...]) -> Iterator[Tuple[str, str]]:
    """Itère (origine, texte) sur le rendu pydoc plaintext des modules donnés."""
    import pydoc  # noqa: PLC0415

    for name in modules:
        try:
            mod = importlib.import_module(name)
            text = pydoc.render_doc(mod, renderer=pydoc.plaintext)
        except Exception as e:  # import/render peut échouer selon la plateforme
            logger.warning("pydoc %s ignoré : %s", name, e)
            continue
        if text and text.strip():
            yield f"pydoc/{name}", text


# ─── Chunking code-aware ──────────────────────────────────────────────


def _to_units(text: str) -> List[Tuple[str, str]]:
    """Découpe le texte en (section, bloc) — blocs séparés par lignes vides,
    titre courant suivi selon les soulignés Sphinx. Préserve les newlines."""
    lines = text.splitlines()
    units: List[Tuple[str, str]] = []
    section = ""
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        block = "\n".join(buf).rstrip()
        if block.strip():
            units.append((section, block))
        buf = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < n else ""
        title = line.strip()
        # Titre : ligne non vide suivie d'un souligné d'au moins sa demi-longueur.
        if title and _UNDERLINE_RE.match(nxt) and len(nxt.strip()) >= max(3, len(title) // 2):
            flush()
            section = title
            i += 2
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        buf.append(line)
        i += 1
    flush()
    return units


def _hard_split(block: str, max_chars: int) -> List[str]:
    """Découpe un bloc trop long sur les frontières de ligne (garde l'indent)."""
    out: List[str] = []
    cur: List[str] = []
    length = 0
    for line in block.split("\n"):
        if cur and length + len(line) + 1 > max_chars:
            out.append("\n".join(cur))
            cur, length = [], 0
        cur.append(line)
        length += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def chunk_code_doc(
    text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHUNK_CHARS
) -> List[Tuple[str, str]]:
    """Découpe un document en (chunk, section), code-aware.

    - Préserve indentation/newlines (ne collapse PAS les espaces).
    - Empaquette des blocs entiers jusqu'à `max_chars`, sans traverser une
      frontière de section (un chunk = une seule section).
    - Un bloc plus long que `max_chars` (gros exemple) est coupé par lignes.
    """
    units = _to_units(text)
    chunks: List[Tuple[str, str]] = []
    cur: List[str] = []
    cur_section = ""
    cur_len = 0

    def emit() -> None:
        nonlocal cur, cur_len
        if cur:
            # strip("\n") seulement : préserve l'indentation de tête (code),
            # les blocs étant déjà rstrip'd et sans ligne vide encadrante.
            body = "\n\n".join(cur).strip("\n")
            if len(body.strip()) >= min_chars:
                chunks.append((body, cur_section))
        cur, cur_len = [], 0

    for section, block in units:
        pieces = _hard_split(block, max_chars) if len(block) > max_chars else [block]
        for piece in pieces:
            if cur and (section != cur_section or cur_len + len(piece) + 2 > max_chars):
                emit()
            if not cur:
                cur_section = section
            cur.append(piece)
            cur_len += len(piece) + 2
    emit()
    return chunks


# ─── Ingestion → index persisté ───────────────────────────────────────


def _meta(origin: str, section: str, source: str) -> dict:
    return {
        "domain": "code",
        "language": "python",
        "source": source,
        "origin": origin,
        "section": section,
        "type": "code_doc",
        "confidence": 0.9,
    }


def build_index(
    danann,
    *,
    bundle_dir: Path,
    categories: Tuple[str, ...],
    pydoc_modules: Tuple[str, ...],
    max_files: int | None,
    max_chars: int,
) -> int:
    """Ingère bundle + pydoc dans `danann` (incrémental, par document)."""
    total = 0
    files = 0

    def sources() -> Iterator[Tuple[str, str, str]]:
        for origin, text in iter_bundle_docs(bundle_dir, categories):
            yield origin, text, "python-docs"
        for origin, text in extract_pydoc(pydoc_modules):
            yield origin, text, "pydoc"

    for origin, text, source in sources():
        if max_files is not None and files >= max_files:
            logger.info("Limite --max-files (%d) atteinte", max_files)
            break
        chunks = chunk_code_doc(text, max_chars=max_chars)
        if not chunks:
            continue
        texts = [c for c, _ in chunks]
        metas = [_meta(origin, sec, source) for _, sec in chunks]
        total += danann.index(texts, metas)
        files += 1
        if files % 20 == 0:
            logger.info("  %d docs, %d chunks…", files, total)

    logger.info("Ingéré : %d documents, %d chunks", files, total)
    return total


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--compression", default="int8", choices=["none", "int8", "binary"])
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="Catégories du bundle (ex: tutorial,library,howto,faq).",
    )
    parser.add_argument(
        "--pydoc-modules", default=",".join(DEFAULT_PYDOC_MODULES),
        help="Modules stdlib pour pydoc, ou 'none'.",
    )
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--max-files", type=int, default=None, help="Cap de docs (test rapide).")
    parser.add_argument("--min-chunks", type=int, default=1)
    parser.add_argument("--no-fetch", action="store_true", help="Ne pas (re)télécharger le bundle.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from modules.danann.store import Danann  # noqa: PLC0415

    bundle_dir = Path(args.bundle_dir)
    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    pydoc_modules = (
        () if args.pydoc_modules.strip().lower() == "none"
        else tuple(m.strip() for m in args.pydoc_modules.split(",") if m.strip())
    )

    if not args.no_fetch and categories:
        fetch_python_text_bundle(bundle_dir)

    danann = Danann(compression=args.compression, use_reranker=False)
    total = build_index(
        danann, bundle_dir=bundle_dir, categories=categories,
        pydoc_modules=pydoc_modules, max_files=args.max_files, max_chars=args.max_chars,
    )

    if total < args.min_chunks:
        sys.exit(f"ÉCHEC : {total} chunks < {args.min_chunks}.")

    out = Path(args.output)
    danann.save_index(out)

    mem = danann.memory_bytes()
    float_equiv = danann.count() * 384 * 4
    ratio = float_equiv / mem if mem else 0.0
    print("─" * 60)
    print(f"Sources     : bundle {categories} + pydoc ({len(pydoc_modules)} modules)")
    print(f"Chunks      : {danann.count()}")
    print(f"Compression : {args.compression}")
    print(f"Index RAM   : {mem/1024:.1f} KB (float32 équiv ~{float_equiv/1024:.1f} KB → ×{ratio:.1f})")
    print(f"Sauvé       : {out}/  (corpus.json + vectors.npz)")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
