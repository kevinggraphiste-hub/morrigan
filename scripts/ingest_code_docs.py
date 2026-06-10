"""
Ingestion d'un corpus de documentation CODE dans un index Danann persisté (Phase 2B).

**Registre de sources multi-langage** (`iter_source`) — le chunker, l'ingestion
et l'index sont partagés ; ajouter un langage = brancher une source. Sources
actuelles (`--sources`) :
  - `python` : bundle texte officiel (docs.python.org, tutorial/howto/faq/library)
    + introspection `pydoc` de modules stdlib → langage `python`.
  - `man` : pages man locales (bash, git, grep, sed, awk, find…) — source
    **offline souveraine**, langages `bash` / `git` / `shell`.
  - `mdn` : docs web MDN (`mdn/content`, sparse clone git limité à
    `files/en-us/web/{javascript,css,html}`) → langages `javascript` /
    `css` / `html`. Markdown : front-matter + macros Kuma nettoyés,
    chunking sur titres `##` (hors code-fences).
  - `docker` : docs officielles Docker (`docker/docs`, sparse clone limité à
    `content/{get-started,manuals,reference}`) → langage `docker`.
    Markdown Hugo : front-matter + shortcodes `{{< … >}}`/`{{% … %}}` nettoyés.
  - `postgresql` : doc HTML **pré-buildée** du tarball docs officiel
    (`postgresql-X.Y-docs.tar.gz`, auto-suit la dernière version stable) →
    langage `sql`. HTML converti en pseudo-markdown code-aware (`<pre>` →
    fences, `<hN>` → titres `#`), pages scopées par préfixe (`--pg-prefixes`).

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
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from html.parser import HTMLParser
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

# man pages par défaut (source locale offline souveraine) : shell + git + CLI.
DEFAULT_MAN_PAGES = (
    "bash", "grep", "sed", "awk", "find", "xargs", "tar", "curl", "ssh", "rsync",
    "jq", "make", "git", "git-commit", "git-rebase", "git-merge", "git-log",
    "git-branch", "git-checkout", "git-stash", "git-remote", "git-reset",
    "git-cherry-pick", "git-bisect",
)

# MDN : sparse clone du repo de contenu officiel, limité aux aires utiles.
DEFAULT_MDN_DIR = Path("data/code_docs/mdn")
MDN_REPO_URL = "https://github.com/mdn/content.git"
DEFAULT_MDN_AREAS = ("javascript", "css", "html")

# Docker : sparse clone du repo de docs officiel, limité aux aires utiles
# (`guides` = tutoriels par cas d'usage, exclu par défaut ; `includes` = partials).
DEFAULT_DOCKER_DIR = Path("data/code_docs/docker-docs")
DOCKER_REPO_URL = "https://github.com/docker/docs.git"
DEFAULT_DOCKER_AREAS = ("get-started", "manuals", "reference")

# PostgreSQL : doc HTML pré-buildée embarquée dans le tarball source officiel.
DEFAULT_PG_DIR = Path("data/code_docs/postgres-html")
PG_SOURCE_LISTING = "https://ftp.postgresql.org/pub/source/"
# Préfixes de pages ingérés par défaut : le cœur SQL utilisable en Q/R
# (tutoriel, référence SQL `sql-*`, types, requêtes, DDL/DML, fonctions,
# index, perfs) — pas les internals/protocole/API C.
DEFAULT_PG_PREFIXES = (
    "tutorial", "sql", "datatype", "queries", "ddl", "dml",
    "functions", "indexes", "textsearch", "performance",
)

# Sources disponibles dans le registre (cf. iter_source).
ALL_SOURCES = ("python", "man", "mdn", "docker", "postgresql")

# Sources dont les documents sont du markdown (chunking sur titres `#`).
# `postgresql` : le HTML est converti en pseudo-markdown (titres #, fences).
MARKDOWN_SOURCES = {"mdn", "docker", "postgresql"}

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


# ─── man pages (source locale, offline) ───────────────────────────────


_BACKSPACE_RE = re.compile(r".\x08")  # overstrike `X\x08X` / `_\x08X` → garde le 2e


def man_language(page: str) -> str:
    """Langage logique d'une page man (pour la métadonnée + le filtrage)."""
    if page == "bash":
        return "bash"
    if page == "git" or page.startswith("git-"):
        return "git"
    return "shell"


def render_man(page: str) -> str | None:
    """Rend une page man en texte propre (overstrike retiré), ou None si absente."""
    if shutil.which("man") is None:
        return None
    if subprocess.run(["man", "-w", page], capture_output=True).returncode != 0:
        return None
    env = dict(os.environ, MANWIDTH="80", LC_ALL="C")
    try:
        out = subprocess.run(
            ["man", page], capture_output=True, text=True, env=env, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return _BACKSPACE_RE.sub("", out.stdout)


def extract_man(pages: Tuple[str, ...]) -> Iterator[Tuple[str, str, str]]:
    """Itère (origine, texte, langage) sur les pages man disponibles."""
    for page in pages:
        text = render_man(page)
        if text:
            yield f"man/{page}", text, man_language(page)
        else:
            logger.warning("man %s indisponible → ignoré", page)


# ─── MDN (mdn/content, sparse clone) ──────────────────────────────────


_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
# Ligne entièrement macro(s) Kuma ({{Compat}}, {{Specifications}}, sidebars…).
_MACRO_LINE_RE = re.compile(r"^\s*(\{\{[^{}]*\}\}\s*)+$")
# Macros xref inline : on garde l'argument cité ({{jsxref("Array")}} → `Array`).
_MACRO_XREF_RE = re.compile(r'\{\{\s*\w+\(\s*"([^"]+)"[^{}]*\)\s*\}\}')
_MACRO_ANY_RE = re.compile(r"\{\{[^{}]*\}\}")


def mdn_language(area: str) -> str:
    """Langage logique d'une aire MDN (`files/en-us/web/<area>`)."""
    return {"api": "javascript", "http": "http"}.get(area, area)


def parse_mdn_page(raw: str) -> Tuple[str, str]:
    """(titre, corps nettoyé) d'une page MDN : front-matter YAML extrait,
    macros Kuma retirées (les xref gardent leur argument)."""
    title = ""
    body = raw
    m = _FRONT_MATTER_RE.match(raw)
    if m:
        body = raw[m.end():]
        tm = re.search(r"^title:\s*(.+)$", m.group(1), re.MULTILINE)
        if tm:
            title = tm.group(1).strip().strip("'\"")
    body = "\n".join(l for l in body.splitlines() if not _MACRO_LINE_RE.match(l))
    body = _MACRO_XREF_RE.sub(lambda mm: f"`{mm.group(1)}`", body)
    body = _MACRO_ANY_RE.sub("", body)
    return title, body


def fetch_mdn_content(
    dest: Path = DEFAULT_MDN_DIR, areas: Tuple[str, ...] = DEFAULT_MDN_AREAS
) -> Path:
    """Sparse clone de mdn/content limité aux aires demandées (idempotent)."""
    dest = Path(dest)
    web = dest / "files" / "en-us" / "web"
    if web.exists() and next(web.rglob("index.md"), None) is not None:
        logger.info("Contenu MDN déjà présent (%s) → pas de re-clone", dest)
        return dest
    if shutil.which("git") is None:
        raise RuntimeError("git requis pour la source mdn (sparse clone).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Sparse clone MDN (%s) → %s", ",".join(areas), dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MDN_REPO_URL, str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set",
         *(f"files/en-us/web/{a}" for a in areas)],
        check=True,
    )
    return dest


def iter_mdn_docs(
    mdn_dir: Path, areas: Tuple[str, ...]
) -> Iterator[Tuple[str, str, str]]:
    """Itère (origine, texte, langage) sur les pages `index.md` des aires MDN."""
    web = Path(mdn_dir) / "files" / "en-us" / "web"
    for area in areas:
        adir = web / area
        if not adir.exists():
            logger.warning("Aire MDN absente : %s", area)
            continue
        for md in sorted(adir.rglob("index.md")):
            try:
                raw = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            title, body = parse_mdn_page(raw)
            if title:
                body = f"{title}\n\n{body}"
            rel = md.parent.relative_to(web).as_posix()
            yield f"mdn/{rel}", body, mdn_language(area)


# ─── Docker (docker/docs, sparse clone, markdown Hugo) ────────────────


# Ligne entièrement shortcode(s) Hugo ({{< tabs >}}, {{% include %}}…).
_SHORTCODE_LINE_RE = re.compile(r"^\s*(\{\{[<%][^{}]*[>%]\}\}\s*)+$")
_SHORTCODE_ANY_RE = re.compile(r"\{\{[<%][^{}]*[>%]\}\}")


def parse_docker_page(raw: str) -> Tuple[str, str]:
    """(titre, corps nettoyé) d'une page docker/docs : front-matter Hugo
    extrait, shortcodes `{{< … >}}` / `{{% … %}}` retirés."""
    title = ""
    body = raw
    m = _FRONT_MATTER_RE.match(raw)
    if m:
        body = raw[m.end():]
        tm = re.search(r"^title:\s*(.+)$", m.group(1), re.MULTILINE)
        if tm:
            title = tm.group(1).strip().strip("'\"")
    body = "\n".join(l for l in body.splitlines() if not _SHORTCODE_LINE_RE.match(l))
    body = _SHORTCODE_ANY_RE.sub("", body)
    return title, body


def fetch_docker_docs(
    dest: Path = DEFAULT_DOCKER_DIR, areas: Tuple[str, ...] = DEFAULT_DOCKER_AREAS
) -> Path:
    """Sparse clone de docker/docs limité aux aires demandées (idempotent)."""
    dest = Path(dest)
    content = dest / "content"
    if content.exists() and next(content.rglob("*.md"), None) is not None:
        logger.info("Contenu docker/docs déjà présent (%s) → pas de re-clone", dest)
        return dest
    if shutil.which("git") is None:
        raise RuntimeError("git requis pour la source docker (sparse clone).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Sparse clone docker/docs (%s) → %s", ",".join(areas), dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         DOCKER_REPO_URL, str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set",
         *(f"content/{a}" for a in areas)],
        check=True,
    )
    return dest


def iter_docker_docs(
    docker_dir: Path, areas: Tuple[str, ...]
) -> Iterator[Tuple[str, str, str]]:
    """Itère (origine, texte, langage) sur les pages markdown docker/docs."""
    content = Path(docker_dir) / "content"
    for area in areas:
        adir = content / area
        if not adir.exists():
            logger.warning("Aire docker/docs absente : %s", area)
            continue
        for md in sorted(adir.rglob("*.md")):
            try:
                raw = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            title, body = parse_docker_page(raw)
            if title:
                body = f"{title}\n\n{body}"
            rel = md.relative_to(content).as_posix()
            yield f"docker/{rel}", body, "docker"


# ─── PostgreSQL (doc HTML pré-buildée du tarball source) ──────────────


def resolve_pg_tarball_url(listing: str = PG_SOURCE_LISTING) -> str:
    """URL du tarball **docs** (HTML pré-buildé, ~4 Mo) de la dernière version
    stable (vX.Y, pas de beta/rc — auto-suit la version, rien à coder en dur).
    ⚠️ le tarball *source* ne contient plus que les SGML, pas le HTML."""
    with urllib.request.urlopen(listing, timeout=30) as r:  # noqa: S310
        html = r.read().decode("utf-8", "replace")
    versions = re.findall(r'href="v(\d+)\.(\d+)/"', html)
    if not versions:
        raise RuntimeError("Aucune version stable trouvée sur le listing PostgreSQL.")
    major, minor = max((int(a), int(b)) for a, b in versions)
    return f"{listing}v{major}.{minor}/postgresql-{major}.{minor}-docs.tar.gz"


def fetch_postgres_html(dest: Path = DEFAULT_PG_DIR) -> Path:
    """Télécharge le tarball source PostgreSQL et en extrait la doc HTML
    pré-buildée (`*/doc/src/sgml/html/*.html`) à plat dans `dest` (idempotent)."""
    dest = Path(dest)
    if dest.exists() and next(dest.glob("*.html"), None) is not None:
        logger.info("Doc PostgreSQL déjà présente (%s) → pas de re-téléchargement", dest)
        return dest
    url = resolve_pg_tarball_url()
    logger.info("Téléchargement du tarball PostgreSQL : %s", url)
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310
        data = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if "/doc/src/sgml/html/" not in member.name or not member.name.endswith(".html"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            (dest / Path(member.name).name).write_bytes(f.read())
            extracted += 1
    if not extracted:
        raise RuntimeError("Aucune page HTML trouvée dans le tarball PostgreSQL.")
    logger.info("Doc PostgreSQL extraite : %d pages → %s", extracted, dest)
    return dest


class _HTMLToText(HTMLParser):
    """HTML → pseudo-markdown code-aware : `<hN>` → titres `#`, `<pre>` →
    code-fences (contenu verbatim), navigation (`navheader`/`navfooter`)
    et `<script>/<style>` ignorés. Compatible `chunk_code_doc(markdown=True)`."""

    _BLOCK_TAGS = {"p", "div", "table", "ul", "ol", "dl", "dt", "dd", "blockquote", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        # Pile du skip : on mémorise le tag ouvrant et on ne dépile que sur
        # SA balise fermante (un `</table>` imbriqué ne déséquilibre rien).
        self._skip_stack: List[str] = []
        self._pre = 0
        self._heading: int | None = None
        self._hbuf: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._skip_stack:
            if tag == self._skip_stack[-1]:
                self._skip_stack.append(tag)  # imbrication du même tag
            return
        cls = dict(attrs).get("class", "") or ""
        if tag in ("script", "style") or "navheader" in cls or "navfooter" in cls:
            self._skip_stack.append(tag)
            return
        if tag == "pre":
            self._pre += 1
            self.parts.append("\n```\n")
        elif len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._heading = int(tag[1])
            self._hbuf = []
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append("  ")
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            if tag == self._skip_stack[-1]:
                self._skip_stack.pop()
            return
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
            self.parts.append("\n```\n")
        elif self._heading is not None and tag == f"h{self._heading}":
            title = re.sub(r"\s+", " ", "".join(self._hbuf)).strip()
            title = re.sub(r"\s*#$", "", title)  # « # » du lien d'ancre PostgreSQL
            self.parts.append(f"\n\n{'#' * self._heading} {title}\n\n")
            self._heading = None
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._heading is not None:
            self._hbuf.append(data)
        elif self._pre:
            self.parts.append(data)          # verbatim : code préservé
        else:
            self.parts.append(re.sub(r"[ \t]+", " ", data))

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(raw: str) -> str:
    """Convertit une page HTML en pseudo-markdown code-aware."""
    parser = _HTMLToText()
    parser.feed(raw)
    return parser.text()


def iter_postgres_docs(
    pg_dir: Path, prefixes: Tuple[str, ...]
) -> Iterator[Tuple[str, str, str]]:
    """Itère (origine, texte, langage) sur les pages HTML PostgreSQL scopées."""
    pg_dir = Path(pg_dir)
    if not pg_dir.exists():
        logger.warning("Doc PostgreSQL absente : %s", pg_dir)
        return
    for page in sorted(pg_dir.glob("*.html")):
        stem = page.stem
        if prefixes and not any(
            stem == p or stem.startswith(f"{p}-") for p in prefixes
        ):
            continue
        try:
            raw = page.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        yield f"postgres/{stem}", html_to_text(raw), "sql"


# ─── Chunking code-aware ──────────────────────────────────────────────


_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _to_units(text: str, markdown: bool = False) -> List[Tuple[str, str]]:
    """Découpe le texte en (section, bloc) — blocs séparés par lignes vides,
    titre courant suivi selon les soulignés Sphinx. Préserve les newlines.

    `markdown=True` reconnaît AUSSI les titres `#`…`######`, mais jamais à
    l'intérieur d'un code-fence (un `# commentaire` bash n'est pas un titre)."""
    lines = text.splitlines()
    units: List[Tuple[str, str]] = []
    section = ""
    buf: List[str] = []
    in_fence = False

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
        if markdown and _MD_FENCE_RE.match(line):
            in_fence = not in_fence
        if markdown and not in_fence:
            hm = _MD_HEADING_RE.match(line)
            if hm:
                flush()
                section = hm.group(1)
                i += 1
                continue
        nxt = lines[i + 1] if i + 1 < n else ""
        title = line.strip()
        # Titre : ligne non vide suivie d'un souligné d'au moins sa demi-longueur.
        if (
            not in_fence and title and _UNDERLINE_RE.match(nxt)
            and len(nxt.strip()) >= max(3, len(title) // 2)
        ):
            flush()
            section = title
            i += 2
            continue
        if not line.strip() and not in_fence:
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
    text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHUNK_CHARS,
    markdown: bool = False,
) -> List[Tuple[str, str]]:
    """Découpe un document en (chunk, section), code-aware.

    - Préserve indentation/newlines (ne collapse PAS les espaces).
    - Empaquette des blocs entiers jusqu'à `max_chars`, sans traverser une
      frontière de section (un chunk = une seule section).
    - Un bloc plus long que `max_chars` (gros exemple) est coupé par lignes.
    - `markdown=True` : sections aussi sur titres `#`, hors code-fences.
    """
    units = _to_units(text, markdown=markdown)
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


def _meta(origin: str, section: str, source: str, language: str) -> dict:
    return {
        "domain": "code",
        "language": language,
        "source": source,
        "origin": origin,
        "section": section,
        "type": "code_doc",
        "confidence": 0.9,
    }


def iter_source(
    name: str, *, bundle_dir: Path, categories: Tuple[str, ...],
    pydoc_modules: Tuple[str, ...], man_pages: Tuple[str, ...],
    mdn_dir: Path = DEFAULT_MDN_DIR, mdn_areas: Tuple[str, ...] = DEFAULT_MDN_AREAS,
    docker_dir: Path = DEFAULT_DOCKER_DIR,
    docker_areas: Tuple[str, ...] = DEFAULT_DOCKER_AREAS,
    pg_dir: Path = DEFAULT_PG_DIR,
    pg_prefixes: Tuple[str, ...] = DEFAULT_PG_PREFIXES,
) -> Iterator[Tuple[str, str, str, str]]:
    """Registre de sources → itère (origine, texte, source, langage).

    Ajouter un langage = brancher une nouvelle source ici ; le chunker,
    l'ingestion et l'index restent partagés.
    """
    if name == "python":
        for origin, text in iter_bundle_docs(bundle_dir, categories):
            yield origin, text, "python-docs", "python"
        for origin, text in extract_pydoc(pydoc_modules):
            yield origin, text, "pydoc", "python"
    elif name == "man":
        for origin, text, language in extract_man(man_pages):
            yield origin, text, "man", language
    elif name == "mdn":
        for origin, text, language in iter_mdn_docs(mdn_dir, mdn_areas):
            yield origin, text, "mdn", language
    elif name == "docker":
        for origin, text, language in iter_docker_docs(docker_dir, docker_areas):
            yield origin, text, "docker", language
    elif name == "postgresql":
        for origin, text, language in iter_postgres_docs(pg_dir, pg_prefixes):
            yield origin, text, "postgresql", language
    else:
        raise ValueError(f"Source inconnue : {name!r} (dispo : {ALL_SOURCES})")


def build_index(
    danann,
    *,
    sources: Tuple[str, ...],
    bundle_dir: Path,
    categories: Tuple[str, ...],
    pydoc_modules: Tuple[str, ...],
    man_pages: Tuple[str, ...],
    mdn_dir: Path = DEFAULT_MDN_DIR,
    mdn_areas: Tuple[str, ...] = DEFAULT_MDN_AREAS,
    docker_dir: Path = DEFAULT_DOCKER_DIR,
    docker_areas: Tuple[str, ...] = DEFAULT_DOCKER_AREAS,
    pg_dir: Path = DEFAULT_PG_DIR,
    pg_prefixes: Tuple[str, ...] = DEFAULT_PG_PREFIXES,
    max_files: int | None,
    max_chars: int,
) -> int:
    """Ingère les sources sélectionnées dans `danann` (incrémental, par doc)."""
    total = 0
    files = 0
    by_lang: dict[str, int] = {}

    for name in sources:
        for origin, text, source, language in iter_source(
            name, bundle_dir=bundle_dir, categories=categories,
            pydoc_modules=pydoc_modules, man_pages=man_pages,
            mdn_dir=mdn_dir, mdn_areas=mdn_areas,
            docker_dir=docker_dir, docker_areas=docker_areas,
            pg_dir=pg_dir, pg_prefixes=pg_prefixes,
        ):
            if max_files is not None and files >= max_files:
                logger.info("Limite --max-files (%d) atteinte", max_files)
                return total
            chunks = chunk_code_doc(
                text, max_chars=max_chars, markdown=(source in MARKDOWN_SOURCES)
            )
            if not chunks:
                continue
            texts = [c for c, _ in chunks]
            metas = [_meta(origin, sec, source, language) for _, sec in chunks]
            inserted = danann.index(texts, metas)
            total += inserted
            by_lang[language] = by_lang.get(language, 0) + inserted
            files += 1
            if files % 20 == 0:
                logger.info("  %d docs, %d chunks…", files, total)

    logger.info("Ingéré : %d documents, %d chunks  par langage: %s", files, total, by_lang)
    return total


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--compression", default="int8", choices=["none", "int8", "binary"])
    parser.add_argument(
        "--sources", default=",".join(ALL_SOURCES),
        help=f"Sources à ingérer parmi {ALL_SOURCES} (ex: python,man).",
    )
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="Catégories du bundle Python (ex: tutorial,library,howto,faq).",
    )
    parser.add_argument(
        "--pydoc-modules", default=",".join(DEFAULT_PYDOC_MODULES),
        help="Modules stdlib pour pydoc, ou 'none'.",
    )
    parser.add_argument(
        "--man-pages", default=",".join(DEFAULT_MAN_PAGES),
        help="Pages man à ingérer, ou 'none'.",
    )
    parser.add_argument("--mdn-dir", default=str(DEFAULT_MDN_DIR))
    parser.add_argument(
        "--mdn-areas", default=",".join(DEFAULT_MDN_AREAS),
        help="Aires MDN (sous files/en-us/web/) à ingérer (ex: javascript,css,html).",
    )
    parser.add_argument("--docker-dir", default=str(DEFAULT_DOCKER_DIR))
    parser.add_argument(
        "--docker-areas", default=",".join(DEFAULT_DOCKER_AREAS),
        help="Aires docker/docs (sous content/) à ingérer (ex: get-started,manuals,reference).",
    )
    parser.add_argument("--pg-dir", default=str(DEFAULT_PG_DIR))
    parser.add_argument(
        "--pg-prefixes", default=",".join(DEFAULT_PG_PREFIXES),
        help="Préfixes de pages PostgreSQL à ingérer, ou 'all' pour tout.",
    )
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--max-files", type=int, default=None, help="Cap de docs (test rapide).")
    parser.add_argument("--min-chunks", type=int, default=1)
    parser.add_argument("--no-fetch", action="store_true", help="Ne pas (re)télécharger le bundle.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from modules.danann.store import Danann  # noqa: PLC0415

    bundle_dir = Path(args.bundle_dir)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    pydoc_modules = (
        () if args.pydoc_modules.strip().lower() == "none"
        else tuple(m.strip() for m in args.pydoc_modules.split(",") if m.strip())
    )
    man_pages = (
        () if args.man_pages.strip().lower() == "none"
        else tuple(m.strip() for m in args.man_pages.split(",") if m.strip())
    )
    mdn_dir = Path(args.mdn_dir)
    mdn_areas = tuple(a.strip() for a in args.mdn_areas.split(",") if a.strip())
    docker_dir = Path(args.docker_dir)
    docker_areas = tuple(a.strip() for a in args.docker_areas.split(",") if a.strip())
    pg_dir = Path(args.pg_dir)
    pg_prefixes = (
        () if args.pg_prefixes.strip().lower() == "all"
        else tuple(p.strip() for p in args.pg_prefixes.split(",") if p.strip())
    )

    # Les téléchargements ne se font que si la source concernée le requiert.
    if "python" in sources and not args.no_fetch and categories:
        fetch_python_text_bundle(bundle_dir)
    if "mdn" in sources and not args.no_fetch and mdn_areas:
        fetch_mdn_content(mdn_dir, mdn_areas)
    if "docker" in sources and not args.no_fetch and docker_areas:
        fetch_docker_docs(docker_dir, docker_areas)
    if "postgresql" in sources and not args.no_fetch:
        fetch_postgres_html(pg_dir)

    danann = Danann(compression=args.compression, use_reranker=False)
    total = build_index(
        danann, sources=sources, bundle_dir=bundle_dir, categories=categories,
        pydoc_modules=pydoc_modules, man_pages=man_pages,
        mdn_dir=mdn_dir, mdn_areas=mdn_areas,
        docker_dir=docker_dir, docker_areas=docker_areas,
        pg_dir=pg_dir, pg_prefixes=pg_prefixes,
        max_files=args.max_files, max_chars=args.max_chars,
    )

    if total < args.min_chunks:
        sys.exit(f"ÉCHEC : {total} chunks < {args.min_chunks}.")

    out = Path(args.output)
    danann.save_index(out)

    mem = danann.memory_bytes()
    float_equiv = danann.count() * 384 * 4
    ratio = float_equiv / mem if mem else 0.0
    print("─" * 60)
    print(f"Sources     : {sources}")
    print(f"Chunks      : {danann.count()}")
    print(f"Compression : {args.compression}")
    print(f"Index RAM   : {mem/1024:.1f} KB (float32 équiv ~{float_equiv/1024:.1f} KB → ×{ratio:.1f})")
    print(f"Sauvé       : {out}/  (corpus.json + vectors.npz)")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
