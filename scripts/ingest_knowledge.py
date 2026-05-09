"""
MORRIGAN — Script d'ingestion de corpus (Phase 2).

Lit des fichiers texte/markdown depuis data/knowledge/,
les decoupe en chunks avec metadonnees riches, et les indexe dans Danann.

Metadonnees extraites par chunk :
  - source    : nom du fichier d'origine
  - domain    : domaine detecte (reseau, ia, mythologie, projet, code, general)
  - type      : type de chunk (definition, comparison, explanation, fact, example)
  - section   : titre de section markdown le plus proche (## header)
  - confidence: score de confiance de la source (0.0-1.0)
  - version   : version du schema de metadonnees

Usage :
    python scripts/ingest_knowledge.py
    python scripts/ingest_knowledge.py --source data/knowledge/ --backend memory
    python scripts/ingest_knowledge.py --backend supabase
"""

import argparse
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from modules.danann.store import Danann


# Configuration du chunking
CHUNK_SIZE = 400       # Caracteres par chunk
CHUNK_OVERLAP = 50     # Overlap entre chunks

# Version du schema de metadonnees (pour migration future)
METADATA_VERSION = "2.0"

# ─── Detection de domaine ───────────────────────────────────────────

# Mapping filename patterns -> domain
_DOMAIN_FROM_FILENAME: Dict[str, str] = {
    "reseau": "reseau",
    "protocole": "reseau",
    "network": "reseau",
    "tcp": "reseau",
    "dns": "reseau",
    "ia": "ia",
    "intelligence": "ia",
    "neural": "ia",
    "transformer": "ia",
    "llm": "ia",
    "mamba": "ia",
    "rwkv": "ia",
    "mythologi": "mythologie",
    "celt": "mythologie",
    "druide": "mythologie",
    "morrigan": "projet",
    "projet": "projet",
    "code": "code",
    "python": "code",
    "javascript": "code",
    "bash": "code",
    "sql": "code",
    "html": "code",
    "css": "code",
}

# Keyword patterns -> domain (fallback si filename pas assez clair)
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "reseau": ["tcp", "udp", "http", "dns", "ip", "protocole", "port", "paquet", "routage", "firewall", "tls", "ssh"],
    "ia": ["neurone", "transformer", "embedding", "inference", "entrainement", "modele", "parametre", "gpu", "token", "attention", "llm", "cfc", "lnn"],
    "mythologie": ["dieu", "deesse", "celtique", "druide", "tuatha", "dagda", "brigid", "ogham", "cuchulainn", "sidhe", "fal"],
    "projet": ["morrigan", "module", "orchestrateur", "pipeline", "scathach", "danann", "cauldron", "brigid"],
    "code": ["fonction", "variable", "class", "import", "return", "def ", "const ", "let ", "select ", "from "],
}


def _normalize_text(text: str) -> str:
    """Lowercase + strip accents."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detect_domain(filename: str, text: str) -> str:
    """Detecte le domaine d'un chunk a partir du filename et du contenu."""
    fn = _normalize_text(filename)

    # 1. Match sur le filename
    for pattern, domain in _DOMAIN_FROM_FILENAME.items():
        if pattern in fn:
            return domain

    # 2. Fallback : compter les keywords dans le texte
    text_norm = _normalize_text(text)
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_norm)

    if scores:
        best = max(scores, key=scores.get)  # type: ignore
        if scores[best] >= 2:
            return best

    return "general"


# ─── Detection de type de chunk ─────────────────────────────────────

# Patterns indicateurs de type
_COMPARISON_PATTERNS = [
    r"\bvs\b", r"\bcontre\b", r"\bcompar", r"\bdifference\b",
    r"\btandis que\b", r"\balors que\b", r"\bcontrairement\b",
    r"\bplus .+ que\b", r"\bmoins .+ que\b",
]
_DEFINITION_PATTERNS = [
    r"\best un[e]?\b", r"\best le\b", r"\best la\b",
    r"\bdesigne\b", r"\bsignifie\b", r"\bconsiste a\b",
    r"\bse definit\b",
]
_EXAMPLE_PATTERNS = [
    r"\bpar exemple\b", r"\bnotamment\b", r"\bcomme\b.+\bet\b",
    r"\btypiquement\b", r"\ben pratique\b",
]


def detect_chunk_type(text: str) -> str:
    """Detecte le type semantique d'un chunk."""
    text_norm = _normalize_text(text)

    # Comparaison
    if sum(1 for p in _COMPARISON_PATTERNS if re.search(p, text_norm)) >= 2:
        return "comparison"

    # Definition
    if any(re.search(p, text_norm) for p in _DEFINITION_PATTERNS):
        return "definition"

    # Exemple
    if any(re.search(p, text_norm) for p in _EXAMPLE_PATTERNS):
        return "example"

    # Explication (chunks longs avec structure)
    if len(text) > 200:
        return "explanation"

    return "fact"


# ─── Confiance source ───────────────────────────────────────────────

# Heuristique simple : les fichiers bien structures (markdown avec headers)
# et de taille raisonnable obtiennent un score plus eleve.
def estimate_confidence(content: str, filename: str) -> float:
    """Estime la confiance d'une source (0.0-1.0)."""
    score = 0.5  # base

    # Bonus : fichier markdown structure
    header_count = len(re.findall(r"(?m)^#+\s", content))
    if header_count >= 3:
        score += 0.2
    elif header_count >= 1:
        score += 0.1

    # Bonus : taille raisonnable (ni trop court, ni trop long)
    if 500 < len(content) < 10000:
        score += 0.1

    # Bonus : pas de code executif (signe de documentation curee)
    if "```" not in content:
        score += 0.05

    # Plafond
    return min(1.0, score)


# ─── Section markdown ───────────────────────────────────────────────

def extract_sections(content: str) -> List[Tuple[str, str]]:
    """
    Decoupe un fichier markdown en sections (header, body).

    Retourne une liste de (section_title, section_body).
    Si pas de headers, retourne une seule section avec titre vide.
    """
    lines = content.split("\n")
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_body: List[str] = []

    for line in lines:
        header_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if header_match:
            # Sauver la section precedente
            if current_body or current_title:
                body = "\n".join(current_body).strip()
                if body:
                    sections.append((current_title, body))
            current_title = header_match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    # Derniere section
    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_title, body))

    return sections if sections else [("", content)]


# ─── Chunking ───────────────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def split_into_sentences(text: str) -> List[str]:
    """Decoupe en phrases (approximation simple)."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Decoupe un texte en chunks, en respectant les limites de phrases.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size].strip())
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]


# ─── Ingestion ──────────────────────────────────────────────────────

def load_file(path: Path) -> Tuple[str, dict]:
    """Charge un fichier et retourne son contenu + metadata de base."""
    content = path.read_text(encoding="utf-8")
    metadata = {
        "source": path.name,
        "path": str(path),
        "size": len(content),
    }
    return content, metadata


def ingest_directory(
    danann: Danann,
    source_dir: Path,
    extensions: Tuple[str, ...] = (".txt", ".md"),
) -> int:
    """
    Ingere tous les fichiers d'un repertoire dans Danann.

    Phase 2 : metadonnees riches par chunk (domain, type, section,
    confidence, version).
    """
    total_chunks = 0
    files_processed = 0

    for ext in extensions:
        for file_path in source_dir.rglob(f"*{ext}"):
            print(f"  Lecture: {file_path.name}")
            content, base_metadata = load_file(file_path)

            # Confiance de la source
            source_confidence = estimate_confidence(content, file_path.name)

            # Decouper par sections markdown puis en chunks
            sections = extract_sections(content)
            all_chunks: List[str] = []
            all_metadatas: List[Dict[str, Any]] = []

            chunk_idx = 0
            for section_title, section_body in sections:
                chunks = chunk_text(section_body)
                for chunk in chunks:
                    domain = detect_domain(file_path.name, chunk)
                    chunk_type = detect_chunk_type(chunk)

                    meta = {
                        **base_metadata,
                        "chunk_index": chunk_idx,
                        "section": section_title,
                        "domain": domain,
                        "type": chunk_type,
                        "confidence": source_confidence,
                        "version": METADATA_VERSION,
                    }
                    all_chunks.append(chunk)
                    all_metadatas.append(meta)
                    chunk_idx += 1

            if not all_chunks:
                continue

            inserted = danann.index(all_chunks, all_metadatas)
            total_chunks += inserted
            files_processed += 1

            # Stats par domaine/type
            domains = {}
            types = {}
            for m in all_metadatas:
                d = m.get("domain", "?")
                t = m.get("type", "?")
                domains[d] = domains.get(d, 0) + 1
                types[t] = types.get(t, 0) + 1

            print(f"    -> {len(all_chunks)} chunks indexes")
            print(f"       domaines: {domains}")
            print(f"       types:    {types}")

    print()
    print(f"Total: {files_processed} fichier(s), {total_chunks} chunks")
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingere un corpus dans Danann")
    parser.add_argument(
        "--source",
        default="data/knowledge",
        help="Repertoire source contenant les fichiers .txt/.md",
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "supabase"],
        default="memory",
        help="Backend Danann (memory ou supabase)",
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("  MORRIGAN — Ingestion de corpus dans Danann")
    print("=" * 60)
    print()
    print(f"Source  : {args.source}")
    print(f"Backend : {args.backend}")
    print()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"ERREUR: le repertoire {source_dir} n'existe pas.")
        sys.exit(1)

    # Initialiser Danann avec le backend choisi
    danann = Danann(
        backend=args.backend,
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_KEY", ""),
    )

    # Ingestion
    total = ingest_directory(danann, source_dir)

    # Test rapide
    if total > 0:
        print()
        print("--- Test rapide ---")
        test_queries = ["Morrigan", "reseau de neurones", "mythologie celtique"]
        for q in test_queries:
            results = danann.search(q, top_k=2)
            if results:
                top = results[0]
                print(f"  '{q}' -> [{top[1]:.2f}] {top[0][:80]}...")

    print()
    print("Ingestion terminee.")


if __name__ == "__main__":
    main()
