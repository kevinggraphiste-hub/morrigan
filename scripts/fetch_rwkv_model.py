"""
Télécharge le modèle RWKV GGUF utilisé par le backend Scáthach (Phase 3).

Le fichier GGUF (~1 Go) est gitignoré — option B : artefact téléchargé,
pas commité. Ce script le récupère depuis HuggingFace dans
`data/models/`.

Usage :
    .venv-uv/bin/python scripts/fetch_rwkv_model.py
    .venv-uv/bin/python scripts/fetch_rwkv_model.py --quant Q2_K   # + léger, qualité moindre

Quantizations disponibles (RWKV-6 World 1.6B) :
    Q2_K   ~654 Mo  — trop agressif, sortie dégénérée (à éviter)
    Q3_K   ~801 Mo  — acceptable si RAM serrée
    Q4_K   ~993 Mo  — DÉFAUT, plancher de qualité validé
    Q5_K  ~1174 Mo  — meilleure qualité si RAM le permet
    Q8_0  ~1750 Mo  — quasi-fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.scathach.rwkv_backend import DEFAULT_REPO

DEST = Path(__file__).resolve().parent.parent / "data" / "models"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--quant", default="Q4_K",
        help="Quantization (Q2_K, Q3_K, Q4_K, Q5_K, Q8_0). Défaut Q4_K.",
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub manquant (cf. requirements.txt).")

    filename = f"rwkv-6-world-1.6b-{args.quant}.gguf"
    print(f"Téléchargement {args.repo}/{filename} → {DEST}/ …")
    DEST.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=args.repo, filename=filename, local_dir=str(DEST))
    size_mb = Path(path).stat().st_size / 1e6
    print(f"OK : {path} ({size_mb:.0f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
