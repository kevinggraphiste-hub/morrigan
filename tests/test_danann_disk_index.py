"""Tests de la persistance disque de l'index Danann (Phase 4 PR 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

pytest.importorskip("sentence_transformers")

from modules.danann.store import Danann

CORPUS = [
    "TCP est un protocole de transport fiable qui garantit la livraison.",
    "La déesse Brigid est associée à la forge et à la poésie.",
    "Un réseau neuronal liquide utilise des dynamiques à temps continu.",
]
META = [{"domain": "reseau"}, {"domain": "myth"}, {"domain": "ia"}]


def _build(compression: str) -> Danann:
    d = Danann(compression=compression, use_reranker=False)
    d.index(CORPUS, META)
    return d


@pytest.mark.parametrize("compression", ["none", "int8", "binary"])
def test_save_load_roundtrip_preserves_search(compression, tmp_path: Path):
    src = _build(compression)
    src.save_index(tmp_path / "idx")
    assert (tmp_path / "idx" / "corpus.json").exists()
    assert (tmp_path / "idx" / "vectors.npz").exists()

    loaded = Danann.load_index(tmp_path / "idx", use_reranker=False)
    assert loaded.compression == compression
    assert loaded.count() == len(CORPUS)

    # Le top-1 sur une requête nette doit être identique avant/après.
    q = "protocole réseau fiable"
    assert src.search(q, top_k=1)[0][0] == loaded.search(q, top_k=1)[0][0]


def test_loaded_compressed_has_no_float32(tmp_path: Path):
    src = _build("int8")
    src.save_index(tmp_path / "idx")
    loaded = Danann.load_index(tmp_path / "idx", use_reranker=False)
    assert loaded.embeddings is None
    assert loaded._int8 is not None


def test_loaded_binary_has_both_indexes(tmp_path: Path):
    src = _build("binary")
    src.save_index(tmp_path / "idx")
    loaded = Danann.load_index(tmp_path / "idx", use_reranker=False)
    assert loaded._binary is not None
    assert loaded._int8 is not None


def test_metadata_preserved(tmp_path: Path):
    src = _build("int8")
    src.save_index(tmp_path / "idx")
    loaded = Danann.load_index(tmp_path / "idx", use_reranker=False)
    res = loaded.search("réseau neuronal", top_k=1, domain="ia")
    assert res
    assert res[0][2].get("domain") == "ia"


# ─── Script build_compressed_index ─────────────────────────────────


def test_build_script_end_to_end(tmp_path: Path):
    from scripts.build_compressed_index import main

    src = tmp_path / "corpus"
    src.mkdir()
    (src / "reseau.md").write_text(
        "# Réseaux\nTCP est fiable. UDP est rapide. IP route les paquets.\n",
        encoding="utf-8",
    )
    out = tmp_path / "idx"
    rc = main([
        "--source", str(src), "--output", str(out),
        "--compression", "int8", "--min-chunks", "1",
    ])
    assert rc == 0
    assert (out / "corpus.json").exists()
    assert (out / "vectors.npz").exists()

    loaded = Danann.load_index(out, use_reranker=False)
    assert loaded.compression == "int8"
    assert loaded.count() >= 1
    assert loaded.embeddings is None  # compressé, pas de float32


def test_build_script_missing_source(tmp_path: Path):
    from scripts.build_compressed_index import main

    with pytest.raises(SystemExit):
        main(["--source", str(tmp_path / "absent"), "--output", str(tmp_path / "x")])
