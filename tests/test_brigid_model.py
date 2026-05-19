"""Tests unitaires du classifieur Brigid (CfC) + save/load.

Tous les tests dépendent de torch + ncps. On `importorskip` proprement
pour qu'ils tournent où les deps sont là (CI + venv local complet) et
soient skippés ailleurs sans casser la suite.

Couvre :
  - construction du modèle (params, architecture, déterminisme via seed)
  - forward pass (shapes, types)
  - save/load roundtrip (state + métadonnées)
  - garde-fous métadonnées (incompat embedder, labels désordonnés)
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, ".")

# Skip tout le module si torch ou ncps manquent.
torch = pytest.importorskip("torch")
pytest.importorskip("ncps")

from modules.brigid.dataset import LABELS  # noqa: E402
from modules.brigid.embedder import EMBED_DIM, EMBED_MODEL_NAME  # noqa: E402
from modules.brigid.model import (  # noqa: E402
    DEFAULT_HIDDEN_DIM,
    CheckpointMetadata,
    build_classifier,
    load_checkpoint,
    save_checkpoint,
)


# ─── Construction du modèle ────────────────────────────────────────


def test_build_classifier_defaults():
    model = build_classifier()
    assert model.input_dim == EMBED_DIM
    assert model.hidden_dim == DEFAULT_HIDDEN_DIM
    assert model.num_classes == len(LABELS)


def test_build_classifier_custom_hidden():
    model = build_classifier(hidden_dim=32)
    assert model.hidden_dim == 32


def test_build_classifier_seed_is_deterministic():
    m1 = build_classifier(seed=42)
    m2 = build_classifier(seed=42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2), "même seed → mêmes poids"


def test_build_classifier_different_seeds_differ():
    m1 = build_classifier(seed=1)
    m2 = build_classifier(seed=2)
    # Au moins UN paramètre doit différer.
    diff = any(
        not torch.allclose(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    assert diff, "seeds différentes → poids différents (au moins partiellement)"


# ─── Forward pass ──────────────────────────────────────────────────


def test_forward_batch():
    model = build_classifier(seed=42)
    x = torch.randn(4, EMBED_DIM)
    out = model(x)
    assert out.shape == (4, len(LABELS))
    assert out.dtype == torch.float32


def test_forward_single_example():
    model = build_classifier(seed=42)
    x = torch.randn(1, EMBED_DIM)
    out = model(x)
    assert out.shape == (1, len(LABELS))


def test_forward_deterministic_in_eval():
    """En mode eval, deux forward passes sur le même input donnent le même résultat."""
    model = build_classifier(seed=42)
    model.eval()
    x = torch.randn(2, EMBED_DIM)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


# ─── Save / load ───────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    model = build_classifier(seed=42)
    ckpt = tmp_path / "test.pt"
    save_checkpoint(model, ckpt)
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0

    reloaded = load_checkpoint(ckpt)
    # Mêmes forward que l'original.
    x = torch.randn(3, EMBED_DIM)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        a = model(x)
        b = reloaded(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_save_writes_metadata(tmp_path: Path):
    model = build_classifier(seed=42)
    ckpt = tmp_path / "test.pt"
    save_checkpoint(model, ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "state_dict" in payload
    assert "metadata" in payload
    meta = payload["metadata"]
    assert meta["input_dim"] == EMBED_DIM
    assert meta["embed_model_name"] == EMBED_MODEL_NAME
    assert tuple(meta["labels"]) == LABELS


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_rejects_incompatible_embedder(tmp_path: Path):
    """Si l'embedder a changé, le checkpoint est sémantiquement obsolète."""
    model = build_classifier(seed=42)
    ckpt = tmp_path / "test.pt"
    # On sauve avec des métadonnées modifiées en douce pour simuler un
    # checkpoint produit avec un autre modèle d'embedding.
    bad_meta = CheckpointMetadata(
        input_dim=EMBED_DIM,
        hidden_dim=model.hidden_dim,
        num_classes=model.num_classes,
        labels=LABELS,
        embed_model_name="sentence-transformers/some-other-model",
    )
    save_checkpoint(model, ckpt, bad_meta)
    with pytest.raises(ValueError, match="Embedder"):
        load_checkpoint(ckpt)


def test_load_rejects_reordered_labels(tmp_path: Path):
    """Si l'ordre des LABELS a changé, le checkpoint n'aligne plus."""
    model = build_classifier(seed=42)
    ckpt = tmp_path / "test.pt"
    reordered = tuple(reversed(LABELS))
    bad_meta = CheckpointMetadata(
        input_dim=EMBED_DIM,
        hidden_dim=model.hidden_dim,
        num_classes=model.num_classes,
        labels=reordered,
        embed_model_name=EMBED_MODEL_NAME,
    )
    save_checkpoint(model, ckpt, bad_meta)
    with pytest.raises(ValueError, match="Labels"):
        load_checkpoint(ckpt)


def test_checkpoint_metadata_asdict_is_serializable():
    """Métadonnées convertibles en dict pour torch.save (qui pickle)."""
    meta = CheckpointMetadata(
        input_dim=384,
        hidden_dim=16,
        num_classes=6,
        labels=LABELS,
        embed_model_name=EMBED_MODEL_NAME,
    )
    d = asdict(meta)
    assert d["input_dim"] == 384
    assert d["labels"] == LABELS
