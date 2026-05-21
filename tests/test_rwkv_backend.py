"""Tests du backend RWKV de Scáthach (PR A Phase 3).

Deux niveaux :
  - Logique du wrapper (format de prompt, config, dégradation) —
    testable SANS modèle ni llama_cpp.
  - Smoke de génération réelle — gated par `importorskip llama_cpp`
    + skip si le GGUF n'est pas présent (gitignoré ; présent en local
    après `fetch_rwkv_model.py`, absent sur CI tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from modules.scathach.rwkv_backend import (
    DEFAULT_FILENAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPEAT_PENALTY,
    RWKVBackend,
)


# ─── format_prompt (pur, pas de modèle) ────────────────────────────


def test_format_prompt_without_context():
    p = RWKVBackend.format_prompt("Qu'est-ce que TCP ?")
    assert p == "User: Qu'est-ce que TCP ?\n\nAssistant:"


def test_format_prompt_with_context():
    p = RWKVBackend.format_prompt(
        "Compare TCP et UDP", context=["TCP est fiable", "UDP est rapide"]
    )
    # Format RWKV World respecté + contexte injecté avant la question.
    assert p.startswith("User:")
    assert p.endswith("Assistant:")
    assert "TCP est fiable" in p
    assert "UDP est rapide" in p
    assert "Compare TCP et UDP" in p


def test_format_prompt_ignores_empty_context_lines():
    p = RWKVBackend.format_prompt("Q", context=["  ", "", "fait utile"])
    assert "fait utile" in p
    # Les lignes vides ne produisent pas de puces orphelines.
    assert "- \n" not in p


def test_format_prompt_empty_context_list_is_like_none():
    p = RWKVBackend.format_prompt("Q", context=[])
    assert p == "User: Q\n\nAssistant:"


# ─── Config / défauts ──────────────────────────────────────────────


def test_default_model_path_points_to_gguf():
    assert str(DEFAULT_MODEL_PATH).endswith(".gguf")
    assert DEFAULT_FILENAME in str(DEFAULT_MODEL_PATH)


def test_repeat_penalty_default_is_high():
    """Garde-fou : RWKV part en boucle sans repeat_penalty élevé."""
    assert DEFAULT_REPEAT_PENALTY >= 1.2


# ─── Dégradation gracieuse (modèle absent) ─────────────────────────


def test_unavailable_when_model_missing(tmp_path: Path):
    b = RWKVBackend(model_path=tmp_path / "absent.gguf")
    assert b.is_available() is False
    assert b.load_error is not None
    assert "absent" in b.load_error.lower() or "gguf" in b.load_error.lower()


def test_generate_raises_when_unavailable(tmp_path: Path):
    b = RWKVBackend(model_path=tmp_path / "absent.gguf")
    with pytest.raises(RuntimeError, match="indisponible"):
        b.generate("User: salut\n\nAssistant:")


def test_answer_raises_when_unavailable(tmp_path: Path):
    b = RWKVBackend(model_path=tmp_path / "absent.gguf")
    with pytest.raises(RuntimeError):
        b.answer("salut")


# ─── Smoke de génération réelle (gated) ────────────────────────────

pytestmark_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason=(
        "GGUF RWKV absent (gitignoré). Lance scripts/fetch_rwkv_model.py "
        "pour activer ce test en local."
    ),
)


@pytestmark_model
def test_real_generation_is_coherent():
    pytest.importorskip("llama_cpp")
    b = RWKVBackend()
    assert b.is_available() is True
    text = b.answer(
        "Réponds juste 'bonjour'.", max_tokens=20, seed=42
    )
    # On ne vérifie pas le contenu exact (modèle stochastique quantizé),
    # juste qu'on récupère du texte non vide et non dégénéré.
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    # Garde-fou anti-boucle : pas plus de 5 répétitions du même token court.
    words = text.split()
    if len(words) >= 6:
        assert len(set(words)) > 1, f"Sortie dégénérée : {text!r}"


@pytestmark_model
def test_real_generation_with_rag_context():
    pytest.importorskip("llama_cpp")
    b = RWKVBackend()
    text = b.answer(
        "Quelle est la capitale ?",
        context=["La capitale de la France est Paris."],
        max_tokens=30,
        seed=42,
    )
    assert isinstance(text, str)
    assert len(text.strip()) > 0
