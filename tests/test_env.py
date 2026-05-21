"""Tests du chargement .env (core.env.load_env)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from core.env import load_env


def test_load_env_missing_file_returns_false(tmp_path: Path):
    assert load_env(tmp_path / "absent.env") is False


def test_load_env_loads_variables(tmp_path: Path, monkeypatch):
    pytest.importorskip("dotenv")
    env_file = tmp_path / ".env"
    env_file.write_text("MORRIGAN_TEST_VAR=hello123\n")
    monkeypatch.delenv("MORRIGAN_TEST_VAR", raising=False)

    assert load_env(env_file) is True
    assert os.environ["MORRIGAN_TEST_VAR"] == "hello123"


def test_load_env_does_not_override_real_env(tmp_path: Path, monkeypatch):
    """L'environnement réel l'emporte sur le fichier (override=False)."""
    pytest.importorskip("dotenv")
    env_file = tmp_path / ".env"
    env_file.write_text("MORRIGAN_TEST_VAR2=from_file\n")
    monkeypatch.setenv("MORRIGAN_TEST_VAR2", "from_real_env")

    load_env(env_file)
    assert os.environ["MORRIGAN_TEST_VAR2"] == "from_real_env"


def test_load_env_ignores_comments_and_blank(tmp_path: Path, monkeypatch):
    pytest.importorskip("dotenv")
    env_file = tmp_path / ".env"
    env_file.write_text("# commentaire\n\nMORRIGAN_TEST_VAR3=ok\n")
    monkeypatch.delenv("MORRIGAN_TEST_VAR3", raising=False)
    load_env(env_file)
    assert os.environ["MORRIGAN_TEST_VAR3"] == "ok"
