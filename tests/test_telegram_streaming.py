"""Tests du streaming Telegram (helper stream_collect + _chunk_message).

On teste la logique de throttle/édition sans Telegram réel : on injecte
une horloge factice et un `edit` qui enregistre les appels.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

sys.path.insert(0, ".")

# python-telegram-bot requis pour importer le module (importe `telegram`).
pytest.importorskip("telegram")

from interfaces.telegram_bot import _chunk_message, stream_collect


async def _agen(pieces):
    for p in pieces:
        yield p


class FakeClock:
    """Horloge contrôlée : avance de `step` à chaque lecture."""

    def __init__(self, step: float = 0.5) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        v = self.t
        self.t += self.step
        return v


def _run(coro):
    return asyncio.run(coro)


# ─── stream_collect ────────────────────────────────────────────────


def test_stream_collect_returns_full_text():
    edits = []

    async def edit(text, final):
        edits.append((text, final))

    final = _run(stream_collect(_agen(["a", "b", "c"]), edit, interval=0.0, clock=FakeClock()))
    assert final == "abc"


def test_stream_collect_always_does_final_edit():
    edits = []

    async def edit(text, final):
        edits.append((text, final))

    _run(stream_collect(_agen(["x", "y"]), edit, interval=999.0, clock=FakeClock()))
    # Avec un interval géant, aucune édition intermédiaire, mais l'édition
    # finale doit avoir lieu.
    finals = [e for e in edits if e[1] is True]
    assert len(finals) == 1
    assert finals[0][0] == "xy"


def test_stream_collect_throttles_intermediate_edits():
    edits = []

    async def edit(text, final):
        edits.append((text, final))

    # interval=1.0, horloge +0.5/lecture → édition ~1 fois sur 2.
    pieces = [str(i) for i in range(6)]
    _run(stream_collect(_agen(pieces), edit, interval=1.0, clock=FakeClock(step=0.5)))
    intermediate = [e for e in edits if e[1] is False]
    # Bien moins d'éditions que de morceaux (throttle actif).
    assert len(intermediate) < len(pieces)
    assert any(e[1] for e in edits)  # édition finale présente


def test_stream_collect_swallows_edit_errors():
    """Une édition qui throw ne casse pas le flux ni le résultat final."""
    calls = {"n": 0}

    async def flaky_edit(text, final):
        calls["n"] += 1
        raise RuntimeError("message not modified")

    final = _run(stream_collect(_agen(["a", "b"]), flaky_edit, interval=0.0, clock=FakeClock()))
    assert final == "ab"  # malgré les erreurs d'édition
    assert calls["n"] >= 1


def test_stream_collect_ignores_blank_only_before_content():
    """Pas d'édition tant qu'il n'y a que du blanc."""
    edits = []

    async def edit(text, final):
        edits.append((text, final))

    _run(stream_collect(_agen([" ", " ", "vrai"]), edit, interval=0.0, clock=FakeClock()))
    # Édition finale OK ; les éditions intermédiaires sur du blanc pur évitées.
    assert edits[-1] == ("  vrai", True)


# ─── _chunk_message (déjà présent, on couvre le découpage) ────────


def test_chunk_message_short_stays_single():
    assert _chunk_message("court") == ["court"]


def test_chunk_message_splits_long():
    text = "a" * 9000
    parts = _chunk_message(text, limit=4000)
    assert len(parts) >= 3
    assert all(len(p) <= 4000 for p in parts)


def test_chunk_message_prefers_newline_cut():
    text = "x" * 3000 + "\n" + "y" * 2000
    parts = _chunk_message(text, limit=4000)
    # La coupe doit tomber sur le retour à la ligne (1er bloc = les x).
    assert parts[0] == "x" * 3000
