"""Tests de la surface OpenAI-compatible (interfaces/openai_compat.py).

Même approche que test_api.py : AnDagda avec un `scathach` factice injecté via
`create_app(dagda=...)`, requêtes via httpx + ASGITransport (zéro réseau),
convention `asyncio.run` (pas de pytest-asyncio).

Couvre le contrat qu'un client OpenAI standard (dont Gungnir) attend :
`/v1/models`, `/v1/chat/completions` (non-stream + stream), auth Bearer,
extraction du dernier message user.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import AsyncIterator

sys.path.insert(0, ".")

from httpx import ASGITransport, AsyncClient

from core.dagda import AnDagda
from core.types import ModuleInput, ModuleOutput, MorriganModule
from interfaces.api import create_app
from interfaces.openai_compat import _last_user_query, _sanitize_session
from pydantic import BaseModel


class _Msg(BaseModel):
    role: str
    content: str = ""


class _FakeScathach(MorriganModule):
    """Réponse fixe, expose `process` ET `stream`."""

    def __init__(self, response: str = "Bonjour depuis Morrigan."):
        self.response = response
        self.last_generated_by = "fake"

    async def process(self, input: ModuleInput) -> ModuleOutput:
        return ModuleOutput(
            result=self.response, confidence=0.99, metadata={"generated_by": "fake"}
        )

    async def stream(self, input: ModuleInput) -> AsyncIterator[str]:
        for word in self.response.split():
            yield word + " "

    async def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> dict:
        return {"backend": "fake"}


async def _build_test_dagda(response: str = "Bonjour depuis Morrigan.") -> AnDagda:
    dagda = AnDagda(config_path="config/_test_does_not_exist.yaml")
    dagda.register_module("scathach", _FakeScathach(response))
    await dagda.initialize()
    return dagda


def _client(dagda: AnDagda, **kw) -> AsyncClient:
    app = create_app(dagda=dagda, **kw)
    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://morrigan.test"
    )


def _parse_openai_sse(payload: str) -> list:
    """Extrait les objets JSON des lignes `data:` (hors `[DONE]`)."""
    chunks = []
    for block in payload.strip().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                data = line[len("data:") :].strip()
                if data and data != "[DONE]":
                    chunks.append(json.loads(data))
    return chunks


# ─── Unitaires (helpers) ──────────────────────────────────────────────


def test_sanitize_session_defaults_and_charset():
    assert _sanitize_session(None) == "default"
    assert _sanitize_session("") == "default"
    assert _sanitize_session("conv-123_ok.v2") == "conv-123_ok.v2"
    # Caractères hors charset retirés.
    assert _sanitize_session("a b!@#c") == "abc"
    # Que des caractères invalides → repli "default".
    assert _sanitize_session("!@# ") == "default"


def test_last_user_query_takes_last_user_message():
    msgs = [
        _Msg(role="system", content="tu es un assistant"),
        _Msg(role="user", content="première question"),
        _Msg(role="assistant", content="une réponse"),
        _Msg(role="user", content="  la vraie question  "),
    ]
    assert _last_user_query(msgs) == "la vraie question"


def test_last_user_query_no_user_raises():
    import pytest

    msgs = [_Msg(role="system", content="x"), _Msg(role="assistant", content="y")]
    with pytest.raises(Exception):
        _last_user_query(msgs)


# ─── /v1/models ───────────────────────────────────────────────────────


def test_list_models():
    async def _run() -> dict:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.get("/v1/models")
        return {"status": r.status_code, "body": r.json()}

    out = asyncio.run(_run())
    assert out["status"] == 200
    assert out["body"]["object"] == "list"
    ids = [m["id"] for m in out["body"]["data"]]
    assert "morrigan" in ids


# ─── /v1/chat/completions (non-stream) ────────────────────────────────


def test_chat_completion_non_stream_shape():
    async def _run() -> dict:
        dagda = await _build_test_dagda(response="Salut le monde.")
        async with _client(dagda) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "morrigan",
                    "messages": [{"role": "user", "content": "Bonjour ?"}],
                },
            )
        return {"status": r.status_code, "body": r.json()}

    out = asyncio.run(_run())
    assert out["status"] == 200
    body = out["body"]
    assert body["object"] == "chat.completion"
    assert body["model"] == "morrigan"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Salut le monde."
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] >= 1


def test_chat_completion_uses_last_user_message():
    """Le routage (conversation vs autre) dépend du dernier message user."""

    async def _run() -> dict:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "ignore-moi"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "Salut"},
                    ]
                },
            )
        return r.json()

    body = asyncio.run(_run())
    # "Salut" → heuristique conversation (pas de Brigid registered).
    assert body["choices"][0]["message"]["content"]


def test_chat_completion_rejects_no_user_message():
    async def _run() -> int:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "system", "content": "x"}]},
            )
        return r.status_code

    assert asyncio.run(_run()) == 400


def test_chat_completion_rejects_empty_messages():
    async def _run() -> int:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post("/v1/chat/completions", json={"messages": []})
        return r.status_code

    assert asyncio.run(_run()) == 422


# ─── /v1/chat/completions (stream SSE OpenAI) ─────────────────────────


def test_chat_completion_stream():
    async def _run() -> str:
        dagda = await _build_test_dagda(response="un deux trois")
        async with _client(dagda) as c:
            async with c.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Salut"}],
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                payload = ""
                async for chunk in r.aiter_text():
                    payload += chunk
        return payload

    payload = asyncio.run(_run())
    assert payload.rstrip().endswith("data: [DONE]")
    chunks = _parse_openai_sse(payload)
    assert chunks
    # Tous des chunks de complétion.
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # Le 1er annonce le rôle.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Contenu reconstitué.
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert content.strip() == "un deux trois"
    # Le dernier chunk clôture.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# ─── Auth Bearer (et X-API-Key) ───────────────────────────────────────


def test_chat_completion_auth_bearer():
    async def _run() -> tuple:
        dagda = await _build_test_dagda()
        async with _client(dagda, api_key="secret") as c:
            body = {"messages": [{"role": "user", "content": "x"}]}
            r_no = await c.post("/v1/chat/completions", json=body)
            r_bad = await c.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer wrong"},
            )
            r_bearer = await c.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer secret"},
            )
            r_xkey = await c.post(
                "/v1/chat/completions",
                json=body,
                headers={"X-API-Key": "secret"},
            )
        return (
            r_no.status_code,
            r_bad.status_code,
            r_bearer.status_code,
            r_xkey.status_code,
        )

    no, bad, bearer, xkey = asyncio.run(_run())
    assert no == 401
    assert bad == 401
    assert bearer == 200
    assert xkey == 200
