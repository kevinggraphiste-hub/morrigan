"""Tests de l'API HTTP (Phase 5 — interfaces/api.py).

On compose un AnDagda avec un module `scathach` factice (sans dep RWKV ni
embeddings) et on l'injecte via `create_app(dagda=...)`. Les tests
tapent /health, /stats, /query et /query/stream via `httpx.AsyncClient`
+ `ASGITransport` (zéro réseau, CI-safe).

On suit la convention du repo : `asyncio.run(_async())` plutôt que
pytest-asyncio (pas dans les deps).
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
from interfaces.api import MAX_QUERY_CHARS, create_app


class _FakeScathach(MorriganModule):
    """Module factice — réponse fixe, expose `process` ET `stream`."""

    def __init__(self, response: str = "Bonjour depuis Morrigan."):
        self.response = response
        self.last_generated_by = "fake"

    async def process(self, input: ModuleInput) -> ModuleOutput:
        return ModuleOutput(
            result=self.response,
            confidence=0.99,
            metadata={"generated_by": "fake"},
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


def _client(dagda: AnDagda) -> AsyncClient:
    app = create_app(dagda=dagda)
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://morrigan.test",
    )


def _parse_sse(payload: str) -> list[tuple[str, dict]]:
    """Parse un flux SSE basique en liste de (event, data_dict)."""
    events: list[tuple[str, dict]] = []
    for block in payload.strip().split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


# ─── /health ──────────────────────────────────────────────────────────


def test_health_lists_modules():
    async def _run() -> dict:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.get("/health")
        return {"status": r.status_code, "body": r.json()}

    out = asyncio.run(_run())
    assert out["status"] == 200
    assert out["body"]["status"] == "ok"
    assert "scathach" in out["body"]["modules"]


# ─── /stats (initial : 0 requêtes) ────────────────────────────────────


def test_stats_empty_initial():
    async def _run() -> dict:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.get("/stats")
        return r.json()

    body = asyncio.run(_run())
    assert "Morrigan" in body["text"]
    assert body["queries"] == 0
    assert body["by_type"] == {}


# ─── /query (réponse complète) ────────────────────────────────────────


def test_query_returns_response_and_routing():
    async def _run() -> dict:
        dagda = await _build_test_dagda(response="Salut le monde.")
        async with _client(dagda) as c:
            r = await c.post(
                "/query",
                json={"query": "Bonjour ?", "session_id": "sess-1"},
            )
        return {"status": r.status_code, "body": r.json()}

    out = asyncio.run(_run())
    assert out["status"] == 200
    body = out["body"]
    assert body["response"] == "Salut le monde."
    # Salutation détectée par heuristiques (pas de Brigid registered).
    assert body["query_type"] == "conversation"
    assert body["generated_by"] == "fake"
    assert body["latency_s"] >= 0.0
    assert "scathach" in body["modules"]


def test_query_rejects_empty():
    async def _run() -> int:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post("/query", json={"query": ""})
        return r.status_code

    assert asyncio.run(_run()) == 422


# ─── /query/stream (SSE) ──────────────────────────────────────────────


def test_query_stream_yields_pieces_and_done():
    async def _run() -> str:
        dagda = await _build_test_dagda(response="un deux trois")
        async with _client(dagda) as c:
            async with c.stream(
                "POST",
                "/query/stream",
                json={"query": "Salut", "session_id": "sess-stream"},
            ) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                payload = ""
                async for chunk in r.aiter_text():
                    payload += chunk
        return payload

    payload = asyncio.run(_run())
    events = _parse_sse(payload)
    assert len(events) >= 2

    pieces = [data["piece"] for ev, data in events if ev == "message"]
    assert "".join(pieces).strip() == "un deux trois"

    done_events = [data for ev, data in events if ev == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    assert done["query_type"] == "conversation"
    assert "scathach" in done["modules"]
    assert done["latency_s"] >= 0.0


def test_query_updates_stats():
    async def _run() -> dict:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            await c.post("/query", json={"query": "Salut"})
            r = await c.get("/stats")
        return r.json()

    body = asyncio.run(_run())
    assert body["queries"] == 1
    assert body["by_type"].get("conversation") == 1


# ─── Durcissement : bornes d'entrée ───────────────────────────────────


def test_query_rejects_too_long():
    async def _run() -> int:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post("/query", json={"query": "a" * (MAX_QUERY_CHARS + 1)})
        return r.status_code

    assert asyncio.run(_run()) == 422


def test_query_rejects_bad_session_id():
    async def _run() -> int:
        dagda = await _build_test_dagda()
        async with _client(dagda) as c:
            r = await c.post(
                "/query", json={"query": "ok", "session_id": "bad id!"}
            )
        return r.status_code

    assert asyncio.run(_run()) == 422


# ─── Durcissement : clé API optionnelle ───────────────────────────────


def test_api_key_enforced_when_set():
    async def _run() -> tuple[int, int, int, int]:
        dagda = await _build_test_dagda()
        app = create_app(dagda=dagda, api_key="secret")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://morrigan.test"
        ) as c:
            r_no = await c.post("/query", json={"query": "x"})
            r_bad = await c.post(
                "/query", json={"query": "x"}, headers={"X-API-Key": "wrong"}
            )
            r_ok = await c.post(
                "/query", json={"query": "x"}, headers={"X-API-Key": "secret"}
            )
            r_health = await c.get("/health")  # /health reste ouvert
        return r_no.status_code, r_bad.status_code, r_ok.status_code, r_health.status_code

    no, bad, ok, health = asyncio.run(_run())
    assert no == 401
    assert bad == 401
    assert ok == 200
    assert health == 200


# ─── Durcissement : limite de concurrence (503) ───────────────────────


def test_concurrency_limit_returns_503():
    """Avec max_concurrent=1, une 2e génération concurrente est rejetée 503."""

    async def _run() -> tuple[int, int]:
        started = asyncio.Event()
        release = asyncio.Event()

        class _SlowScathach(MorriganModule):
            last_generated_by = "slow"

            async def process(self, input: ModuleInput) -> ModuleOutput:
                started.set()
                await release.wait()  # tient le sémaphore
                return ModuleOutput(
                    result="ok", confidence=0.9, metadata={"generated_by": "slow"}
                )

            async def stream(self, input: ModuleInput) -> AsyncIterator[str]:
                yield "ok"

            async def health_check(self) -> bool:
                return True

            def get_capabilities(self) -> dict:
                return {"backend": "slow"}

        dagda = AnDagda(config_path="config/_test_does_not_exist.yaml")
        dagda.register_module("scathach", _SlowScathach())
        await dagda.initialize()
        app = create_app(dagda=dagda, max_concurrent=1)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://morrigan.test"
        ) as c:
            # 1re requête en tâche de fond : occupe l'unique permis.
            t1 = asyncio.create_task(c.post("/query", json={"query": "un"}))
            await started.wait()
            # 2e requête pendant que la 1re tient le sémaphore → 503.
            r2 = await c.post("/query", json={"query": "deux"})
            release.set()
            r1 = await t1
        return r1.status_code, r2.status_code

    s1, s2 = asyncio.run(_run())
    assert s1 == 200
    assert s2 == 503
