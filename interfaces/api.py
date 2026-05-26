"""
API HTTP pour Morrigan — FastAPI + SSE.

Expose AnDagda par-dessus HTTP. Endpoints :

  POST /query          JSON in/out — réponse complète
  POST /query/stream   SSE — token par token via `dagda.process_stream`
  GET  /health         vivacité + liste des modules
  GET  /stats          observabilité (format texte + compteurs JSON)

Lancement :
    uvicorn interfaces.api:app --host 0.0.0.0 --port 8000

Le dagda de production (Brigid + Ogham + build_danann + Scáthach RWKV +
Cauldron) est construit au startup via le lifespan. Tests : passer un
`dagda` à `create_app(dagda=...)` pour bypasser la construction réelle.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.dagda import AnDagda

logger = logging.getLogger("morrigan.api")


async def _build_default_dagda() -> AnDagda:
    """Compose le dagda de prod. Imports lazy pour garder l'import du
    module léger (tests + uvicorn --reload)."""
    from core.env import load_env
    from core.knowledge import build_danann
    from modules.brigid.model import Brigid
    from modules.cauldron.memory import Cauldron
    from modules.ogham.engine import Ogham
    from modules.scathach.generator import Scathach

    load_env()
    dagda = AnDagda()
    dagda.register_module("brigid", Brigid())
    dagda.register_module("ogham", Ogham())
    dagda.register_module("danann", build_danann())
    dagda.register_module("scathach", Scathach(backend="rwkv"))
    dagda.register_module("cauldron", Cauldron())
    await dagda.initialize()
    return dagda


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = "default"


class QueryResponse(BaseModel):
    response: str
    query_type: str
    modules: list[str]
    domain_hint: Optional[str] = None
    generated_by: Optional[str] = None
    latency_s: float


def _routing_payload(dagda: AnDagda) -> dict:
    """Snapshot du dernier routage (mis à jour par process/process_stream)."""
    routing = dagda.last_routing
    return {
        "query_type": routing.query_type.value if routing else "unknown",
        "modules": list(routing.modules) if routing else [],
        "domain_hint": routing.domain_hint if routing else None,
        "generated_by": dagda.last_generated_by,
        "latency_s": dagda.last_latency_s,
    }


def create_app(dagda: Optional[AnDagda] = None) -> FastAPI:
    """Construit l'app FastAPI.

    Si `dagda` est fourni, il est utilisé tel quel (tests). Sinon le
    lifespan compose le dagda de prod au startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Le dagda peut déjà être attaché (passé en argument) — utile pour
        # les tests qui n'exécutent pas le cycle ASGI lifespan.
        if getattr(app.state, "dagda", None) is None:
            logger.info("API : construction du dagda de prod…")
            app.state.dagda = await _build_default_dagda()
        yield

    app = FastAPI(
        title="Morrigan API",
        version="0.1",
        description=(
            "API HTTP par-dessus AnDagda. POST /query (JSON), "
            "POST /query/stream (SSE), GET /health, GET /stats."
        ),
        lifespan=lifespan,
    )
    if dagda is not None:
        app.state.dagda = dagda

    @app.get("/health")
    async def health(request: Request) -> dict:
        dg: AnDagda = request.app.state.dagda
        return {"status": "ok", "modules": list(dg.modules.keys())}

    @app.get("/stats")
    async def stats(request: Request) -> dict:
        dg: AnDagda = request.app.state.dagda
        return {"text": dg.format_stats(), **dg.stats}

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest, request: Request) -> QueryResponse:
        dg: AnDagda = request.app.state.dagda
        response = await dg.process(req.query, session_id=req.session_id)
        payload = _routing_payload(dg)
        return QueryResponse(response=response, **payload)

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
        dg: AnDagda = request.app.state.dagda

        async def event_stream() -> AsyncIterator[bytes]:
            try:
                async for piece in dg.process_stream(
                    req.query, session_id=req.session_id
                ):
                    line = "data: " + json.dumps({"piece": piece}, ensure_ascii=False)
                    yield (line + "\n\n").encode("utf-8")
            except Exception as exc:  # pragma: no cover — défensif
                logger.exception("Erreur dans /query/stream")
                err = json.dumps({"message": str(exc)}, ensure_ascii=False)
                yield (f"event: error\ndata: {err}\n\n").encode("utf-8")
                return

            done = json.dumps(_routing_payload(dg), ensure_ascii=False)
            yield (f"event: done\ndata: {done}\n\n").encode("utf-8")

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


def main() -> None:
    """Entry point : `python -m interfaces.api`."""
    import uvicorn

    uvicorn.run("interfaces.api:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
