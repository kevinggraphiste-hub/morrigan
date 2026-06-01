"""
API HTTP pour Morrigan — FastAPI + SSE.

Expose AnDagda par-dessus HTTP. Endpoints :

  POST /query          JSON in/out — réponse complète
  POST /query/stream   SSE — token par token via `dagda.process_stream`
  GET  /health         vivacité + liste des modules (toujours ouvert)
  GET  /stats          observabilité (format texte + compteurs JSON)

Lancement :
    python -m interfaces.api          # bind 127.0.0.1:8000 par défaut

Le dagda de production (Brigid + Ogham + build_danann + Scáthach RWKV +
Cauldron) est construit au startup via le lifespan. Tests : passer un
`dagda` à `create_app(dagda=...)` pour bypasser la construction réelle.

Durcissement (cf. audit 2026-05-29) :
  - Concurrence des générations bornée par un sémaphore → 503 si saturé
    (la génération RWKV CPU est coûteuse : sans borne, DoS trivial).
  - `query` / `session_id` bornés en longueur (et charset pour la session).
  - SSE : la génération s'arrête si le client se déconnecte.
  - Erreurs : message générique au client, détail loggé côté serveur.
  - Auth optionnelle par clé API (`MORRIGAN_API_KEY`) sur /query, /stats.
  - Bind sur 127.0.0.1 par défaut (exposition réseau = choix explicite).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.dagda import AnDagda

logger = logging.getLogger("morrigan.api")

# ── Limites (surchargées par l'environnement) ───────────────────────────
# Bornes statiques sur les entrées : empêchent un prompt/charge illimité.
MAX_QUERY_CHARS = int(os.getenv("MORRIGAN_API_MAX_QUERY_CHARS", "4000"))
MAX_SESSION_CHARS = int(os.getenv("MORRIGAN_API_MAX_SESSION_CHARS", "128"))
# Générations RWKV simultanées max (CPU-bound). Au-delà → 503.
DEFAULT_MAX_CONCURRENT = int(os.getenv("MORRIGAN_API_MAX_CONCURRENT", "2"))


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
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    # Charset borné : le session_id sert de clé de session en mémoire ;
    # on évite les valeurs exotiques/illimitées.
    session_id: str = Field(
        "default", max_length=MAX_SESSION_CHARS, pattern=r"^[A-Za-z0-9._\-]+$"
    )


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


def _check_capacity(request: Request) -> asyncio.Semaphore:
    """Renvoie le sémaphore de génération si une place est libre, sinon 503.

    Borne le nombre de générations concurrentes (RWKV CPU = coûteux). La
    vérification est non bloquante : une requête de trop est rejetée plutôt
    que mise en file d'attente illimitée."""
    sem: asyncio.Semaphore = request.app.state.gen_semaphore
    if sem.locked():
        raise HTTPException(
            status_code=503,
            detail="Service occupé (générations simultanées max atteint).",
        )
    return sem


def create_app(
    dagda: Optional[AnDagda] = None,
    *,
    api_key: Optional[str] = None,
    max_concurrent: Optional[int] = None,
) -> FastAPI:
    """Construit l'app FastAPI.

    Si `dagda` est fourni, il est utilisé tel quel (tests). Sinon le
    lifespan compose le dagda de prod au startup.

    `api_key` (ou env `MORRIGAN_API_KEY`) : si défini, /query, /query/stream
    et /stats exigent l'en-tête `X-API-Key`. /health reste ouvert (sonde de
    vivacité). `max_concurrent` (ou env) borne les générations simultanées.
    """
    expected_key = api_key if api_key is not None else os.getenv("MORRIGAN_API_KEY")
    concurrency = max_concurrent or DEFAULT_MAX_CONCURRENT

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
    app.state.gen_semaphore = asyncio.Semaphore(concurrency)

    async def require_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        """Dépendance d'auth : no-op si aucune clé n'est configurée."""
        if expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Clé API invalide ou absente.")

    @app.get("/health")
    async def health(request: Request) -> dict:
        dg: AnDagda = request.app.state.dagda
        return {"status": "ok", "modules": list(dg.modules.keys())}

    @app.get("/stats", dependencies=[Depends(require_api_key)])
    async def stats(request: Request) -> dict:
        dg: AnDagda = request.app.state.dagda
        return {"text": dg.format_stats(), **dg.stats}

    @app.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def query(req: QueryRequest, request: Request) -> QueryResponse:
        dg: AnDagda = request.app.state.dagda
        sem = _check_capacity(request)
        async with sem:
            try:
                response = await dg.process(req.query, session_id=req.session_id)
            except Exception:
                logger.exception("Erreur dans /query")
                raise HTTPException(status_code=500, detail="Erreur interne de génération.")
        payload = _routing_payload(dg)
        return QueryResponse(response=response, **payload)

    @app.post("/query/stream", dependencies=[Depends(require_api_key)])
    async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
        dg: AnDagda = request.app.state.dagda
        # Capacité vérifiée avant d'ouvrir le flux (pour pouvoir renvoyer un
        # vrai 503) ; le sémaphore est tenu pendant toute la génération.
        sem = _check_capacity(request)

        async def event_stream() -> AsyncIterator[bytes]:
            async with sem:
                try:
                    async for piece in dg.process_stream(
                        req.query, session_id=req.session_id
                    ):
                        # Le client a coupé : on arrête la génération au lieu
                        # de la laisser tourner dans le vide (zombie).
                        if await request.is_disconnected():
                            logger.info("Client déconnecté — arrêt du stream")
                            return
                        line = "data: " + json.dumps(
                            {"piece": piece}, ensure_ascii=False
                        )
                        yield (line + "\n\n").encode("utf-8")
                except Exception:
                    logger.exception("Erreur dans /query/stream")
                    err = json.dumps(
                        {"message": "Erreur interne de génération."},
                        ensure_ascii=False,
                    )
                    yield (f"event: error\ndata: {err}\n\n").encode("utf-8")
                    return

                done = json.dumps(_routing_payload(dg), ensure_ascii=False)
                yield (f"event: done\ndata: {done}\n\n").encode("utf-8")

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


def main() -> None:
    """Entry point : `python -m interfaces.api`.

    Bind sur 127.0.0.1 par défaut : exposer l'API sur le réseau est un
    choix explicite (`MORRIGAN_API_HOST=0.0.0.0`), et devrait s'accompagner
    d'une clé API (`MORRIGAN_API_KEY`) et/ou d'un reverse proxy.
    """
    import uvicorn

    host = os.getenv("MORRIGAN_API_HOST", "127.0.0.1")
    port = int(os.getenv("MORRIGAN_API_PORT", "8000"))
    uvicorn.run("interfaces.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
