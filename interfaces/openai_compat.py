"""Adaptateur OpenAI-compatible par-dessus AnDagda.

But : permettre à un client OpenAI standard (dont **Gungnir**, qui retombe sur
son `OpenAIProvider` pour tout provider custom avec un `base_url`) de parler à
Morrigan **sans rien changer côté client**. On expose le strict minimum de la
surface « Chat Completions » :

  POST /v1/chat/completions   non-stream + stream (SSE façon OpenAI)
  GET  /v1/models             liste à un seul modèle (Morrigan)

Le shim est **purement additif** : il enveloppe `AnDagda.process` /
`process_stream` (le cœur RAG strict reste intact) et n'altère aucun des
endpoints natifs `/query`. On peut le retirer en supprimant l'appel à
`add_openai_compat_routes` dans `create_app` — zéro effet de bord.

Choix d'adaptation (cf. archi Morrigan) :
  - `messages[]` → on prend le **dernier message `user`** comme requête. Morrigan
    n'est pas un modèle de chat multi-tours classique ; la mémoire de session est
    gérée par Cauldron via `session_id`, pas par le ré-envoi de l'historique.
  - `session_id` ← champ OpenAI optionnel `user` (assaini), sinon "default".
  - `usage` : Morrigan ne compte pas les tokens → estimation grossière par mots
    (modèle local souverain = coût 0 de toute façon).
  - Auth : en-tête `Authorization: Bearer <clé>` (ce qu'envoie le SDK OpenAI),
    en plus de `X-API-Key` accepté par cohérence avec l'API native.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.dagda import AnDagda

logger = logging.getLogger("morrigan.api.openai")

MODEL_ID = "morrigan"
# Borne sur la requête extraite (cohérente avec l'API native).
MAX_QUERY_CHARS = int(os.getenv("MORRIGAN_API_MAX_QUERY_CHARS", "4000"))


class _ChatMessage(BaseModel):
    role: str
    content: str = ""


class _ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[_ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    # Champ OpenAI optionnel : sert de clé de session Morrigan (Cauldron).
    user: Optional[str] = Field(default=None, max_length=128)


def _sanitize_session(user: Optional[str]) -> str:
    """Réduit le champ `user` au charset accepté par Morrigan (session_id)."""
    if not user:
        return "default"
    cleaned = "".join(c for c in user if c.isalnum() or c in "._-")[:128]
    return cleaned or "default"


def _last_user_query(messages: List[_ChatMessage]) -> str:
    """Dernier message `user` non vide → requête Morrigan."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()[:MAX_QUERY_CHARS]
    raise HTTPException(status_code=400, detail="Aucun message 'user' exploitable.")


def _estimate_tokens(text: str) -> int:
    """Estimation grossière (mots) — Morrigan ne tokenise pas le modèle ici."""
    return max(1, len(text.split()))


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def add_openai_compat_routes(app: FastAPI, *, expected_key: Optional[str]) -> None:
    """Enregistre les routes OpenAI-compatibles sur `app`.

    `expected_key` : si défini, exige `Authorization: Bearer <clé>` (ou
    `X-API-Key`). Sinon ouvert (cohérent avec l'API native sans clé).
    """

    def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
        if not expected_key:
            return
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Clé API invalide ou absente.")

    def _check_capacity(request: Request):
        sem = request.app.state.gen_semaphore
        if sem.locked():
            raise HTTPException(
                status_code=503,
                detail="Service occupé (générations simultanées max atteint).",
            )
        return sem

    @app.get("/v1/models")
    async def list_models(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> dict:
        _check_auth(authorization, x_api_key)
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "scarletwolf",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: _ChatCompletionRequest,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ):
        _check_auth(authorization, x_api_key)
        dg: AnDagda = request.app.state.dagda
        query = _last_user_query(req.messages)
        session_id = _sanitize_session(req.user)
        model = req.model or MODEL_ID
        sem = _check_capacity(request)

        if req.stream:
            return StreamingResponse(
                _stream_completion(dg, sem, request, query, session_id, model),
                media_type="text/event-stream",
            )

        async with sem:
            try:
                answer = await dg.process(query, session_id=session_id)
            except Exception:
                logger.exception("Erreur dans /v1/chat/completions")
                raise HTTPException(
                    status_code=500, detail="Erreur interne de génération."
                )

        prompt_tokens = _estimate_tokens(query)
        completion_tokens = _estimate_tokens(answer)
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


async def _stream_completion(
    dg: AnDagda,
    sem,
    request: Request,
    query: str,
    session_id: str,
    model: str,
) -> AsyncIterator[bytes]:
    """Flux SSE au format OpenAI (`chat.completion.chunk` + `[DONE]`)."""
    cid = _completion_id()
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: Optional[str]) -> bytes:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode(
            "utf-8"
        )

    async with sem:
        # 1er chunk : annonce du rôle (convention OpenAI).
        yield _chunk({"role": "assistant"}, None)
        try:
            async for piece in dg.process_stream(query, session_id=session_id):
                if await request.is_disconnected():
                    logger.info("Client déconnecté — arrêt du stream OpenAI")
                    return
                yield _chunk({"content": piece}, None)
        except Exception:
            logger.exception("Erreur dans /v1/chat/completions (stream)")
            # Pas de canal d'erreur standard en cours de flux OpenAI : on
            # clôture proprement (finish_reason) plutôt que de casser le parsing.
            yield _chunk({}, "stop")
            yield b"data: [DONE]\n\n"
            return

        yield _chunk({}, "stop")
        yield b"data: [DONE]\n\n"
