"""
MORRIGAN — Interface Telegram.

Bot Telegram basé sur python-telegram-bot v22 (async natif).
Chaque chat Telegram = une session Cauldron isolée.

Lancement :
    set MORRIGAN_TELEGRAM_TOKEN=xxx:yyy
    .venv/Scripts/python interfaces/telegram.py
"""

import asyncio
import logging
import os
import sys
import time
from typing import AsyncIterator, Awaitable, Callable

sys.path.insert(0, ".")

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.dagda import AnDagda
from core.env import load_env
from core.knowledge import build_danann
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron

logger = logging.getLogger("morrigan.telegram")

# Limite Telegram : 4096 caracteres par message
TELEGRAM_MAX_LEN = 4000

# Intervalle minimal entre deux editions de message (anti flood-control
# Telegram : ~1 edit/s est sûr).
STREAM_EDIT_INTERVAL = 1.0


async def stream_collect(
    pieces: AsyncIterator[str],
    edit: Callable[[str, bool], Awaitable[None]],
    *,
    interval: float = STREAM_EDIT_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Consomme un flux de morceaux et édite un message au fil de l'eau.

    - Accumule les morceaux ; appelle `edit(texte, final=False)` au plus
      une fois toutes `interval` secondes (throttle anti flood Telegram).
    - Appelle `edit(texte, final=True)` à la fin avec le texte complet.
    - Les erreurs d'édition (ex: "message not modified", flood) sont
      avalées : une édition ratée ne casse pas le flux.

    Renvoie le texte final accumulé. `clock` est injectable pour les tests.
    """
    accumulated = ""
    last_edit = 0.0
    async for piece in pieces:
        accumulated += piece
        now = clock()
        if accumulated.strip() and (now - last_edit) >= interval:
            last_edit = now
            try:
                await edit(accumulated, False)
            except Exception:  # noqa: BLE001 - édition best-effort
                pass
    try:
        await edit(accumulated, True)
    except Exception:  # noqa: BLE001
        pass
    return accumulated


def _session_id(update: Update) -> str:
    """ID de session base sur le chat_id Telegram."""
    return f"tg_{update.effective_chat.id}"


def _chunk_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Decoupe un texte long en morceaux <= limit pour respecter Telegram."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        cut = text[:limit]
        # Essayer de couper sur un retour a la ligne
        nl = cut.rfind("\n")
        if nl > limit // 2:
            cut = text[:nl]
        parts.append(cut)
        text = text[len(cut):].lstrip()
    return parts


class MorriganTelegramBot:
    """Wrapper Telegram autour d'An Dagda."""

    def __init__(self, dagda: AnDagda, cauldron: Cauldron):
        self.dagda = dagda
        self.cauldron = cauldron

    async def cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Morrigan s'eveille.\n\n"
            "Architecture IA modulaire post-LLM.\n"
            "Pose-moi une question sur les reseaux, la mythologie celtique, "
            "les architectures IA alternatives, ou le projet lui-meme.\n\n"
            "Commandes :\n"
            "/help  — aide\n"
            "/reset — efface ma memoire de notre conversation\n"
            "/stats — statistiques des modules"
        )

    async def cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Morrigan decompose chaque requete a travers plusieurs modules :\n"
            "- An Dagda  : orchestration\n"
            "- Brigid    : pattern recognition (LNN)\n"
            "- Ogham     : raisonnement symbolique\n"
            "- Danann    : memoire vectorielle\n"
            "- Scathach  : generation de texte (RWKV, streaming)\n"
            "- Cauldron  : memoire de conversation\n\n"
            "Generation neuronale RWKV avec RAG strict (0 hallucination) : "
            "je reponds depuis mon corpus, ou j'admets que je ne sais pas. "
            "La reponse s'affiche au fil de l'eau."
        )

    async def cmd_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        sid = _session_id(update)
        self.cauldron.sessions.pop(sid, None)
        await update.message.reply_text("Memoire de conversation effacee.")

    async def cmd_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # Observabilité : compteurs cumulés + trace de la dernière requête
        # (type, routage, Brigid + probas, generated_by, latence).
        await update.message.reply_text(self.dagda.format_stats())

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_input = update.message.text
        sid = _session_id(update)
        logger.info("TG[%s]: %s", sid, user_input[:80])

        # Feedback visuel + message placeholder qu'on va éditer au fil de l'eau.
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        placeholder = await update.message.reply_text("▌")

        async def _edit(text: str, final: bool) -> None:
            if final:
                display = (text.strip() or "[Morrigan] (réponse vide)")
                display = _chunk_message(display)[0]  # 1er bloc ; reste envoyé après
            else:
                # Curseur pour montrer que la génération est en cours.
                display = text[:TELEGRAM_MAX_LEN].rstrip() + " ▌"
            await placeholder.edit_text(display)

        try:
            response = await stream_collect(
                self.dagda.process_stream(user_input, session_id=sid), _edit
            )
        except Exception as exc:
            logger.exception("Erreur lors du traitement")
            response = f"[Morrigan] Erreur interne : {exc}"
            try:
                await placeholder.edit_text(response)
            except Exception:  # noqa: BLE001
                pass

        # Memoriser dans Cauldron (An Dagda ne le fait pas automatiquement)
        response = response.strip() or "[Morrigan] (réponse vide)"
        self.cauldron.add_turn(sid, "user", user_input)
        self.cauldron.add_turn(sid, "morrigan", response)

        # Débordement : si la réponse dépasse la limite Telegram, le 1er
        # bloc est déjà dans le message édité ; on envoie le reste à part.
        parts = _chunk_message(response)
        for extra in parts[1:]:
            await update.message.reply_text(extra)


async def _build_dagda() -> tuple[AnDagda, Cauldron]:
    """Construit et initialise l'orchestrateur avec tous les modules."""
    dagda = AnDagda()
    cauldron = Cauldron()

    dagda.register_module("brigid", Brigid())
    dagda.register_module("ogham", Ogham())
    # Danann : index persisté (MORRIGAN_INDEX) si présent, sinon
    # ingestion de data/knowledge. Cf. core.knowledge.build_danann.
    dagda.register_module("danann", build_danann())
    # backend RWKV → vraie génération + streaming (fallback template si
    # le modèle GGUF est absent).
    dagda.register_module("scathach", Scathach(backend="rwkv"))
    dagda.register_module("cauldron", cauldron)

    await dagda.initialize()

    return dagda, cauldron


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Charge .env si présent (token, clés…) — sinon env réel.
    load_env()

    token = os.getenv("MORRIGAN_TELEGRAM_TOKEN", "").strip()
    if not token:
        print(
            "ERREUR : variable d'environnement MORRIGAN_TELEGRAM_TOKEN absente.\n"
            "Définis-la dans un fichier .env (cf. .env.example) :\n"
            "    MORRIGAN_TELEGRAM_TOKEN=123456:ABC-xxx\n"
            "ou en ligne :\n"
            "    MORRIGAN_TELEGRAM_TOKEN=123456:ABC-xxx python interfaces/telegram_bot.py"
        )
        sys.exit(1)

    # On initialise Dagda dans une boucle eventloop dediee avant de demarrer
    # le bot. app.run_polling() creera ensuite sa propre loop.
    dagda, cauldron = asyncio.run(_build_dagda())

    bot = MorriganTelegramBot(dagda, cauldron)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("help", bot.cmd_help))
    app.add_handler(CommandHandler("reset", bot.cmd_reset))
    app.add_handler(CommandHandler("stats", bot.cmd_stats))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message)
    )

    logger.info("Morrigan Telegram bot demarre. Ctrl+C pour arreter.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
