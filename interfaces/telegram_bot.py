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
from pathlib import Path

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
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.danann.store import Danann
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron
from scripts.ingest_knowledge import ingest_directory

logger = logging.getLogger("morrigan.telegram")

# Limite Telegram : 4096 caracteres par message
TELEGRAM_MAX_LEN = 4000


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
            "- Scathach  : generation de texte\n"
            "- Cauldron  : memoire de conversation\n\n"
            "Phase 1 : generation par templates. RWKV viendra en Phase 2."
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
        lines = ["Modules enregistres :"]
        for name, module in self.dagda.modules.items():
            caps = module.get_capabilities()
            extra = ""
            if "indexed_chunks" in caps:
                extra = f" ({caps['indexed_chunks']} chunks)"
            lines.append(f"- {name}{extra}")
        await update.message.reply_text("\n".join(lines))

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_input = update.message.text
        sid = _session_id(update)
        logger.info("TG[%s]: %s", sid, user_input[:80])

        # Feedback visuel pendant le traitement
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            response = await self.dagda.process(user_input, session_id=sid)
        except Exception as exc:
            logger.exception("Erreur lors du traitement")
            response = f"[Morrigan] Erreur interne : {exc}"

        # Memoriser dans Cauldron (An Dagda ne le fait pas automatiquement)
        self.cauldron.add_turn(sid, "user", user_input)
        self.cauldron.add_turn(sid, "morrigan", response)

        for part in _chunk_message(response):
            await update.message.reply_text(part)


async def _build_dagda() -> tuple[AnDagda, Cauldron]:
    """Construit et initialise l'orchestrateur avec tous les modules."""
    dagda = AnDagda()
    cauldron = Cauldron()

    dagda.register_module("brigid", Brigid())
    dagda.register_module("ogham", Ogham())
    dagda.register_module("danann", Danann(backend="memory"))
    dagda.register_module("scathach", Scathach())
    dagda.register_module("cauldron", cauldron)

    await dagda.initialize()

    # Ingestion automatique du corpus
    knowledge_dir = Path("data/knowledge")
    if knowledge_dir.exists():
        danann = dagda.modules["danann"]
        total = ingest_directory(danann, knowledge_dir)
        logger.info("Corpus charge : %d chunks depuis %s", total, knowledge_dir)
    else:
        logger.warning("Dossier %s introuvable — Danann vide", knowledge_dir)

    return dagda, cauldron


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    token = os.getenv("MORRIGAN_TELEGRAM_TOKEN", "").strip()
    if not token:
        print(
            "ERREUR : variable d'environnement MORRIGAN_TELEGRAM_TOKEN absente.\n"
            "Definissez-la avant de lancer le bot :\n"
            "    set MORRIGAN_TELEGRAM_TOKEN=123456:ABC-xxx"
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
