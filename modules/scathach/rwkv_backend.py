"""
SCÁTHACH — Backend RWKV (génération neuronale, Phase 3).

Inférence RWKV quantizé via llama.cpp (`llama-cpp-python`). RWKV est un
RNN à attention linéaire : mémoire constante, inférence linéaire en
longueur, CPU-friendly — cohérent avec la philo Morrigan "PC modeste".

Modèle par défaut : RWKV-6 World 1.6B, quantization Q4_K (~993 Mo, GGUF).
Validé en local : génère du français cohérent à ~10-12 tok/s sur CPU.
Q2_K testé mais trop agressif (sortie dégénérée) → Q4_K est le plancher
de qualité. Le fichier GGUF est gitignoré (option B : artefact
téléchargé via `scripts/fetch_rwkv_model.py`, pas commité).

Dégradation gracieuse : si `llama_cpp` n'est pas installé ou le modèle
absent, le backend est "indisponible" et Scáthach (PR B) retombe sur
ses templates Jinja2 — pas de crash.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger("morrigan.scathach.rwkv")

# Repo + fichier GGUF par défaut (cf. scripts/fetch_rwkv_model.py).
DEFAULT_REPO = "latestissue/rwkv-6-world-1b6-gguf"
DEFAULT_FILENAME = "rwkv-6-world-1.6b-Q4_K.gguf"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "models" / DEFAULT_FILENAME
)

# Defaults de sampling RWKV. repeat_penalty élevé est ESSENTIEL :
# sans lui, RWKV part en boucle ("de-de-de…"). Validé empiriquement.
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.7
DEFAULT_REPEAT_PENALTY = 1.3
DEFAULT_MAX_TOKENS = 256

# Format de prompt RWKV World. Le modèle est entraîné sur ce schéma
# exact ("User:" / "Assistant:" séparés par double newline) — s'en
# écarter dégrade fortement la qualité.
_STOP_SEQUENCES = ["\n\nUser:", "\n\nUser :", "\nUser:"]


class RWKVBackend:
    """Backend de génération RWKV via llama.cpp.

    Chargement paresseux : le modèle (~1 Go) n'est chargé qu'au premier
    `generate()`. `is_available()` permet à l'appelant (Scáthach) de
    décider du fallback sans déclencher le chargement.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: int = 1024,
        n_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None
        self._load_error: Optional[str] = None
        # llama.cpp partage un unique contexte : `self._llm(...)` n'est PAS
        # sûr en appels concurrents. On sérialise l'inférence (de toute façon
        # CPU-bound mono-contexte) pour pouvoir l'appeler depuis des threads
        # (offload hors event-loop côté Scáthach).
        self._infer_lock = threading.Lock()

    # ─── Chargement ─────────────────────────────────────────────

    def _try_load(self) -> bool:
        """Charge le modèle au 1er usage. Renvoie True si OK."""
        if self._llm is not None:
            return True
        if self._load_error is not None:
            return False  # déjà tenté, déjà échoué

        if not self.model_path.exists():
            self._load_error = (
                f"Modèle GGUF absent : {self.model_path}. "
                f"Lance `python scripts/fetch_rwkv_model.py`."
            )
            logger.warning("RWKV indisponible — %s", self._load_error)
            return False

        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as e:
            self._load_error = f"llama-cpp-python non installé ({e})"
            logger.warning("RWKV indisponible — %s", self._load_error)
            return False

        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            logger.info("RWKV chargé : %s", self.model_path.name)
            return True
        except Exception as e:  # pragma: no cover - dépend de l'env llama.cpp
            self._load_error = f"Échec chargement llama.cpp : {e}"
            logger.error("RWKV — %s", self._load_error)
            return False

    def is_available(self) -> bool:
        """True si le backend peut générer (lib + modèle présents)."""
        return self._try_load()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ─── Prompt ─────────────────────────────────────────────────

    @staticmethod
    def format_prompt(
        query: str,
        context: Optional[List[str]] = None,
        strict: bool = True,
    ) -> str:
        """Construit un prompt au format RWKV World.

        - `context` : chunks Danann + faits Ogham. Injectés avant la
          question (RAG).
        - `strict` (défaut True) : instruit le modèle de répondre
          UNIQUEMENT à partir du contexte et de dire "Je ne sais pas"
          sinon. C'est le cœur du "0 hallucination" de Morrigan — on
          ne peut pas garantir à 100 % qu'un 1.6B obéisse, mais le
          grounding + l'instruction réduisent fortement l'invention.
          Sans contexte, l'appelant (Scáthach) ne devrait pas appeler
          le LLM en mode strict (il renvoie un "je ne sais pas"
          déterministe via template).
        """
        if context:
            ctx = "\n".join(f"- {c}" for c in context if c.strip())
            if strict:
                user = (
                    "Réponds à la question en t'appuyant UNIQUEMENT sur les "
                    "informations ci-dessous. Si la réponse ne s'y trouve pas, "
                    'réponds exactement "Je ne sais pas.".\n\n'
                    f"Informations :\n{ctx}\n\n"
                    f"Question : {query}"
                )
            else:
                user = (
                    "En t'appuyant sur ces informations :\n"
                    f"{ctx}\n\n"
                    f"Réponds à la question : {query}"
                )
        else:
            user = query
        return f"User: {user}\n\nAssistant:"

    # ─── Génération ─────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        seed: Optional[int] = None,
    ) -> str:
        """Génère du texte à partir d'un prompt déjà formaté.

        Lève `RuntimeError` si le backend n'est pas disponible — c'est à
        l'appelant de vérifier `is_available()` d'abord (Scáthach le fait
        et retombe sur ses templates sinon).
        """
        if not self._try_load():
            raise RuntimeError(
                f"RWKVBackend indisponible : {self._load_error}"
            )

        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": _STOP_SEQUENCES,
        }
        if seed is not None:
            kwargs["seed"] = seed

        with self._infer_lock:
            out = self._llm(prompt, **kwargs)  # type: ignore[misc]
        return out["choices"][0]["text"].strip()

    def answer(
        self,
        query: str,
        context: Optional[List[str]] = None,
        strict: bool = True,
        **gen_kwargs,
    ) -> str:
        """Raccourci : formate le prompt RWKV World (strict par défaut) puis génère."""
        prompt = self.format_prompt(query, context, strict=strict)
        return self.generate(prompt, **gen_kwargs)

    # ─── Streaming ──────────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        """Génère en streaming : yield les morceaux de texte au fil de l'eau.

        Ne réduit pas le temps TOTAL, mais le 1er token arrive en <1s —
        l'utilisateur voit la réponse se construire au lieu d'attendre.
        C'est le levier "ressenti" sur CPU lent (cf. docs/benchmarks.md).

        Lève RuntimeError si le backend n'est pas disponible.
        """
        if not self._try_load():
            raise RuntimeError(f"RWKVBackend indisponible : {self._load_error}")

        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": _STOP_SEQUENCES,
            "stream": True,
        }
        if seed is not None:
            kwargs["seed"] = seed

        # Le lock couvre toute l'itération : un seul stream à la fois sur le
        # contexte llama.cpp (libéré quand le générateur s'épuise ou est fermé).
        with self._infer_lock:
            for chunk in self._llm(prompt, **kwargs):  # type: ignore[misc]
                piece = chunk["choices"][0]["text"]
                if piece:
                    yield piece

    def answer_stream(
        self,
        query: str,
        context: Optional[List[str]] = None,
        strict: bool = True,
        **gen_kwargs,
    ) -> Iterator[str]:
        """Streaming version de answer() : formate puis yield les morceaux."""
        prompt = self.format_prompt(query, context, strict=strict)
        yield from self.generate_stream(prompt, **gen_kwargs)
