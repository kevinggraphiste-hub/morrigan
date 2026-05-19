"""
BRIGID — Architecture LNN (Liquid Neural Network).

Deux choses cohabitent dans ce module :

1. `Brigid(MorriganModule)` : le module enregistré dans An Dagda.
   Reste un **squelette placeholder** tant que PR C (inférence +
   intégration) n'est pas livrée. Il ne consomme pas le modèle entraîné.

2. `IntentClassifier(nn.Module)` : le **modèle CfC entraînable**
   (Liquid Time-Constant Closed-form). Utilisé par
   `scripts/train_brigid.py` pour produire le checkpoint, sera chargé
   par PR C pour faire de la vraie classification.

Cette séparation est volontaire : on livre l'entraînement et le
checkpoint d'abord, sans toucher au pipeline existant ; PR C wire le
classifieur dans `Brigid.process()` et An Dagda. Pas de régression de
routage possible dans PR B.

Architecture du classifieur :

    query (str)
      └─> IntentEmbedder (MiniLM, 384-D, normalisé)
            └─> CfC(input=384, units=hidden_dim)  ← cœur LNN
                  └─> Linear(hidden_dim → 6)
                        └─> 6 logits (cross-entropy)

Petit volontairement : on cible un modèle compact (< 100 KB) pour
honorer la promesse "tourne sur PC modeste". Sur 504 exemples de
classification, 16 unités liquides suffisent largement.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.types import ModuleInput, ModuleOutput, MorriganModule
from modules.brigid.dataset import LABELS

logger = logging.getLogger("morrigan.brigid")

# Valeur par défaut pour la taille de la couche liquide. Configurable
# via le script de training. Doit suffire pour 6 classes + 504 ex.
DEFAULT_HIDDEN_DIM = 16

# Chemin par défaut du checkpoint (gitignoré : *.pt). Réentraîné au CI.
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "models"
    / "brigid_cfc.pt"
)


@dataclass
class CheckpointMetadata:
    """Métadonnées sauvegardées avec le state_dict du modèle.

    Sert à reconstruire l'architecture à l'identique et à détecter une
    incompatibilité au load (changement d'embedder, de classes, …).
    """

    input_dim: int
    hidden_dim: int
    num_classes: int
    labels: Tuple[str, ...]
    embed_model_name: str
    # Métriques finales — info, pas de validation.
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    epochs: Optional[int] = None
    seed: Optional[int] = None


# ─── IntentClassifier (CfC réel) ────────────────────────────────────


def build_classifier(
    input_dim: int = 384,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_classes: int = len(LABELS),
    seed: Optional[int] = None,
):
    """Construit un `IntentClassifier` prêt à entraîner.

    Import retardé de torch / ncps : ce module reste importable même
    sans ces dépendances (utile pour les outils légers).
    """
    import torch  # noqa: PLC0415
    from ncps.torch import CfC  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    if seed is not None:
        torch.manual_seed(seed)

    class IntentClassifier(nn.Module):
        """Embedding 384-D → CfC(hidden) → Linear → logits 6.

        Note : CfC est conçu pour des séries temporelles. Pour de la
        classification one-shot d'embedding, on présente la séquence
        de longueur 1 et on lit le dernier timestep. Pragmatique et
        cohérent avec l'identité LNN du projet.
        """

        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.cfc = CfC(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x : (B, input_dim). On wrappe en (B, T=1, input_dim).
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # ncps CfC renvoie (output, final_hidden_state). output a
            # la forme (B, T, hidden_dim) en mode batch_first.
            output, _ = self.cfc(x)
            last = output[:, -1, :]  # (B, hidden_dim)
            return self.head(last)  # (B, num_classes)

    model = IntentClassifier()
    return model


def save_checkpoint(
    model,
    path: Path = DEFAULT_CHECKPOINT_PATH,
    metadata: Optional[CheckpointMetadata] = None,
) -> Path:
    """Sauve le state_dict + métadonnées dans un seul fichier `.pt`."""
    import torch  # noqa: PLC0415

    from modules.brigid.embedder import EMBED_DIM, EMBED_MODEL_NAME  # noqa: PLC0415

    if metadata is None:
        metadata = CheckpointMetadata(
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
            num_classes=model.num_classes,
            labels=LABELS,
            embed_model_name=EMBED_MODEL_NAME,
        )

    # Sanity : embedder cohérent avec l'architecture.
    if metadata.input_dim != EMBED_DIM:
        raise ValueError(
            f"input_dim {metadata.input_dim} != EMBED_DIM {EMBED_DIM} ; "
            "incohérence checkpoint ↔ embedder."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": asdict(metadata),
        },
        path,
    )
    logger.info("Checkpoint sauvé : %s", path)
    return path


def load_checkpoint(path: Path = DEFAULT_CHECKPOINT_PATH):
    """Recharge un classifieur à partir d'un fichier `.pt`.

    Vérifie que `embed_model_name` et `labels` correspondent toujours
    à la config courante — sinon le checkpoint est sémantiquement
    obsolète et il faut le réentraîner.
    """
    import torch  # noqa: PLC0415

    from modules.brigid.embedder import EMBED_MODEL_NAME  # noqa: PLC0415

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}. "
            "Lance `python scripts/train_brigid.py` (ou la CI brigid-train)."
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]

    if meta["embed_model_name"] != EMBED_MODEL_NAME:
        raise ValueError(
            f"Embedder du checkpoint ({meta['embed_model_name']}) != "
            f"embedder courant ({EMBED_MODEL_NAME}). Réentraîne."
        )
    if tuple(meta["labels"]) != LABELS:
        raise ValueError(
            f"Labels du checkpoint {meta['labels']} != LABELS courants "
            f"{LABELS}. Réentraîne après avoir aligné l'ordre."
        )

    model = build_classifier(
        input_dim=meta["input_dim"],
        hidden_dim=meta["hidden_dim"],
        num_classes=meta["num_classes"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


# ─── Brigid (MorriganModule) — toujours placeholder en PR B ─────────


class Brigid(MorriganModule):
    """
    Module neuronal de Morrigan.

    PR B (cette PR) : `IntentClassifier` et `train_brigid.py` livrés,
    mais `Brigid.process()` reste un placeholder. C'est volontaire :
    PR C wire le classifieur ici et dans An Dagda en une étape
    atomique testable séparément.

    PR C : `process()` charge le checkpoint via `load_checkpoint()`,
    embed la query, fait la forward pass, renvoie label + confidence.
    """

    def __init__(self) -> None:
        self.initialized = False
        logger.info("Brigid (LNN) — squelette ; classifieur livré en PR B mais non wiré")

    async def process(self, input: ModuleInput) -> ModuleOutput:
        logger.info("Brigid traite: %s", input.query[:60])
        return ModuleOutput(
            result={"patterns": [], "classification": "unknown"},
            confidence=0.1,
            metadata={"phase": 0, "note": "Squelette — wiring en PR C"},
        )

    async def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Brigid",
            "type": "neural_network",
            "architecture": "LTC/CfC (Liquid Neural Network)",
            "capabilities": [
                "intent_classification",
                "semantic_encoding",
                "creative_association",
            ],
            "phase": 0,
        }
