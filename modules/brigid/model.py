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
from modules.brigid.dataset import ID_TO_LABEL, LABELS

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


@dataclass(frozen=True)
class IntentClassification:
    """Résultat de la classification d'intention par Brigid.

    `label` est l'un des canoniques de `LABELS` (et donc un `QueryType.value`),
    `confidence` est la probabilité du label gagnant (max-softmax),
    `probabilities` mappe **chaque** label à sa proba pour traçabilité.
    """

    label: str
    confidence: float
    probabilities: Dict[str, float]


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


# ─── Brigid (MorriganModule) — vraie inférence CfC en PR C ──────────


class Brigid(MorriganModule):
    """
    Module neuronal de Morrigan — classifieur d'intention CfC.

    PR C : `Brigid.process()` charge le checkpoint au premier appel
    (lazy), encode la query via MiniLM, forward CfC, renvoie label +
    confidence + probas par classe. Le checkpoint est cherché à
    `DEFAULT_CHECKPOINT_PATH` sauf passage explicite via constructeur.

    Dégradation gracieuse : si le checkpoint manque (ex: dev qui n'a pas
    encore entraîné), `process()` renvoie un `ModuleOutput` invalide avec
    un message explicite — pas de crash, et An Dagda repasse sur ses
    heuristiques mots-clés (fallback).
    """

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self._model = None  # chargé au 1er appel
        self._load_error: Optional[str] = None
        logger.info(
            "Brigid (LNN/CfC) — sera chargée depuis %s au premier appel",
            self._checkpoint_path,
        )

    def _try_load(self) -> bool:
        """Tente de charger le checkpoint. Renvoie True si OK."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False  # déjà tenté, déjà échoué — on n'insiste pas
        try:
            self._model = load_checkpoint(self._checkpoint_path)
            logger.info("Brigid : checkpoint chargé depuis %s", self._checkpoint_path)
            return True
        except (FileNotFoundError, ValueError) as e:
            self._load_error = str(e)
            logger.warning(
                "Brigid : checkpoint non disponible (%s) — An Dagda tombera "
                "sur ses heuristiques.",
                e,
            )
            return False

    def classify_intent(self, query: str) -> Optional[IntentClassification]:
        """Classifie une query en label + confidence + probas.

        API sync, utilisable directement par An Dagda.classify_query
        (qui est sync). Renvoie `None` si le checkpoint n'est pas
        chargeable — l'appelant doit alors fallback.
        """
        if not self._try_load():
            return None

        import torch  # noqa: PLC0415
        from modules.brigid.embedder import get_embedder  # noqa: PLC0415

        # Encode → forward → softmax
        embedding = get_embedder().encode_one(query).unsqueeze(0)  # (1, 384)
        with torch.no_grad():
            logits = self._model(embedding)  # (1, num_classes)
            probas = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        confidence, best_idx = probas.max(dim=0)
        label = ID_TO_LABEL[int(best_idx.item())]

        # Toutes les probas pour traçabilité (debug / Cauldron / metadata).
        all_probas = {
            ID_TO_LABEL[i]: float(probas[i].item()) for i in range(len(LABELS))
        }
        return IntentClassification(
            label=label,
            confidence=float(confidence.item()),
            probabilities=all_probas,
        )

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Interface MorriganModule — délègue à classify_intent (sync)."""
        logger.info("Brigid traite: %s", input.query[:60])
        classif = self.classify_intent(input.query)

        if classif is None:
            # Checkpoint indisponible — renvoie un output dégradé, ne pas
            # lever d'exception pour ne pas casser le pipeline.
            return ModuleOutput(
                result={"classification": None},
                confidence=0.0,
                metadata={
                    "phase": 2,
                    "model": "CfC",
                    "loaded": False,
                    "error": self._load_error,
                },
                errors=[self._load_error or "Checkpoint non chargeable"],
            )

        return ModuleOutput(
            result={
                "classification": classif.label,
                "probabilities": classif.probabilities,
            },
            confidence=classif.confidence,
            metadata={
                "phase": 2,
                "model": "CfC",
                "loaded": True,
                "checkpoint": str(self._checkpoint_path),
            },
        )

    async def health_check(self) -> bool:
        """Considère Brigid healthy si le checkpoint est chargeable.

        Renvoie quand même `True` si le checkpoint est juste absent —
        c'est un mode dégradé attendu (dev avant entraînement), pas une
        défaillance. L'erreur est loguée à l'`__init__`.
        """
        # Tentative au premier health_check pour valider le chargement.
        self._try_load()
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        loaded = self._model is not None
        return {
            "name": "Brigid",
            "type": "neural_network",
            "architecture": "LTC/CfC (Liquid Neural Network)",
            "capabilities": [
                "intent_classification",
                "semantic_encoding",
                "creative_association",
            ],
            "phase": 2 if loaded else 1,
            "checkpoint_loaded": loaded,
        }
