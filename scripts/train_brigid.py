"""
Entraîne le classifieur d'intention Brigid (CfC) sur le dataset livré
dans `data/training/intent_classification.jsonl`.

Pipeline :
    JSONL  ─>  split stratifié  ─>  embeddings MiniLM (384-D)
                                       │
                                       ▼
                            CfC(hidden) ─> Linear(6)  ──>  CE loss
                                       │
                                       ▼
                          checkpoint `data/models/brigid_cfc.pt`

Sortie console : courbe loss + accuracy par époque, accuracy par
classe à la fin, taille du checkpoint.

Sortie code :
  - 0 si val_accuracy >= --min-accuracy
  - 1 sinon (utilisé par la CI `brigid-train.yml` comme garde-fou)

Usage typique (local) :
    .venv-uv/bin/python scripts/train_brigid.py --epochs 80

Usage CI : voir `.github/workflows/brigid-train.yml`.

CPU only : la philo Morrigan exige que tout tourne sur PC modeste.
Pour 504 exemples × 80 époques × ~60 K params, un i5 fait < 30 s.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

# Sans cet ajout, le script ne peut pas importer `modules.*` quand on
# le lance depuis la racine du repo (cas typique en CI et en local).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.brigid.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    ID_TO_LABEL,
    LABELS,
    LabeledExample,
    load_dataset,
    split_train_val,
)
from modules.brigid.embedder import EMBED_DIM, EMBED_MODEL_NAME, get_embedder  # noqa: E402
from modules.brigid.model import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    CheckpointMetadata,
    build_classifier,
    save_checkpoint,
)

logger = logging.getLogger("morrigan.brigid.train")


def _set_seed(seed: int) -> None:
    """Fixe la seed pour reproductibilité — torch + python + numpy si dispo."""
    import torch  # noqa: PLC0415

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass


def _embed_examples(examples: list[LabeledExample]):
    """Calcule les embeddings + labels-ids → (X, y) tensors."""
    import torch  # noqa: PLC0415

    embedder = get_embedder()
    texts = [ex.query for ex in examples]
    X = embedder.encode(texts)  # (N, 384), déjà CPU
    y = torch.tensor([ex.label_id for ex in examples], dtype=torch.long)
    return X, y


def _evaluate(model, X, y) -> tuple[float, dict[str, float]]:
    """Retourne (accuracy globale, accuracy par classe)."""
    import torch  # noqa: PLC0415

    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct = (preds == y).float()
        acc = correct.mean().item()

        per_class: dict[str, float] = {}
        for class_id, label in ID_TO_LABEL.items():
            mask = y == class_id
            if mask.sum() > 0:
                per_class[label] = correct[mask].mean().item()
    model.train()
    return acc, per_class


def train(args: argparse.Namespace) -> int:
    """Boucle d'entraînement. Retourne le code de sortie pour `main`."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    _set_seed(args.seed)

    # ─── Données ──────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    logger.info("Chargement du dataset : %s", dataset_path)
    examples = load_dataset(dataset_path)
    train_ex, val_ex = split_train_val(
        examples, val_ratio=args.val_ratio, seed=args.seed
    )
    logger.info(
        "Split : %d train / %d val (val_ratio=%.2f, seed=%d)",
        len(train_ex), len(val_ex), args.val_ratio, args.seed,
    )

    # ─── Embeddings (une fois pour toutes) ───────────────────────
    t0 = time.time()
    logger.info("Calcul des embeddings…")
    X_train, y_train = _embed_examples(train_ex)
    X_val, y_val = _embed_examples(val_ex)
    logger.info(
        "Embeddings : train %s, val %s (%.1fs)",
        tuple(X_train.shape), tuple(X_val.shape), time.time() - t0,
    )

    # ─── Modèle ───────────────────────────────────────────────────
    model = build_classifier(
        input_dim=EMBED_DIM,
        hidden_dim=args.hidden,
        num_classes=len(LABELS),
        seed=args.seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Modèle : %d paramètres (hidden=%d)", n_params, args.hidden)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ─── Boucle d'entraînement ────────────────────────────────────
    best_val_acc = 0.0
    best_state = None
    n_train = len(train_ex)
    indices = torch.arange(n_train)
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Shuffle des indices d'entraînement pour cette époque.
        perm = indices[torch.randperm(n_train)]
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc, _ = _evaluate(model, X_train, y_train)
        val_acc, _ = _evaluate(model, X_val, y_val)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            logger.info(
                "epoch %3d/%d | loss %.4f | train acc %.3f | val acc %.3f%s",
                epoch, args.epochs, avg_loss, train_acc, val_acc,
                "  ← best" if improved else "",
            )

    train_secs = time.time() - t_train_start

    # ─── Best state + éval finale ─────────────────────────────────
    assert best_state is not None, "best_state non défini — aucune époque ?"
    model.load_state_dict(best_state)
    train_acc, _ = _evaluate(model, X_train, y_train)
    val_acc, per_class = _evaluate(model, X_val, y_val)

    logger.info("─" * 60)
    logger.info("Résultat final (best val) — entraînement : %.1fs", train_secs)
    logger.info("  train accuracy : %.3f", train_acc)
    logger.info("  val accuracy   : %.3f", val_acc)
    logger.info("  val par classe :")
    for label in LABELS:
        n_class = sum(1 for ex in val_ex if ex.label == label)
        logger.info(
            "    %-13s : %.3f  (%d ex)", label, per_class.get(label, 0.0), n_class
        )

    # ─── Sauvegarde ───────────────────────────────────────────────
    metadata = CheckpointMetadata(
        input_dim=EMBED_DIM,
        hidden_dim=args.hidden,
        num_classes=len(LABELS),
        labels=LABELS,
        embed_model_name=EMBED_MODEL_NAME,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        epochs=args.epochs,
        seed=args.seed,
    )
    out_path = Path(args.output)
    save_checkpoint(model, out_path, metadata)
    size_kb = out_path.stat().st_size / 1024
    logger.info("Checkpoint : %s (%.1f KB)", out_path, size_kb)

    # ─── Verdict ──────────────────────────────────────────────────
    if val_acc < args.min_accuracy:
        logger.error(
            "ÉCHEC : val_accuracy %.3f < min_accuracy %.3f. "
            "Vérifie hyperparamètres (lr/epochs/hidden) ou qualité du dataset.",
            val_acc, args.min_accuracy,
        )
        return 1

    logger.info(
        "OK : val_accuracy %.3f >= min_accuracy %.3f", val_acc, args.min_accuracy
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-accuracy", type=float, default=0.65,
        help="Échec si val_accuracy en dessous (défaut 0.65, garde-fou CI)",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
