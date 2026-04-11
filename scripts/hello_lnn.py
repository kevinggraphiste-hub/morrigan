"""
BRIGID — Prototype de classification d'intention via LNN.

Demontre qu'un reseau Liquid Neural Network (CfC) avec ~1000 parametres
peut apprendre a classifier des intentions a partir d'embeddings.

Phase 0 : preuve de concept avec donnees synthetiques.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import numpy as np
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# === Configuration ===
EMBEDDING_DIM = 32      # Dimension des embeddings (simplifie pour le proto)
NUM_CLASSES = 4         # factual, creative, reasoning, conversation
HIDDEN_SIZE = 16        # Neurones dans le reseau LNN
EPOCHS = 100
LEARNING_RATE = 0.01


def generate_synthetic_data(n_samples: int = 200):
    """
    Genere des donnees synthetiques pour 4 classes d'intention.

    En Phase 2, ces embeddings viendront de sentence-transformers.
    Ici on simule des clusters dans l'espace vectoriel.
    """
    data = []
    labels = []

    for i in range(n_samples):
        class_id = i % NUM_CLASSES
        # Chaque classe a un centroide different dans l'espace
        centroid = torch.zeros(EMBEDDING_DIM)
        centroid[class_id * 8:(class_id + 1) * 8] = 1.0
        # Bruit gaussien autour du centroide
        sample = centroid + torch.randn(EMBEDDING_DIM) * 0.3
        data.append(sample)
        labels.append(class_id)

    return torch.stack(data), torch.tensor(labels, dtype=torch.long)


def main():
    print("=" * 60)
    print("  BRIGID — Prototype LNN pour Classification d'Intention")
    print("=" * 60)
    print()

    # 1. Donnees
    X, y = generate_synthetic_data(400)
    # Reshape pour LNN: (batch, seq_len=1, features)
    X = X.unsqueeze(1)

    # Split train/test
    split = 320
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Modele LNN
    wiring = AutoNCP(HIDDEN_SIZE, NUM_CLASSES)
    model = CfC(EMBEDDING_DIM, wiring)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: CfC (Closed-form Continuous-time)")
    print(f"Neurones:     {HIDDEN_SIZE}")
    print(f"Parametres:   {n_params:,}")
    print(f"Classes:      {NUM_CLASSES} (factual, creative, reasoning, conversation)")
    print(f"Train/Test:   {len(X_train)}/{len(X_test)}")
    print()

    # 3. Entrainement
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Entrainement...")
    for epoch in range(EPOCHS):
        model.train()
        output, _ = model(X_train)
        logits = output[:, -1, :]  # Prendre le dernier timestep
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_out, _ = model(X_test)
                test_logits = test_out[:, -1, :]
                preds = test_logits.argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
                print(f"  Epoch {epoch+1:3d}/{EPOCHS} — Loss: {loss.item():.4f} — Test Acc: {acc:.1%}")

    # 4. Resultat final
    model.eval()
    with torch.no_grad():
        test_out, _ = model(X_test)
        test_logits = test_out[:, -1, :]
        preds = test_logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()

    print()
    print(f"=== Resultat Final ===")
    print(f"Precision: {acc:.1%}")
    print(f"Avec seulement {n_params:,} parametres")
    print()

    class_names = ["factual", "creative", "reasoning", "conversation"]
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == y_test[mask]).float().mean().item()
            print(f"  {name:15s}: {class_acc:.1%}")

    print()
    print("Brigid est prete a apprendre.")


if __name__ == "__main__":
    main()
