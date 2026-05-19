"""Tests d'inférence Brigid + intégration An Dagda (PR C).

Deux niveaux :

1. **Inférence Brigid** : `classify_intent` + `process()` end-to-end.
   Requiert un checkpoint. Si `data/models/brigid_cfc.pt` existe on
   l'utilise (run local après training, ou artefact CI extrait à la
   main) ; sinon on entraîne un mini-modèle dans une fixture
   session-scopée (≈ 20 s sur CPU, MiniLM peut être déjà en cache).

2. **An Dagda + Brigid** : `classify_query` doit utiliser Brigid en
   premier quand registered et confiant ; tomber sur les heuristiques
   sinon. Le shortcut fence markdown garde la priorité absolue.

3. **Régression heuristiques pure** : An Dagda sans Brigid doit
   produire exactement les mêmes routages que la branche `main` avant
   cette PR (covert par `tests/test_dagda.py`, déjà vert).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

# Skip tout le module si torch ou ncps manquent (env minimal).
torch = pytest.importorskip("torch")
pytest.importorskip("ncps")
pytest.importorskip("sentence_transformers")

from core.dagda import AnDagda  # noqa: E402
from core.types import ModuleInput, QueryType  # noqa: E402
from modules.brigid.dataset import LABELS  # noqa: E402
from modules.brigid.model import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    Brigid,
    IntentClassification,
    load_checkpoint,
)


# ─── Fixture : checkpoint utilisable ──────────────────────────────


@pytest.fixture(scope="session")
def brigid_checkpoint(tmp_path_factory) -> Path:
    """Renvoie un checkpoint utilisable.

    - Si `data/models/brigid_cfc.pt` existe (dev local, ou CI qui l'a
      téléchargé en artefact), on le réutilise tel quel.
    - Sinon, on entraîne un mini-modèle (20 époques, seuil bas) dans
      un fichier temporaire. Volontairement court : on teste l'API
      d'inférence, pas la qualité du modèle (la qualité est validée
      par le workflow `brigid-train.yml`).
    """
    if DEFAULT_CHECKPOINT_PATH.exists():
        # Sanity : le checkpoint doit pouvoir se charger, sinon il est
        # obsolète (changement de LABELS, d'embedder…) — on le bypasse.
        try:
            load_checkpoint(DEFAULT_CHECKPOINT_PATH)
            return DEFAULT_CHECKPOINT_PATH
        except (ValueError, FileNotFoundError):
            pass

    # Mini-training pour le test (autonome).
    from scripts.train_brigid import main as train_main  # noqa: PLC0415

    ckpt = tmp_path_factory.mktemp("brigid") / "test_ckpt.pt"
    rc = train_main([
        "--epochs", "20",
        "--seed", "42",
        "--output", str(ckpt),
        "--min-accuracy", "0.5",
        "--log-every", "100",
    ])
    assert rc == 0, "mini-training a échoué (vérifie embeddings + ncps)"
    return ckpt


@pytest.fixture
def brigid(brigid_checkpoint: Path) -> Brigid:
    """Instance Brigid prête à classifier."""
    return Brigid(checkpoint_path=brigid_checkpoint)


# ─── classify_intent (API sync) ──────────────────────────────────


def test_classify_intent_shape(brigid: Brigid):
    """Classifie une query connue et vérifie la forme du résultat."""
    classif = brigid.classify_intent("Salut !")
    assert isinstance(classif, IntentClassification)
    assert classif.label in LABELS
    assert 0.0 <= classif.confidence <= 1.0
    assert set(classif.probabilities.keys()) == set(LABELS)
    assert abs(sum(classif.probabilities.values()) - 1.0) < 1e-5


def test_classify_intent_confidence_matches_max_proba(brigid: Brigid):
    """`confidence` doit être max(probabilities)."""
    classif = brigid.classify_intent("Écris un poème sur la mer.")
    assert classif.confidence == pytest.approx(
        max(classif.probabilities.values()), abs=1e-6
    )


def test_classify_intent_no_checkpoint_returns_none(tmp_path: Path):
    """Sans checkpoint, classify_intent renvoie None (pas de crash)."""
    b = Brigid(checkpoint_path=tmp_path / "nope.pt")
    assert b.classify_intent("anything") is None


# ─── process() (API async, contrat MorriganModule) ───────────────


def test_process_returns_classification(brigid: Brigid):
    """process() délègue à classify_intent et renvoie un ModuleOutput utile."""
    out = asyncio.run(brigid.process(ModuleInput(query="Bonjour Morrigan.")))
    assert out.success
    assert out.result["classification"] in LABELS
    assert "probabilities" in out.result
    assert out.metadata["phase"] == 2
    assert out.metadata["loaded"] is True


def test_process_no_checkpoint_degraded_not_crash(tmp_path: Path):
    """Sans checkpoint, process renvoie un output dégradé (errors non vide)."""
    b = Brigid(checkpoint_path=tmp_path / "nope.pt")
    out = asyncio.run(b.process(ModuleInput(query="anything")))
    assert not out.success
    assert out.confidence == 0.0
    assert out.metadata["loaded"] is False
    assert out.errors  # message d'erreur lisible


def test_health_check_loads_checkpoint(brigid_checkpoint: Path):
    """health_check tente de charger le checkpoint."""
    b = Brigid(checkpoint_path=brigid_checkpoint)
    assert asyncio.run(b.health_check()) is True
    # Et marque le module comme chargé.
    caps = b.get_capabilities()
    assert caps["checkpoint_loaded"] is True
    assert caps["phase"] == 2


# ─── Intégration An Dagda ↔ Brigid ────────────────────────────────


def test_dagda_uses_brigid_when_confident(brigid: Brigid):
    """Une query bien franche doit être routée *via Brigid*."""
    dagda = AnDagda()
    dagda.register_module("brigid", brigid)
    routing = dagda.classify_query("Écris un poème sur l'automne.")
    assert routing.query_type == QueryType.CREATIVE
    assert "Brigid LNN" in routing.reasoning


def test_dagda_falls_back_when_brigid_unsure():
    """Brigid mocké avec faible confidence → fallback heuristiques.

    On mocke pour avoir un contrôle total sur la confidence renvoyée
    (sans dépendre d'une query particulière du modèle réel).
    """
    class WeakBrigid:
        def classify_intent(self, query):
            # Confidence intentionnellement sous le seuil 0.5
            return IntentClassification(
                label="factual",
                confidence=0.3,
                probabilities={lbl: (0.3 if lbl == "factual" else 0.14)
                               for lbl in LABELS},
            )

    dagda = AnDagda()
    dagda.register_module("brigid", WeakBrigid())  # type: ignore[arg-type]
    # "Pourquoi…" matche le mot-clé reasoning → fallback doit le récupérer.
    routing = dagda.classify_query("Pourquoi le ciel est-il bleu ?")
    assert routing.query_type == QueryType.REASONING
    assert "Brigid" not in routing.reasoning  # routage via heuristiques


def test_dagda_falls_back_when_brigid_unavailable():
    """Brigid registered mais sans checkpoint → fallback transparent."""
    class NoCheckpointBrigid:
        def classify_intent(self, query):
            return None

    dagda = AnDagda()
    dagda.register_module("brigid", NoCheckpointBrigid())  # type: ignore[arg-type]
    routing = dagda.classify_query("Compare TCP et UDP.")
    assert routing.query_type == QueryType.REASONING
    assert "Brigid" not in routing.reasoning


def test_dagda_fence_markdown_keeps_priority_over_brigid(brigid: Brigid):
    """Un fence markdown doit court-circuiter Brigid (signal trop fort)."""
    dagda = AnDagda()
    dagda.register_module("brigid", brigid)
    query = "Vérifie :\n```python\nprint('hi')\n```"
    routing = dagda.classify_query(query)
    assert routing.query_type == QueryType.CODE
    assert "fence" in routing.reasoning.lower()


def test_dagda_routing_map_covers_all_query_types():
    """Le mapping interne couvre tous les types Brigid peut renvoyer."""
    brigid_types = {QueryType(lbl) for lbl in LABELS}
    for qt in brigid_types:
        assert qt in AnDagda._ROUTING_MAP, f"{qt} absent de _ROUTING_MAP"
