"""Human-readable labels for the stored PyHFO classifications."""

from __future__ import annotations

from typing import Any


def model_score_labels(event: dict[str, Any]) -> list[str]:
    scores = event.get("scores") or {}
    labels = event.get("labels") or {}
    parts = []
    for stage, model, negative, positive, target in (
        ("artifact", "Artifact", "artifact", "non-artifact", "real-HFO"),
        ("spike", "Spike", "no spike", "spike", "spike"),
        ("ehfo", "eHFO", "no eHFO", "eHFO", "eHFO"),
    ):
        if scores.get(stage) is None:
            continue
        probability = f"{target} p {scores[stage]:.2f}"
        classification = {0: negative, 1: positive}.get(labels.get(stage))
        value = f"{classification} ({probability})" if classification else probability
        parts.append(f"{model} model: {value}")
    return parts
