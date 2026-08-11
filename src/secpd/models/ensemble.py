"""Option B: Aggregation zweier Wahrscheinlichkeits-Scores (Finanz + Text).

Standard ist die gewichtete Mittelung im **Logit-Raum** — probabilistisch
sauberer als das arithmetische Mittel, weil Extremwerte nicht künstlich zur
Mitte gezogen werden. Das arithmetische Mittel bleibt als einfache,
erklärbare Alternative verfügbar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class EnsembleWeights:
    """Gewichte des Zwei-Komponenten-Ensembles (werden normalisiert)."""

    w_financial: float = 0.6
    w_text: float = 0.4

    def normalized(self) -> tuple[float, float]:
        total = self.w_financial + self.w_text
        if total <= 0:
            raise ValueError("Gewichtssumme muss > 0 sein.")
        return self.w_financial / total, self.w_text / total


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def combine_probabilities(
    p_financial: np.ndarray,
    p_text: np.ndarray,
    weights: EnsembleWeights = EnsembleWeights(),
    *,
    method: Literal["logit", "mean"] = "logit",
) -> np.ndarray:
    """Kombiniert zwei Score-Vektoren zu einem Ensemble-Score in [0, 1]."""
    p_fin = np.asarray(p_financial, dtype=float)
    p_txt = np.asarray(p_text, dtype=float)
    if p_fin.shape != p_txt.shape:
        raise ValueError(f"Shape-Mismatch: {p_fin.shape} vs {p_txt.shape}")

    w_fin, w_txt = weights.normalized()
    if method == "mean":
        return w_fin * p_fin + w_txt * p_txt
    if method == "logit":
        return _sigmoid(w_fin * _logit(p_fin) + w_txt * _logit(p_txt))
    raise ValueError(f"Unbekannte Methode: {method!r}")
