"""Evaluation im Kreditrisiko-Vokabular.

* ROC-AUC (Ranking), PR-AUC/Average Precision (wichtiger bei seltenen
  Positiven), Brier-Score (Kalibrierung des Wahrscheinlichkeitsniveaus).
* Dezil-/Lift-Tabelle: Ereignisrate je Score-Dezil, Lift gegenüber der
  Basisrate, kumulierte Capture-Rate — die Darstellung, in der Kredit-
  risiko-Teams Trennschärfe üblicherweise diskutieren.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_probs(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Kernmetriken; robust gegen Ein-Klassen-Testsets."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    out: dict[str, float] = {
        "n": float(len(y)),
        "positives": float(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        logger.warning("Nur eine Klasse im Testset — AUC-Metriken nicht definiert.")
    return out


def decile_table(y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Ereignisraten je Score-Dezil (Dezil 1 = höchste Scores)."""
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p, dtype=float)})
    # Rank-basierte Bins ⇒ immer n_bins gleich große Gruppen, auch bei Ties.
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    frame["decile"] = n_bins - frame["bin"]  # 1 = Top-Scores

    base_rate = frame["y"].mean() if len(frame) else np.nan
    grouped = (
        frame.groupby("decile")
        .agg(n=("y", "size"), events=("y", "sum"), avg_score=("p", "mean"))
        .sort_index()
    )
    grouped["event_rate"] = grouped["events"] / grouped["n"]
    grouped["lift"] = grouped["event_rate"] / base_rate if base_rate else np.nan
    total_events = grouped["events"].sum()
    grouped["cum_capture"] = (
        grouped["events"].cumsum() / total_events if total_events else np.nan
    )
    return grouped.reset_index()


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:<22} ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier']:.4f}  (n={int(metrics['n'])}, "
        f"Positive={int(metrics['positives'])}, Basisrate={metrics['base_rate']:.2%})"
    )


def print_report(results: dict[str, dict[str, Any]], *, show_deciles_for: str | None = None,
                 y_true: np.ndarray | None = None, p: np.ndarray | None = None) -> None:
    """Kompakter Konsolen-Report über alle Modellvarianten."""
    print("\n" + "=" * 78)
    print("EVALUATION (Testset)")
    print("=" * 78)
    for name, metrics in results.items():
        print(format_metrics(name, metrics))
    if show_deciles_for and y_true is not None and p is not None:
        print("-" * 78)
        print(f"Dezil-Tabelle — {show_deciles_for} (Dezil 1 = höchste Scores):")
        table = decile_table(y_true, p)
        with pd.option_context("display.float_format", "{:,.4f}".format):
            print(table.to_string(index=False))
    print("=" * 78 + "\n")
