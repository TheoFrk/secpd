"""Evaluation im Kreditrisiko-Vokabular.

* ROC-AUC (Ranking), PR-AUC/Average Precision (wichtiger bei seltenen
  Positiven), Brier-Score (Kalibrierung des Wahrscheinlichkeitsniveaus).
* Brier Skill gegenüber konstanter Basisrate, Top-k-Capture/Lift.
* CIK-geclusterter Bootstrap für Konfidenzintervalle bei Firmen-Wiederholung.
* Dezil-/Lift-Tabelle: Ereignisrate je Score-Dezil, Lift gegenüber der
  Basisrate, kumulierte Capture-Rate — die Darstellung, in der Kredit-
  risiko-Teams Trennschärfe üblicherweise diskutieren.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def brier_skill_score(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    reference: float | None = None,
) -> float:
    """1 − Brier(Modell) / Brier(konstante Referenz).

    Default-Referenz = empirische Basisrate von ``y_true`` (Nullmodell).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    ref = float(y.mean()) if reference is None else float(reference)
    brier_model = float(brier_score_loss(y, p))
    brier_ref = float(brier_score_loss(y, np.full(len(y), ref)))
    if brier_ref <= 0:
        return float("nan")
    return 1.0 - brier_model / brier_ref


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
    out["brier_skill"] = (
        brier_skill_score(y, p, reference=out["base_rate"]) if len(y) else float("nan")
    )
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        logger.warning("Nur eine Klasse im Testset — AUC-Metriken nicht definiert.")
    return out


def top_k_capture(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    fractions: Sequence[float] = (0.1, 0.3),
) -> dict[str, float]:
    """Capture/Lift in den Top-``fraction`` der Scores (höchste zuerst).

    Bei konstanten Scores (Nullmodell) ist Ranking undefiniert — dann
    erwarteter Zufalls-Capture (= fraction) und Lift 1.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    total = int(y.sum())
    base = float(y.mean()) if n else float("nan")
    constant = n > 0 and np.unique(p).size == 1
    order = np.argsort(-p, kind="mergesort")
    out: dict[str, float] = {}
    for frac in fractions:
        k = max(1, int(round(frac * n))) if n else 0
        key = f"top_{int(round(100 * frac))}pct"
        out[f"{key}_n"] = float(k)
        if constant:
            out[f"{key}_events"] = float("nan")
            out[f"{key}_capture"] = float(frac) if total else float("nan")
            out[f"{key}_precision"] = base
            out[f"{key}_lift"] = 1.0 if base == base else float("nan")
            continue
        hit = int(y[order[:k]].sum()) if k else 0
        rate = hit / k if k else float("nan")
        out[f"{key}_events"] = float(hit)
        out[f"{key}_capture"] = hit / total if total else float("nan")
        out[f"{key}_precision"] = rate
        out[f"{key}_lift"] = rate / base if base else float("nan")
    return out


def reliability_table(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Einfache Kalibrations-Bins (gleich breite Score-Intervalle)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return pd.DataFrame(columns=["bin", "n", "avg_p", "event_rate"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rechter Rand inklusiv
    bins = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b + 1,
                "p_lo": float(edges[b]),
                "p_hi": float(edges[b + 1]),
                "n": int(mask.sum()),
                "avg_p": float(p[mask].mean()),
                "event_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


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


def _safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


DEFAULT_BOOTSTRAP_METRICS: dict[str, MetricFn] = {
    "roc_auc": _safe_roc,
    "pr_auc": _safe_pr,
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "brier_skill": lambda y, p: brier_skill_score(y, p),
}


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray | pd.Series,
    *,
    n_boot: int = 1_000,
    seed: int = 42,
    alpha: float = 0.05,
    metrics: dict[str, MetricFn] | None = None,
) -> dict[str, dict[str, float]]:
    """Cluster-Bootstrap über ``groups`` (typisch CIK).

    Resampled Gruppen mit Zurücklegen; alle Zeilen einer gezogenen Gruppe
    kommen mit. Liefert Punktwert + Perzentil-CI je Metrik.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(groups)
    if len(y) != len(p) or len(y) != len(g):
        raise ValueError("y_true, p und groups müssen gleiche Länge haben.")
    metric_fns = dict(DEFAULT_BOOTSTRAP_METRICS if metrics is None else metrics)
    unique = pd.unique(g)
    rng = np.random.default_rng(seed)

    point = {name: fn(y, p) for name, fn in metric_fns.items()}
    samples: dict[str, list[float]] = {name: [] for name in metric_fns}

    # Index je Gruppe einmal vorbereiten
    group_to_idx = {u: np.flatnonzero(g == u) for u in unique}

    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_to_idx[u] for u in drawn])
        y_b, p_b = y[idx], p[idx]
        for name, fn in metric_fns.items():
            samples[name].append(fn(y_b, p_b))

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for name, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        out[name] = {
            "point": float(point[name]),
            "ci_low": float(np.nanpercentile(arr, lo_q)),
            "ci_high": float(np.nanpercentile(arr, hi_q)),
            "n_boot": float(n_boot),
        }
    return out


def cluster_bootstrap_delta(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray | pd.Series,
    *,
    n_boot: int = 1_000,
    seed: int = 42,
    alpha: float = 0.05,
    metric: str = "roc_auc",
) -> dict[str, Any]:
    """CI für Metrik(A) − Metrik(B) mit gleichem Cluster-Bootstrap."""
    fns = DEFAULT_BOOTSTRAP_METRICS
    if metric not in fns:
        raise KeyError(f"Unbekannte Metrik: {metric}")
    fn = fns[metric]
    y = np.asarray(y_true).astype(int)
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    g = np.asarray(groups)
    unique = pd.unique(g)
    rng = np.random.default_rng(seed)
    group_to_idx = {u: np.flatnonzero(g == u) for u in unique}
    point = fn(y, a) - fn(y, b)
    deltas: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_to_idx[u] for u in drawn])
        deltas.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    arr = np.asarray(deltas, dtype=float)
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": float(np.nanpercentile(arr, lo_q)),
        "ci_high": float(np.nanpercentile(arr, hi_q)),
        "n_boot": float(n_boot),
    }


def firm_overlap_stats(
    train_ciks: Iterable[Any],
    test_ciks: Iterable[Any],
) -> dict[str, float]:
    """Anteil Testfirmen, die auch im Training vorkommen."""
    tr = {int(c) for c in train_ciks if pd.notna(c)}
    te = {int(c) for c in test_ciks if pd.notna(c)}
    overlap = tr & te
    return {
        "n_train_firms": float(len(tr)),
        "n_test_firms": float(len(te)),
        "n_overlap_firms": float(len(overlap)),
        "overlap_rate": (len(overlap) / len(te)) if te else float("nan"),
        "n_new_test_firms": float(len(te - tr)),
    }


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    skill = metrics.get("brier_skill", float("nan"))
    skill_s = f"{skill:+.3f}" if skill == skill else "n/a"
    return (
        f"{name:<22} ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier']:.4f}  Skill={skill_s}  "
        f"(n={int(metrics['n'])}, Positive={int(metrics['positives'])}, "
        f"Basisrate={metrics['base_rate']:.2%})"
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
