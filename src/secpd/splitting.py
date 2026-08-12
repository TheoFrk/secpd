"""Leakage-bewusste Train/Test-Splits.

Priorität (``strategy="auto"``):

1. **temporal** — letzte Geschäftsjahre als Testfenster (realistischstes
   Deployment-Szenario: Modell aus der Vergangenheit, Anwendung auf Zukunft).
2. **group** — GroupShuffleSplit über die Firma (``cik``): dieselbe Firma
   darf nie gleichzeitig in Train und Test liegen (Firm-Memorization).
3. **random** — stratifiziert; nur als letzter Fallback.

Zusätzlich: ``rolling_origin_splits`` für mehrere zeitliche Cutoffs
(ehrlichere Evaluation bei Firm-Overlap im einzelnen Temporal-Split).

Fallbacks greifen automatisch, wenn ein Split eine Klasse ohne Positive
hinterlassen würde (bei seltenen Labels realistisch).
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)

Strategy = Literal["auto", "temporal", "group", "random"]


def _both_classes(y: pd.Series, idx: np.ndarray) -> bool:
    sub = y.iloc[idx]
    return sub.nunique() >= 2


def smart_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str | None = None,
    year_col: str | None = None,
    test_size: float = 0.2,
    strategy: Strategy = "auto",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Liefert (train_idx, test_idx, verwendete_strategie) als Positions-Indizes."""
    y = df[label_col].astype(int)
    n = len(df)
    all_idx = np.arange(n)

    def temporal() -> tuple[np.ndarray, np.ndarray] | None:
        if year_col is None or year_col not in df.columns:
            return None
        years = pd.to_numeric(df[year_col], errors="coerce")
        counts = years.value_counts().sort_index()
        cum, test_years = 0, []
        for yr in counts.index[::-1]:
            test_years.append(yr)
            cum += counts[yr]
            if cum >= test_size * n:
                break
        mask = years.isin(test_years).to_numpy()
        tr, te = all_idx[~mask], all_idx[mask]
        if len(te) == 0 or len(tr) == 0 or not (_both_classes(y, tr) and _both_classes(y, te)):
            logger.warning("Temporaler Split unbrauchbar (Klassen fehlen) — Fallback.")
            return None
        logger.info("Temporaler Split: Testjahre %s", sorted(int(t) for t in test_years))
        return tr, te

    def grouped() -> tuple[np.ndarray, np.ndarray] | None:
        if group_col is None or group_col not in df.columns:
            return None
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(gss.split(df, y, groups=df[group_col]))
        if not (_both_classes(y, tr) and _both_classes(y, te)):
            logger.warning("Group-Split unbrauchbar (Klassen fehlen) — Fallback.")
            return None
        logger.info("Group-Split über %r (%d Gruppen).", group_col, df[group_col].nunique())
        return tr, te

    def randomized() -> tuple[np.ndarray, np.ndarray]:
        stratify = y if y.nunique() >= 2 else None
        tr, te = train_test_split(
            all_idx, test_size=test_size, random_state=random_state, stratify=stratify
        )
        logger.info("Stratifizierter Random-Split.")
        return tr, te

    order: list[tuple[str, object]] = {
        "temporal": [("temporal", temporal)],
        "group": [("group", grouped)],
        "random": [("random", randomized)],
        "auto": [("temporal", temporal), ("group", grouped), ("random", randomized)],
    }[strategy]

    for name, fn in order:
        result = fn()  # type: ignore[operator]
        if result is not None:
            tr, te = result
            return tr, te, name
    tr, te = randomized()
    return tr, te, "random"


def rolling_origin_splits(
    df: pd.DataFrame,
    *,
    label_col: str,
    year_col: str = "fyear",
    test_window_years: int = 2,
    step_years: int = 2,
    min_train_years: int = 4,
    min_train_positives: int = 5,
    min_test_positives: int = 1,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Mehrere Expanding-Window-Splits: Train ``year ≤ cutoff``, Test danach.

    Testfenster: ``(cutoff, cutoff + test_window_years]``.
    """
    if year_col not in df.columns:
        raise KeyError(f"year_col {year_col!r} fehlt")
    years = pd.to_numeric(df[year_col], errors="coerce")
    y = df[label_col].astype(int)
    uniq = sorted(int(v) for v in years.dropna().unique())
    if len(uniq) < min_train_years + test_window_years:
        logger.warning("Zu wenige Jahre für Rolling-Origin.")
        return []

    # Erster cutoff: nach mindestens min_train_years Kalenderjahren mit Daten
    first_cutoff = uniq[min_train_years - 1]
    last_cutoff = uniq[-1] - test_window_years
    cutoffs = list(range(int(first_cutoff), int(last_cutoff) + 1, step_years))
    out: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    all_idx = np.arange(len(df))
    for cutoff in cutoffs:
        test_lo = cutoff + 1
        test_hi = cutoff + test_window_years
        tr_mask = (years <= cutoff).to_numpy()
        te_mask = ((years >= test_lo) & (years <= test_hi)).to_numpy()
        tr, te = all_idx[tr_mask], all_idx[te_mask]
        if len(tr) == 0 or len(te) == 0:
            continue
        if not (_both_classes(y, tr) and _both_classes(y, te)):
            continue
        n_pos_tr = int(y.iloc[tr].sum())
        n_pos_te = int(y.iloc[te].sum())
        if n_pos_tr < min_train_positives or n_pos_te < min_test_positives:
            continue
        meta = {
            "cutoff": int(cutoff),
            "test_years": [yr for yr in uniq if test_lo <= yr <= test_hi],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "positives_train": n_pos_tr,
            "positives_test": n_pos_te,
        }
        out.append((tr, te, meta))
    logger.info("Rolling-Origin: %d brauchbare Cutoffs", len(out))
    return out
