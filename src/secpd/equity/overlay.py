"""PD → Aktienwert-Signal (Late-Fusion, analog S2/S4 Hybrid).

Niedrige Ausfallwahrscheinlichkeit = höhere Equity-Qualität. Cross-Section
wird als z-Score von ``-pd`` gebildet, damit Long = geringes Distress-Risiko.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_COL = "equity_quality"
Z_COL = "z_secpd_quality"
HYBRID_COL = "z_hybrid_s1_secpd"


def pd_to_equity_quality(pd_score: pd.Series | np.ndarray) -> pd.Series:
    """``1 − PD``, geclippt auf [0, 1]. Höher = sicherer Equity-Claim."""
    s = pd.to_numeric(pd.Series(pd_score), errors="coerce")
    return (1.0 - s).clip(lower=0.0, upper=1.0)


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=1)
    if sd is None or not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def attach_equity_scores(
    df: pd.DataFrame,
    *,
    pd_col: str = "pd_score",
    s1_col: str | None = "z_s1_price_score",
    weight_s1: float = 0.5,
    weight_pd: float = 0.5,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Hängt Qualitäts- und z-Scores an. Optional Late-Fusion mit S1.

    Parameters
    ----------
    group_col:
        Wenn gesetzt (z. B. ``date``), z-Score je Querschnitt, sonst über alle Zeilen.
    """
    out = df.copy()
    if pd_col not in out.columns:
        raise KeyError(f"Spalte {pd_col!r} fehlt — zuerst PD scoren.")
    out[QUALITY_COL] = pd_to_equity_quality(out[pd_col])
    # Long = niedriges PD = hohe Qualität
    raw = -pd.to_numeric(out[pd_col], errors="coerce")
    if group_col and group_col in out.columns:
        out[Z_COL] = raw.groupby(out[group_col], sort=False).transform(_zscore)
    else:
        out[Z_COL] = _zscore(raw)

    if s1_col and s1_col in out.columns:
        z_s1 = pd.to_numeric(out[s1_col], errors="coerce")
        w1, w2 = float(weight_s1), float(weight_pd)
        tot = w1 + w2
        if tot <= 0:
            raise ValueError("Hybrid-Gewichte müssen > 0 sein.")
        w1, w2 = w1 / tot, w2 / tot
        out[HYBRID_COL] = w1 * z_s1.fillna(0.0) + w2 * out[Z_COL].fillna(0.0)
    return out
