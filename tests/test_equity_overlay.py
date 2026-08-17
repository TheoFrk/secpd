"""Aktienwert-Overlay: PD → Qualität / z-Score / S1-Hybrid."""
import numpy as np
import pandas as pd

from secpd.equity.overlay import HYBRID_COL, QUALITY_COL, Z_COL, attach_equity_scores, pd_to_equity_quality


def test_pd_to_quality_bounds() -> None:
    q = pd_to_equity_quality([0.0, 0.2, 1.0, np.nan])
    assert list(q.iloc[:3].round(6)) == [1.0, 0.8, 0.0]
    assert np.isnan(q.iloc[3])


def test_low_pd_ranks_higher() -> None:
    df = pd.DataFrame({"pd_score": [0.01, 0.20, 0.50], "name": ["A", "B", "C"]})
    out = attach_equity_scores(df)
    # niedrigste PD → höchster z_secpd_quality
    assert out.loc[out["pd_score"].idxmin(), Z_COL] == out[Z_COL].max()
    assert out.loc[0, QUALITY_COL] > out.loc[2, QUALITY_COL]


def test_hybrid_weights_match_thesis_50_50() -> None:
    df = pd.DataFrame(
        {
            "pd_score": [0.1, 0.3],
            "z_s1_price_score": [2.0, -2.0],
        }
    )
    out = attach_equity_scores(df, weight_s1=0.5, weight_pd=0.5)
    assert HYBRID_COL in out.columns
    # Hybrid liegt zwischen den Komponenten, nicht identisch zu S1 allein
    assert not np.allclose(out[HYBRID_COL], out["z_s1_price_score"])


def test_group_zscore_per_date() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "pd_score": [0.1, 0.3, 0.1, 0.3],
        }
    )
    out = attach_equity_scores(df, group_col="date")
    g1 = out.loc[out["date"] == "2024-01-01", Z_COL]
    assert abs(float(g1.sum())) < 1e-9
