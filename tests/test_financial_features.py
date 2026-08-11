"""Finanz-Features: Kennzahlenwerte, NaN-Sicherheit, Alias-Auflösung."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.features.financial import add_financial_features


def test_ratios_and_safe_division() -> None:
    df = pd.DataFrame(
        {
            "total_assets": [100.0, 200.0],
            "total_liabilities": [50.0, 150.0],
            "current_assets": [40.0, 60.0],
            "current_liabilities": [20.0, 0.0],  # 0 ⇒ NaN statt inf
            "net_income": [10.0, -5.0],
            "revenue": [80.0, 120.0],
        }
    )
    out, cols = add_financial_features(df)
    assert "fin_leverage" in cols
    assert np.isclose(out.loc[0, "fin_leverage"], 0.5)
    assert np.isclose(out.loc[0, "fin_current_ratio"], 2.0)
    assert np.isnan(out.loc[1, "fin_current_ratio"])  # Division durch 0
    assert np.isclose(out.loc[1, "fin_roa"], -0.025)


def test_compustat_aliases() -> None:
    df = pd.DataFrame({"at": [100.0], "lt": [40.0], "ni": [8.0], "sale": [50.0]})
    out, cols = add_financial_features(df)
    assert np.isclose(out.loc[0, "fin_leverage"], 0.4)
    assert "fin_net_margin" in cols
