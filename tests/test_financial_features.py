"""Finanz-Features: Kennzahlenwerte, NaN-Sicherheit, Alias-Auflösung, Trends."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.data.edgar import annual_financials_from_facts
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
    assert "fin_miss_current_ratio" in cols
    assert out.loc[1, "fin_miss_current_ratio"] == 1.0


def test_compustat_aliases() -> None:
    df = pd.DataFrame({"at": [100.0], "lt": [40.0], "ni": [8.0], "sale": [50.0]})
    out, cols = add_financial_features(df)
    assert np.isclose(out.loc[0, "fin_leverage"], 0.4)
    assert "fin_net_margin" in cols


def test_trends_and_sic_rank() -> None:
    df = pd.DataFrame(
        {
            "cik": [1, 1, 1, 2, 2, 2],
            "fyear": [2018, 2019, 2020, 2018, 2019, 2020],
            "sic": [7372, 7372, 7372, 7372, 7372, 7372],
            "total_assets": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
            "total_liabilities": [40.0, 55.0, 70.0, 80.0, 90.0, 100.0],
            "current_assets": [50.0, 50.0, 50.0, 90.0, 90.0, 90.0],
            "current_liabilities": [20.0, 25.0, 30.0, 40.0, 45.0, 50.0],
            "net_income": [10.0, 8.0, 5.0, 20.0, 15.0, 10.0],
            "revenue": [80.0, 90.0, 100.0, 150.0, 160.0, 170.0],
            "retained_earnings": [30.0, 35.0, 38.0, 60.0, 70.0, 75.0],
            "equity": [60.0, 55.0, 50.0, 120.0, 120.0, 120.0],
            "ebit": [12.0, 10.0, 7.0, 25.0, 20.0, 15.0],
        }
    )
    out, cols = add_financial_features(df)
    assert "fin_d_leverage" in cols
    assert "fin_vol3_roa" in cols
    assert "fin_altman_z" in cols
    assert "fin_leverage_sic_pct" in cols
    # Erstes Jahr: kein Δ
    assert np.isnan(out.loc[0, "fin_d_leverage"])
    # Leverage steigt bei CIK 1: 0.4 → 0.5 → ~0.583
    assert out.loc[1, "fin_d_leverage"] > 0


def test_pit_prefers_earlier_filed() -> None:
    facts = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2019,
                                "fp": "FY",
                                "form": "10-K/A",
                                "filed": "2021-06-01",
                                "val": 999.0,
                            },
                            {
                                "fy": 2019,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2020-03-01",
                                "val": 100.0,
                            },
                        ]
                    }
                }
            }
        },
    }
    df = annual_financials_from_facts(facts)
    assert len(df) == 1
    assert float(df.loc[0, "total_assets"]) == 100.0
