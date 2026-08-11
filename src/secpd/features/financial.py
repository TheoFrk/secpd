"""Tabellarische Finanz-Features: wenige, robuste, PD-nahe Kennzahlen.

Bewusst kompakter Kennzahlenkatalog (Leverage, Liquidität, Profitabilität,
Coverage, Größe — Altman-Z-nahe Bausteine). Fehlende Konzepte werden still
übersprungen; Divisionen sind NaN-sicher (RandomForest + Median-Imputation
kommen damit gut zurecht, ohne Winsorizing-Leakage-Komplexität).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Kanonisches Konzept → Spalten-Aliasse (deckt auch Compustat-Mnemonics ab).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("total_assets", "at", "assets"),
    "total_liabilities": ("total_liabilities", "lt", "liabilities"),
    "current_assets": ("current_assets", "act"),
    "current_liabilities": ("current_liabilities", "lct"),
    "cash": ("cash", "che", "cash_and_equivalents"),
    "long_term_debt": ("long_term_debt", "dltt"),
    "debt_in_current": ("debt_in_current", "dlc"),
    "equity": ("equity", "ceq", "seq", "stockholders_equity", "total_equity"),
    "net_income": ("net_income", "ni", "ib"),
    "revenue": ("revenue", "sale", "sales", "revt"),
    "ebit": ("ebit", "oiadp", "operating_income"),
    "interest_expense": ("interest_expense", "xint"),
    "retained_earnings": ("retained_earnings", "re"),
    "inventory": ("inventory", "invt"),
    "receivables": ("receivables", "rect"),
}


def _resolve(df: pd.DataFrame) -> dict[str, str]:
    """Mappt kanonische Konzepte auf tatsächlich vorhandene Spalten."""
    cols = {c.lower(): c for c in df.columns}
    resolved: dict[str, str] = {}
    for concept, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols:
                resolved[concept] = cols[alias]
                break
    return resolved


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division mit NaN statt inf/Division-by-zero."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_financial_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Berechnet Kennzahlen-Spalten (Prefix ``fin_``) und gibt deren Namen zurück.

    Returns
    -------
    (df, feature_cols):
        DataFrame mit neuen Spalten; Liste der erzeugten Feature-Namen.
    """
    df = df.copy()
    c = _resolve(df)
    feats: dict[str, pd.Series] = {}

    def have(*concepts: str) -> bool:
        return all(k in c for k in concepts)

    if have("total_liabilities", "total_assets"):
        feats["fin_leverage"] = _safe_div(df[c["total_liabilities"]], df[c["total_assets"]])
    if have("long_term_debt", "equity"):
        total_debt = pd.to_numeric(df[c["long_term_debt"]], errors="coerce")
        if "debt_in_current" in c:
            total_debt = total_debt.add(
                pd.to_numeric(df[c["debt_in_current"]], errors="coerce"), fill_value=0
            )
        feats["fin_debt_to_equity"] = _safe_div(total_debt, df[c["equity"]])
    if have("current_assets", "current_liabilities"):
        feats["fin_current_ratio"] = _safe_div(df[c["current_assets"]], df[c["current_liabilities"]])
        if "inventory" in c:
            quick = pd.to_numeric(df[c["current_assets"]], errors="coerce") - pd.to_numeric(
                df[c["inventory"]], errors="coerce"
            )
            feats["fin_quick_ratio"] = _safe_div(quick, df[c["current_liabilities"]])
    if have("cash", "current_liabilities"):
        feats["fin_cash_ratio"] = _safe_div(df[c["cash"]], df[c["current_liabilities"]])
    if have("current_assets", "current_liabilities", "total_assets"):
        wc = pd.to_numeric(df[c["current_assets"]], errors="coerce") - pd.to_numeric(
            df[c["current_liabilities"]], errors="coerce"
        )
        feats["fin_wc_to_assets"] = _safe_div(wc, df[c["total_assets"]])
    if have("net_income", "total_assets"):
        feats["fin_roa"] = _safe_div(df[c["net_income"]], df[c["total_assets"]])
    if have("net_income", "revenue"):
        feats["fin_net_margin"] = _safe_div(df[c["net_income"]], df[c["revenue"]])
    if have("ebit", "interest_expense"):
        feats["fin_interest_coverage"] = _safe_div(df[c["ebit"]], df[c["interest_expense"]])
    if have("retained_earnings", "total_assets"):
        feats["fin_re_to_assets"] = _safe_div(df[c["retained_earnings"]], df[c["total_assets"]])
    if have("revenue", "total_assets"):
        feats["fin_asset_turnover"] = _safe_div(df[c["revenue"]], df[c["total_assets"]])
    if "total_assets" in c:
        ta = pd.to_numeric(df[c["total_assets"]], errors="coerce").clip(lower=0)
        feats["fin_log_assets"] = np.log1p(ta)

    for name, series in feats.items():
        df[name] = series.astype(float)

    feature_cols = sorted(feats.keys())
    if not feature_cols:
        logger.warning(
            "Keine Finanz-Features ableitbar — Spalten prüfen (kanonisches Schema "
            "siehe data/edgar.py TAG_MAP)."
        )
    else:
        logger.info("Finanz-Features erzeugt (%d): %s", len(feature_cols), ", ".join(feature_cols))
    return df, feature_cols
