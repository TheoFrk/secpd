"""Tabellarische Finanz-Features: PD-nahe Kennzahlen inkl. Trends und Industrie.

Bewusst kompakter Katalog (Leverage, Liquidität, Profitabilität, Coverage,
Größe — Altman-/Ohlson-nahe Bausteine), ergänzt um:

* Jahresveränderungen und 3-Jahres-Volatilität (pro CIK)
* Missing-Indikatoren (Median-Imputation sonst unsichtbar)
* Branchen-Perzentile über SIC-2 × Jahr
* Altman-Z-Komposit

Fehlende Konzepte werden still übersprungen; Divisionen sind NaN-sicher.
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

_TREND_BASES: tuple[str, ...] = (
    "fin_leverage",
    "fin_roa",
    "fin_debt_to_equity",
    "fin_current_ratio",
    "fin_log_assets",
    "fin_asset_turnover",
    "fin_net_margin",
)

_SIC_RANK_BASES: tuple[str, ...] = (
    "fin_leverage",
    "fin_roa",
    "fin_current_ratio",
    "fin_wc_to_assets",
)


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


def _base_ratios(df: pd.DataFrame, c: dict[str, str]) -> dict[str, pd.Series]:
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

    # Umsatzwachstum (YoY log)
    if have("revenue") and {"cik", "fyear"}.issubset(df.columns):
        tmp = df[["cik", "fyear"]].copy()
        tmp["_rev"] = pd.to_numeric(df[c["revenue"]], errors="coerce")
        tmp = tmp.sort_values(["cik", "fyear"])
        prev = tmp.groupby("cik", sort=False)["_rev"].shift(1)
        delta = np.log1p(tmp["_rev"].clip(lower=0)) - np.log1p(prev.clip(lower=0))
        feats["fin_d_log_revenue"] = pd.Series(delta.to_numpy(), index=tmp.index).reindex(
            df.index
        )

    # Altman Z (klassische Gewichte; Market Equity ≈ Book Equity)
    if have("current_assets", "current_liabilities", "total_assets") and have(
        "retained_earnings", "total_assets"
    ):
        wc_ta = feats.get("fin_wc_to_assets")
        re_ta = feats.get("fin_re_to_assets")
        ebit_ta = (
            _safe_div(df[c["ebit"]], df[c["total_assets"]])
            if "ebit" in c
            else pd.Series(np.nan, index=df.index)
        )
        eq_tl = (
            _safe_div(df[c["equity"]], df[c["total_liabilities"]])
            if have("equity", "total_liabilities")
            else pd.Series(np.nan, index=df.index)
        )
        sale_ta = feats.get("fin_asset_turnover")
        if wc_ta is not None and re_ta is not None and sale_ta is not None:
            feats["fin_altman_z"] = (
                1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta + 0.6 * eq_tl + 1.0 * sale_ta
            )

    return feats


def _add_missing_indicators(feats: dict[str, pd.Series]) -> dict[str, pd.Series]:
    extra: dict[str, pd.Series] = {}
    for name, series in list(feats.items()):
        if name.startswith("fin_miss_"):
            continue
        extra[f"fin_miss_{name.removeprefix('fin_')}"] = series.isna().astype(float)
    feats.update(extra)
    return feats


def _add_trends(df: pd.DataFrame, feats: dict[str, pd.Series]) -> dict[str, pd.Series]:
    if not {"cik", "fyear"}.issubset(df.columns):
        return feats
    panel = df[["cik", "fyear"]].copy()
    for base in _TREND_BASES:
        if base not in feats:
            continue
        panel[base] = feats[base]
    panel = panel.sort_values(["cik", "fyear"])
    g = panel.groupby("cik", sort=False)
    for base in _TREND_BASES:
        if base not in panel.columns:
            continue
        short = base.removeprefix("fin_")
        delta = g[base].diff()
        vol = g[base].transform(lambda s: s.rolling(3, min_periods=2).std())
        feats[f"fin_d_{short}"] = delta.reindex(df.index)
        feats[f"fin_vol3_{short}"] = vol.reindex(df.index)
    return feats


def _add_sic_ranks(df: pd.DataFrame, feats: dict[str, pd.Series]) -> dict[str, pd.Series]:
    if "sic" not in df.columns or "fyear" not in df.columns:
        return feats
    sic = pd.to_numeric(df["sic"], errors="coerce")
    # 4-stellige SIC → 2-Steller; 3-Steller bleiben als // 10
    sic2 = np.where(sic >= 1000, (sic // 100).astype(float), (sic // 10).astype(float))
    year = pd.to_numeric(df["fyear"], errors="coerce")
    keys = pd.DataFrame({"sic2": sic2, "fyear": year}, index=df.index)
    for base in _SIC_RANK_BASES:
        if base not in feats:
            continue
        tmp = keys.copy()
        tmp["v"] = feats[base].values
        # mind. 5 Beobachtungen je Zelle, sonst NaN (kleine Branchen nicht erzwingen)
        cnt = tmp.groupby(["sic2", "fyear"])["v"].transform("count")
        pct = tmp.groupby(["sic2", "fyear"])["v"].rank(pct=True, method="average")
        pct = pct.where(cnt >= 5)
        feats[f"{base}_sic_pct"] = pct
    return feats


def add_financial_features(
    df: pd.DataFrame,
    *,
    include_trends: bool = True,
    include_missing: bool = True,
    include_industry: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Berechnet Kennzahlen-Spalten (Prefix ``fin_``) und gibt deren Namen zurück.

    Returns
    -------
    (df, feature_cols):
        DataFrame mit neuen Spalten; Liste der erzeugten Feature-Namen.
    """
    df = df.copy()
    c = _resolve(df)
    feats = _base_ratios(df, c)

    if include_trends:
        feats = _add_trends(df, feats)
    if include_industry:
        feats = _add_sic_ranks(df, feats)
    if include_missing:
        feats = _add_missing_indicators(feats)

    for name, series in feats.items():
        df[name] = pd.to_numeric(series, errors="coerce").astype(float)

    # Trainings-Features: alle fin_* außer Hilfsspalten
    feature_cols = sorted(k for k in feats if k.startswith("fin_"))
    if not feature_cols:
        logger.warning(
            "Keine Finanz-Features ableitbar — Spalten prüfen (kanonisches Schema "
            "siehe data/edgar.py TAG_MAP)."
        )
    else:
        n_base = sum(1 for k in feature_cols if not any(
            k.startswith(p) for p in ("fin_miss_", "fin_d_", "fin_vol3_")
        ) and not k.endswith("_sic_pct"))
        logger.info(
            "Finanz-Features erzeugt (%d total, ~%d Level): %s%s",
            len(feature_cols),
            n_base,
            ", ".join(feature_cols[:12]),
            "…" if len(feature_cols) > 12 else "",
        )
    return df, feature_cols
