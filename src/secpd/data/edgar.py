"""EDGAR-Zugriff (nur Home-Setup mit Internet; Ergebnisse werden committet).

Da der Zenodo-Datensatz **keine Finanzkennzahlen** enthält, kommen die
tabellarischen Features aus der SEC-XBRL-API ``companyfacts`` — strukturiert,
kostenlos und über den CIK direkt joinbar:

    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

Die SEC verlangt einen deklarierten User-Agent („Name Kontakt@mail") und
faires Rate-Limiting (< 10 req/s; hier konservativ gedrosselt).
Abdeckung: XBRL-Companyfacts sind erst ab ca. GJ 2009 flächig verfügbar —
Firm-Years ohne Assets sind in der UI nicht scorebar (keine Pseudo-PD).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

#: Kanonisches Schema → Kandidaten-Tags (us-gaap), Reihenfolge = Priorität.
TAG_MAP: dict[str, tuple[str, ...]] = {
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "ebit": ("OperatingIncomeLoss",),
    "interest_expense": ("InterestExpense",),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "inventory": ("InventoryNet",),
    "receivables": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
}

_ANNUAL_FORMS = ("10-K", "10-K/A")
_INTERIM_FORMS = ("10-Q", "10-Q/A")
_PANEL_COLUMNS = ["cik", "fyear", "source_form", "reporting_date", "filing_date", *TAG_MAP.keys()]


def _parse_cik(raw: Any) -> int:
    try:
        return int(str(raw).strip() or 0)
    except (TypeError, ValueError):
        return 0


def _empty_panel(cik: int = 0) -> pd.DataFrame:
    out = pd.DataFrame(columns=_PANEL_COLUMNS)
    if cik:
        out["cik"] = pd.Series(dtype="int64")
    return out


def _usd_entries(gaap: dict[str, Any], tags: tuple[str, ...], *, forms: tuple[str, ...]) -> list[dict[str, Any]]:
    formset = set(forms)
    for tag in tags:
        units = (gaap.get(tag) or {}).get("units", {})
        entries = [e for e in (units.get("USD") or []) if e.get("form") in formset]
        if entries:
            return entries
    return []


def annual_financials_from_facts(
    facts: dict[str, Any],
    *,
    allow_interim: bool = True,
) -> pd.DataFrame:
    """Extrahiert Jahreswerte (Form 10-K, fp=FY) ins kanonische Schema.

    Point-in-time: je ``(fyear, Konzept)`` zählt der **zuerst filed**-Wert
    (frühestes ``filed``-Datum). Spätere 10-K/A-Restatements überschreiben
    nicht — Look-ahead durch nachträgliche Korrekturen wird vermieden.

    Ohne 10-K (Neu-Listing): optional 10-Q-Perioden nach ``end``-Datum —
    Bilanz instant, GuV längste Duration bis zu diesem Stichtag (oft YTD).

    Rückgabe: eine Zeile je Periode mit den Spalten aus :data:`TAG_MAP`.
    """
    cik = _parse_cik(facts.get("cik"))
    gaap = facts.get("facts", {}).get("us-gaap", {})
    annual = _extract_10k_fy(gaap)
    if not annual.empty:
        return _finalize_panel(annual, cik, source_form="10-K", entity_name=facts.get("entityName"))
    if allow_interim:
        interim = _extract_10q_periods(gaap)
        if not interim.empty:
            return _finalize_panel(interim, cik, source_form="10-Q", entity_name=facts.get("entityName"))
    return _empty_panel(cik)


def _finalize_panel(
    df: pd.DataFrame,
    cik: int,
    *,
    source_form: str,
    entity_name: Any = None,
) -> pd.DataFrame:
    out = df.copy()
    out["cik"] = cik
    out["source_form"] = source_form
    name = str(entity_name or "").strip()
    out["entity_name"] = name if name else pd.NA
    for col in _PANEL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    cols = list(_PANEL_COLUMNS)
    if "entity_name" not in cols:
        cols.append("entity_name")
    return out[cols]


def _extract_10k_fy(gaap: dict[str, Any]) -> pd.DataFrame:
    # fyear → concept → (value, filed_ts, period_end)
    per_year: dict[int, dict[str, tuple[float, pd.Timestamp, pd.Timestamp]]] = {}

    for canonical, tags in TAG_MAP.items():
        found_any = False
        for e in _usd_entries(gaap, tags, forms=_ANNUAL_FORMS):
            if e.get("fp") != "FY":
                continue
            fy = e.get("fy")
            val = e.get("val")
            if fy is None or val is None:
                continue
            filed = pd.to_datetime(e.get("filed"), errors="coerce")
            end = pd.to_datetime(e.get("end"), errors="coerce")
            row = per_year.setdefault(int(fy), {})
            prev = row.get(canonical)
            if prev is None:
                row[canonical] = (float(val), filed, end)
                found_any = True
                continue
            _, prev_filed, _ = prev
            if pd.isna(prev_filed) and pd.notna(filed):
                row[canonical] = (float(val), filed, end)
            elif pd.notna(filed) and pd.notna(prev_filed) and filed < prev_filed:
                row[canonical] = (float(val), filed, end)
            found_any = True
        if found_any:
            continue  # erster Tag mit Treffern (über _usd_entries)

    if not per_year:
        return pd.DataFrame()
    rows = []
    for fy, vals in sorted(per_year.items()):
        row: dict[str, Any] = {"fyear": fy}
        filed_candidates = []
        end_candidates = []
        for concept in TAG_MAP:
            packed = vals.get(concept)
            if packed is None:
                continue
            value, filed, end = packed
            row[concept] = value
            if pd.notna(filed):
                filed_candidates.append(filed)
            if pd.notna(end):
                end_candidates.append(end)
        row["filing_date"] = min(filed_candidates) if filed_candidates else pd.NaT
        if end_candidates:
            row["reporting_date"] = max(end_candidates)
        else:
            row["reporting_date"] = pd.Timestamp(f"{fy}-12-31")
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_10q_periods(gaap: dict[str, Any]) -> pd.DataFrame:
    """Eine Zeile je Bilanzstichtag (``end``) aus 10-Q-Fakten."""
    # period_end → concept → (value, filed, duration_days)
    per_end: dict[pd.Timestamp, dict[str, tuple[float, pd.Timestamp, float]]] = {}

    for canonical, tags in TAG_MAP.items():
        for e in _usd_entries(gaap, tags, forms=_INTERIM_FORMS):
            val = e.get("val")
            end = pd.to_datetime(e.get("end"), errors="coerce")
            if val is None or pd.isna(end):
                continue
            filed = pd.to_datetime(e.get("filed"), errors="coerce")
            start = pd.to_datetime(e.get("start"), errors="coerce")
            if pd.isna(start):
                duration = 0.0  # Instant (Bilanz)
            else:
                duration = float((end - start).days)
            end_key = pd.Timestamp(end.normalize())
            row = per_end.setdefault(end_key, {})
            prev = row.get(canonical)
            if prev is None:
                row[canonical] = (float(val), filed, duration)
                continue
            _, prev_filed, prev_dur = prev
            # Instant: frühestes Filing. Duration: längstes Fenster (YTD > Quartal).
            better = False
            if duration == 0.0 and prev_dur == 0.0:
                better = pd.notna(filed) and (pd.isna(prev_filed) or filed < prev_filed)
            elif duration > prev_dur:
                better = True
            elif duration == prev_dur and pd.notna(filed) and (pd.isna(prev_filed) or filed < prev_filed):
                better = True
            if better:
                row[canonical] = (float(val), filed, duration)

    if not per_end:
        return pd.DataFrame()
    rows = []
    for end, vals in sorted(per_end.items()):
        row: dict[str, Any] = {
            "fyear": int(end.year),
            "reporting_date": end,
        }
        filed_candidates = []
        for concept in TAG_MAP:
            packed = vals.get(concept)
            if packed is None:
                continue
            value, filed, _dur = packed
            row[concept] = value
            if pd.notna(filed):
                filed_candidates.append(filed)
        row["filing_date"] = min(filed_candidates) if filed_candidates else pd.NaT
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "total_assets" in out.columns:
        out = out.loc[out["total_assets"].notna()].copy()
    return out


def fetch_company_facts(
    cik: int,
    *,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    cache_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Lädt das companyfacts-JSON eines CIK (wirft bei HTTP-Fehlern).

    Optionaler Datei-Cache (wie Submissions): Lauf ist damit unterbrechbar.
    """
    if not user_agent:
        raise ValueError(
            "SEC verlangt einen User-Agent — SECPD_SEC_UA setzen, "
            'z. B. "Commerzbank Praktikum vorname.nachname@example.com".'
        )
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"CIK{int(cik):010d}.json"
        if cache_file.exists() and not force:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    sess = session or requests.Session()
    resp = sess.get(
        COMPANYFACTS_URL.format(cik=int(cik)),
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_financials_panel(
    ciks: Iterable[int],
    *,
    user_agent: str,
    sleep_s: float = 0.15,
    session: requests.Session | None = None,
    cache_dir: Path | str | None = "data/raw/edgar_companyfacts",
    force: bool = False,
) -> pd.DataFrame:
    """Lädt Companyfacts für mehrere CIKs und stapelt sie zu einem Panel.

    Fehlertolerant: einzelne 404/Fehler werden geloggt und übersprungen.
    Cache-Treffer ohne Netzpause, damit ein Restart nur die fehlenden CIKs holt.
    """
    sess = session or requests.Session()
    frames: list[pd.DataFrame] = []
    ciks = sorted({int(c) for c in ciks})
    cache_root = Path(cache_dir) if cache_dir else None
    for i, cik in enumerate(ciks, start=1):
        cached = bool(
            cache_root
            and (cache_root / f"CIK{int(cik):010d}.json").exists()
            and not force
        )
        try:
            facts = fetch_company_facts(
                cik,
                user_agent=user_agent,
                session=sess,
                cache_dir=cache_root,
                force=force,
            )
            df = annual_financials_from_facts(facts)
            if df.empty:
                logger.warning("CIK %d: keine 10-K/10-Q-Werte in companyfacts", cik)
            else:
                frames.append(df)
        except requests.HTTPError as exc:
            logger.warning("CIK %d übersprungen (%s)", cik, exc)
        except requests.RequestException as exc:
            logger.warning("CIK %d Netzwerkfehler (%s)", cik, exc)
        if i % 25 == 0:
            logger.info("  … %d/%d CIKs abgefragt", i, len(ciks))
        if not cached:
            time.sleep(sleep_s)

    if not frames:
        return _empty_panel()
    panel = pd.concat(frames, ignore_index=True)
    if "cik" not in panel.columns:
        return _empty_panel()
    subset = (
        ["cik", "reporting_date"]
        if "reporting_date" in panel.columns
        else ["cik", "fyear"]
    )
    panel = panel.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    logger.info("Finanz-Panel: %d Firm-Years, %d CIKs", len(panel), panel["cik"].nunique())
    return panel
