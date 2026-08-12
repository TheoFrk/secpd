"""EDGAR-Zugriff (nur Home-Setup mit Internet; Ergebnisse werden committet).

Da der Zenodo-Datensatz **keine Finanzkennzahlen** enthält, kommen die
tabellarischen Features aus der SEC-XBRL-API ``companyfacts`` — strukturiert,
kostenlos und über den CIK direkt joinbar:

    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

Die SEC verlangt einen deklarierten User-Agent („Name Kontakt@mail") und
faires Rate-Limiting (< 10 req/s; hier konservativ gedrosselt).
Abdeckung: XBRL-Companyfacts sind erst ab ca. GJ 2009 flächig verfügbar —
für ältere Firm-Years greift später die Median-Imputation der Pipeline.
"""
from __future__ import annotations

import logging
import time
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


def fetch_company_facts(
    cik: int,
    *,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Lädt das companyfacts-JSON eines CIK (wirft bei HTTP-Fehlern)."""
    if not user_agent:
        raise ValueError(
            "SEC verlangt einen User-Agent — SECPD_SEC_UA setzen, "
            'z. B. "Commerzbank Praktikum vorname.nachname@example.com".'
        )
    sess = session or requests.Session()
    resp = sess.get(
        COMPANYFACTS_URL.format(cik=int(cik)),
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def annual_financials_from_facts(facts: dict[str, Any]) -> pd.DataFrame:
    """Extrahiert Jahreswerte (Form 10-K, fp=FY) ins kanonische Schema.

    Point-in-time: je ``(fyear, Konzept)`` zählt der **zuerst filed**-Wert
    (frühestes ``filed``-Datum). Spätere 10-K/A-Restatements überschreiben
    nicht — Look-ahead durch nachträgliche Korrekturen wird vermieden.

    Rückgabe: eine Zeile je ``fyear`` mit den Spalten aus :data:`TAG_MAP`.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    # fyear → concept → (value, filed_ts)
    per_year: dict[int, dict[str, tuple[float, pd.Timestamp]]] = {}

    for canonical, tags in TAG_MAP.items():
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {})
            entries = units.get("USD") or []
            found_any = False
            for e in entries:
                if e.get("form") not in _ANNUAL_FORMS or e.get("fp") != "FY":
                    continue
                fy = e.get("fy")
                val = e.get("val")
                if fy is None or val is None:
                    continue
                filed = pd.to_datetime(e.get("filed"), errors="coerce")
                row = per_year.setdefault(int(fy), {})
                prev = row.get(canonical)
                if prev is None:
                    row[canonical] = (float(val), filed)
                    found_any = True
                    continue
                _, prev_filed = prev
                # Früheres filed-Datum gewinnt (erster berichteter Wert).
                if pd.isna(prev_filed) and pd.notna(filed):
                    row[canonical] = (float(val), filed)
                elif pd.notna(filed) and pd.notna(prev_filed) and filed < prev_filed:
                    row[canonical] = (float(val), filed)
                found_any = True
            if found_any:
                break  # erster Tag mit Treffern gewinnt (Prioritätsliste)

    if not per_year:
        return pd.DataFrame(columns=["fyear", *TAG_MAP.keys()])
    rows = []
    for fy, vals in sorted(per_year.items()):
        row: dict[str, Any] = {"fyear": fy}
        for concept, (value, _) in vals.items():
            row[concept] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    df["cik"] = int(facts.get("cik", 0))
    return df


def build_financials_panel(
    ciks: Iterable[int],
    *,
    user_agent: str,
    sleep_s: float = 0.15,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Lädt Companyfacts für mehrere CIKs und stapelt sie zu einem Panel.

    Fehlertolerant: einzelne 404/Fehler werden geloggt und übersprungen.
    """
    sess = session or requests.Session()
    frames: list[pd.DataFrame] = []
    ciks = sorted({int(c) for c in ciks})
    for i, cik in enumerate(ciks, start=1):
        try:
            facts = fetch_company_facts(cik, user_agent=user_agent, session=sess)
            frames.append(annual_financials_from_facts(facts))
        except requests.HTTPError as exc:
            logger.warning("CIK %d übersprungen (%s)", cik, exc)
        except requests.RequestException as exc:
            logger.warning("CIK %d Netzwerkfehler (%s)", cik, exc)
        if i % 25 == 0:
            logger.info("  … %d/%d CIKs abgefragt", i, len(ciks))
        time.sleep(sleep_s)

    if not frames:
        return pd.DataFrame(columns=["cik", "fyear", *TAG_MAP.keys()])
    panel = pd.concat(frames, ignore_index=True)
    # Keep first: bei doppelten CIK-Läufen nicht das spätere Panel gewinnen lassen
    panel = panel.drop_duplicates(subset=["cik", "fyear"], keep="first").reset_index(drop=True)
    logger.info("Finanz-Panel: %d Firm-Years, %d CIKs", len(panel), panel["cik"].nunique())
    return panel


def download_raw_filings(
    identifiers: Iterable[str],
    *,
    form: str = "10-K",
    dest: str = "data/raw/edgar",
    company: str = "SEC-PD Praktikum",
    email: str = "",
    limit: int | None = 5,
) -> None:
    """Optionaler Roh-Download ganzer Filings via ``sec-edgar-downloader``.

    Lazy-Import, damit das Paket auf dem air-gapped Server auch ohne dieses
    (Online-)Extra importierbar bleibt.
    """
    from sec_edgar_downloader import Downloader  # noqa: PLC0415 — bewusst lazy

    dl = Downloader(company, email or "unknown@example.com", dest)
    for ident in identifiers:
        dl.get(form, ident, limit=limit, download_details=False)
        logger.info("Heruntergeladen: %s %s", form, ident)
