"""Issuer-Ratings als optionales Label (nie als Feature).

Drei Zielvariablen:

1. **ordinal** (Default): PIT-Notch 1–21 zum Bilanzstichtag (Regression).
2. **speculative** (IG/HY-ähnlich): ``label_rating = 1``, wenn das PIT-Rating
   zum Bilanzstichtag im spekulativen Bereich liegt (``BB+`` und schwächer).
3. **downgrade**: ``label_rating = 1``, wenn der Notch im Prognosehorizont
   nach ``reporting_date`` strikt fällt.

PIT-Join: letzte Rating-Beobachtung mit ``rating_date ≤ reporting_date``.
Ratings fließen bewusst **nicht** in ``fin_*`` / ``evt_*`` — sonst Leakage.

**Quellen**

* **ratingshistory.info** (SEC Regulation 17g-7, Default): echte
  Moody-/Fitch-/Egan-Jones-Historien als Bulk-CSV. Join über LEI und
  normalisierten Emittentennamen auf CIK. ~12 Monate Meldeverzug.
* **FMP Historical Ratings** (optional, ``SECPD_FMP_API_KEY``): FMPs eigene
  Fundamentalnote (A+–D), kein Agency-Rating. Free-Tier ~250 Calls/Tag;
  ``--max-requests`` zählt nur Netz-Hits.
* Beliebiges Agency-CSV im kanonischen Schema
  (``cik`` oder ``ticker``, ``rating_date``, ``rating``, optional ``agency``).

Ticker-/Namens-Map: EDGAR-Submissions-Cache und SEC ``company_tickers.json``.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FMP_HISTORICAL_STABLE = (
    "https://financialmodelingprep.com/stable/ratings-historical"
)
FMP_HISTORICAL_V3 = "https://financialmodelingprep.com/api/v3/historical-rating/{symbol}"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
RATINGSHISTORY_INDEX = "https://ratingshistory.info/"
RATINGSHISTORY_API = "https://ratingshistory.info/api/public/"

DEFAULT_NRSRO_AGENCIES = ("moodys", "fitch", "egan-jones")
DEFAULT_NRSRO_CATEGORIES = ("Corporate", "Financial")
_NRSRO_UA = "secpd/0.1 (research; ratingshistory.info)"

DEFAULT_LABEL_COL = "label_rating"
HY_MAX_NOTCH = 11  # BB+ und schwächer = speculative / HY-ähnlich
IG_MIN_NOTCH = 12  # BBB- und besser

#: S&P / Fitch Long-term (höher = besser).
_SP_FITCH: dict[str, int] = {
    "AAA": 21,
    "AA+": 20,
    "AA": 19,
    "AA-": 18,
    "A+": 17,
    "A": 16,
    "A-": 15,
    "BBB+": 14,
    "BBB": 13,
    "BBB-": 12,
    "BB+": 11,
    "BB": 10,
    "BB-": 9,
    "B+": 8,
    "B": 7,
    "B-": 6,
    "CCC+": 5,
    "CCC": 4,
    "CCC-": 3,
    "CC": 2,
    "C": 1,
    "D": 1,
    "SD": 1,
    "RD": 1,
}

#: Moody's Long-term.
_MOODY: dict[str, int] = {
    "AAA": 21,
    "AA1": 20,
    "AA2": 19,
    "AA3": 18,
    "A1": 17,
    "A2": 16,
    "A3": 15,
    "BAA1": 14,
    "BAA2": 13,
    "BAA3": 12,
    "BA1": 11,
    "BA2": 10,
    "BA3": 9,
    "B1": 8,
    "B2": 7,
    "B3": 6,
    "CAA1": 5,
    "CAA2": 4,
    "CAA3": 3,
    "CA": 2,
    "C": 1,
}

#: FMP-Fundamentalnote — grob auf dieselbe Notch-Skala gelegt.
_FMP: dict[str, int] = {
    "A+": 17,
    "A": 16,
    "A-": 15,
    "B+": 13,
    "B": 12,
    "B-": 11,
    "C+": 8,
    "C": 7,
    "C-": 6,
    "D+": 4,
    "D": 3,
    "D-": 2,
    "E": 1,
}

_FMP_SCORE_TO_NOTCH = {5: 16, 4: 13, 3: 10, 2: 7, 1: 3}

#: Notch → S&P/Fitch-Buchstabe (1 = Default, 21 = AAA).
_NOTCH_TO_SP: dict[int, str] = {
    21: "AAA",
    20: "AA+",
    19: "AA",
    18: "AA-",
    17: "A+",
    16: "A",
    15: "A-",
    14: "BBB+",
    13: "BBB",
    12: "BBB-",
    11: "BB+",
    10: "BB",
    9: "BB-",
    8: "B+",
    7: "B",
    6: "B-",
    5: "CCC+",
    4: "CCC",
    3: "CCC-",
    2: "CC",
    1: "D",
}

#: Notch → Moody's Long-term (konventionelle Schreibweise).
_NOTCH_TO_MOODY: dict[int, str] = {
    21: "Aaa",
    20: "Aa1",
    19: "Aa2",
    18: "Aa3",
    17: "A1",
    16: "A2",
    15: "A3",
    14: "Baa1",
    13: "Baa2",
    12: "Baa3",
    11: "Ba1",
    10: "Ba2",
    9: "Ba3",
    8: "B1",
    7: "B2",
    6: "B3",
    5: "Caa1",
    4: "Caa2",
    3: "Caa3",
    2: "Ca",
    1: "C",
}

_AGENCY_PRIORITY = {
    "moodys": 0,
    "moody": 0,
    "moody's": 0,
    "fitch": 1,
    "s&p": 1,
    "sp": 1,
    "egan-jones": 2,
    "egan": 2,
    "dbrs": 3,
    "fmp": 9,
    "fmp_fundamental": 9,
}

_SKIP_LETTERS = {
    "NR", "NA", "N/A", "WR", "WD", "NONE", "NP",
    "P-1", "P-2", "P-3", "P-4", "P1", "P2", "P3",
    "F1", "F1+", "F2", "F3",
    "A-1", "A-1+", "A-2", "A-3",
}

_SHORT_TERM_NEEDLES = (
    "commercial paper",
    "other short term",
    "short-term",
    "short term",
    "probability of default",
)


class RatingsFetchError(RuntimeError):
    """API-Fehler beim Rating-Fetch (Key, Quota, HTTP)."""


# --------------------------------------------------------------------------- #
# Notch-Skala (AAA=21 … D=1) — Label-Hilfen, nie Features
# --------------------------------------------------------------------------- #


def normalize_letter(raw: Any) -> str:
    """``BBB-`` / ``Baa3`` / ``(P)A1`` → kanonischer Token ohne Leerzeichen."""
    s = str(raw or "").strip().upper().replace(" ", "")
    s = s.replace("*", "")
    if s.startswith("(P)"):
        s = s[3:]
    if s.endswith("-PD"):
        s = s[:-3]
    if s.endswith("/A") or s.endswith("/P"):  # Outlook-Müll
        s = s[:-2]
    return s


def notch_from_rating(
    letter: Any,
    *,
    agency: str | None = None,
    rating_score: Any = None,
) -> int | None:
    """Buchstabenrating → Notch (höher = besser). Unbekannt → None."""
    token = normalize_letter(letter)
    if not token or token in _SKIP_LETTERS or token in {"NR", "NA", "N/A", "WR", "WD", "NONE"}:
        if rating_score is not None and pd.notna(rating_score):
            try:
                return _FMP_SCORE_TO_NOTCH.get(int(rating_score))
            except (TypeError, ValueError):
                return None
        return None
    src = str(agency or "").strip().lower()
    if src in {"moodys", "moody", "moody's", "mdy"}:
        return _MOODY.get(token)
    if src in {"fmp", "fmp_fundamental"}:
        return _FMP.get(token) or _SP_FITCH.get(token)
    # S&P / Fitch / unbekannt: erst S&P, dann Moody, dann FMP
    return _SP_FITCH.get(token) or _MOODY.get(token) or _FMP.get(token)


def notch_to_letter(notch: Any, *, scale: str = "sp") -> str:
    """Notch 1–21 → Buchstabe. ``scale``: ``sp`` (Default), ``moodys`` oder ``fmp``."""
    try:
        n = int(round(float(notch)))
    except (TypeError, ValueError):
        return "—"
    if n != n:  # NaN
        return "—"
    n = min(21, max(1, n))
    src = str(scale or "sp").strip().lower()
    if src in {"fmp", "fmp_fundamental"}:
        best_letter, best_dist = "E", 99
        for letter, val in _FMP.items():
            dist = abs(int(val) - n)
            if dist < best_dist:
                best_letter, best_dist = letter, dist
        return best_letter
    if src in {"moodys", "moody", "moody's", "mdy"}:
        return _NOTCH_TO_MOODY.get(n, "—")
    return _NOTCH_TO_SP.get(n, "—")


def format_notch(notch: Any, *, scale: str = "sp") -> str:
    """Anzeige ``BBB (Notch 13/21)`` bzw. Moody's ``Baa2 (Notch 13/21)``."""
    try:
        n = int(round(float(notch)))
    except (TypeError, ValueError):
        return "—"
    if n != n:
        return "—"
    n = min(21, max(1, n))
    return f"{notch_to_letter(n, scale=scale)} (Notch {n}/21)"


def is_speculative(notch: Any) -> bool:
    """True, wenn Notch ``BB+`` oder schwächer (HY-ähnlich)."""
    if notch is None or (isinstance(notch, float) and pd.isna(notch)):
        return False
    try:
        return int(notch) <= HY_MAX_NOTCH
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# FMP (optional, Fundamentalnote — kein Agency-Rating)
# --------------------------------------------------------------------------- #


def fmp_symbol(ticker: str) -> str:
    """SEC-Ticker → FMP-Symbol (``BRK.B`` → ``BRK-B``)."""
    return str(ticker or "").strip().upper().replace(".", "-")


# --------------------------------------------------------------------------- #
# Ticker-Map
# --------------------------------------------------------------------------- #


def ticker_map_from_submissions(
    cache_dir: Path | str = "data/raw/edgar_submissions",
) -> pd.DataFrame:
    """CIK → Ticker aus dem lokalen Submissions-Cache (kein Netz)."""
    cache_dir = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    if not cache_dir.is_dir():
        return pd.DataFrame(columns=["cik", "ticker", "name", "lei"])
    for path in sorted(cache_dir.glob("CIK*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_cik = payload.get("cik") or path.stem.replace("CIK", "")
        try:
            cik = int(raw_cik)
        except (TypeError, ValueError):
            continue
        tickers = payload.get("tickers") or []
        ticker = ""
        if isinstance(tickers, list) and tickers:
            ticker = str(tickers[0]).strip().upper()
        elif isinstance(tickers, str):
            ticker = tickers.strip().upper()
        rows.append(
            {
                "cik": cik,
                "ticker": ticker,
                "name": payload.get("name") or "",
                "lei": (payload.get("lei") or "") if isinstance(payload.get("lei"), str) else "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["cik"], keep="first").reset_index(drop=True)
    logger.info(
        "Ticker-Map aus Submissions: %d CIKs, davon %d mit Ticker.",
        len(out),
        int((out["ticker"].astype(str).str.len() > 0).sum()),
    )
    return out


def fetch_sec_company_tickers(
    *,
    user_agent: str,
    cache_path: Path | str = "data/raw/company_tickers.json",
    force: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """SEC-Datei ``company_tickers.json`` (CIK, Ticker, Titel)."""
    if not user_agent:
        raise RatingsFetchError(
            "SEC verlangt einen User-Agent — SECPD_SEC_UA setzen."
        )
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        resp = requests.get(
            SEC_COMPANY_TICKERS_URL,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = []
    if isinstance(payload, dict):
        values: Iterable[Any] = payload.values()
    else:
        values = payload
    for rec in values:
        if not isinstance(rec, dict):
            continue
        try:
            cik = int(rec.get("cik_str") or rec.get("cik"))
        except (TypeError, ValueError):
            continue
        ticker = str(rec.get("ticker") or "").strip().upper()
        rows.append({"cik": cik, "ticker": ticker, "name": rec.get("title") or ""})
    out = pd.DataFrame(rows).drop_duplicates(subset=["cik"], keep="first")
    logger.info("SEC company_tickers: %d CIKs.", len(out))
    return out


def merge_ticker_maps(*frames: pd.DataFrame) -> pd.DataFrame:
    """Erste nicht-leere Ticker-Zelle je CIK gewinnt (Submissions vor SEC)."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=["cik", "ticker", "name", "lei"])
    out = pd.concat(parts, ignore_index=True)
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce")
    out = out.dropna(subset=["cik"])
    out["cik"] = out["cik"].astype(int)
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()
    out["name"] = out.get("name", pd.Series("", index=out.index)).fillna("").astype(str)
    if "lei" not in out.columns:
        out["lei"] = ""
    out["lei"] = out["lei"].fillna("").astype(str).str.strip().str.upper()
    out["_empty"] = out["ticker"] == ""
    out["_lei_empty"] = out["lei"] == ""
    out = out.sort_values(["cik", "_empty", "_lei_empty"]).drop_duplicates("cik", keep="first")
    return out.drop(columns=["_empty", "_lei_empty"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# FMP-Fetch
# --------------------------------------------------------------------------- #


def fetch_fmp_historical(
    symbol: str,
    *,
    api_key: str,
    cache_dir: Path | str = "data/raw/fmp_ratings",
    force: bool = False,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Historische FMP-Ratings eines Tickers (Datei-Cache, resumefähig).

    Rückgabe: ``(rows, from_cache)``.
    """
    sym = fmp_symbol(symbol)
    if not sym:
        return [], True
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{sym}.json"
    if cache_file.exists() and not force:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return (payload if isinstance(payload, list) else []), True
    if not api_key:
        raise RatingsFetchError(
            "SECPD_FMP_API_KEY fehlt — kostenlosen Key unter "
            "https://site.financialmodelingprep.com/register anlegen "
            "und in .secpd.env speichern."
        )

    sess = session or requests.Session()
    params = {"symbol": sym, "apikey": api_key}
    resp = sess.get(FMP_HISTORICAL_STABLE, params=params, timeout=timeout)
    if resp.status_code == 404:
        resp = sess.get(
            FMP_HISTORICAL_V3.format(symbol=sym),
            params={"apikey": api_key},
            timeout=timeout,
        )
    if resp.status_code in {401, 403}:
        raise RatingsFetchError(
            "FMP hat den API-Key abgelehnt (401/403). Key prüfen, "
            "Free-Tier-Quota oder Sandbox-Symbole beachten."
        )
    if resp.status_code == 429:
        raise RatingsFetchError("FMP Rate-Limit (429) — später fortsetzen.")
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RatingsFetchError(str(payload["Error Message"]))
    rows = payload if isinstance(payload, list) else []
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return rows, False


def parse_fmp_payload(payload: list[dict[str, Any]], *, ticker: str) -> pd.DataFrame:
    """FMP-JSON → kanonisches Rating-Panel (eine Firma)."""
    recs: list[dict[str, Any]] = []
    for row in payload or []:
        if not isinstance(row, dict):
            continue
        date = row.get("date") or row.get("ratingDate") or row.get("datetime")
        letter = row.get("rating") or row.get("ratingRecommendation")
        score = row.get("ratingScore")
        recs.append(
            {
                "ticker": fmp_symbol(row.get("symbol") or ticker),
                "rating_date": date,
                "rating": letter,
                "rating_score": score,
                "agency": "fmp",
                "source": "fmp",
            }
        )
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return normalize_ratings_panel(out)


def build_fmp_ratings_panel(
    ticker_map: pd.DataFrame,
    *,
    api_key: str,
    cache_dir: Path | str = "data/raw/fmp_ratings",
    sleep_s: float = 0.25,
    force: bool = False,
    cache_only: bool = False,
    limit: int | None = None,
    max_requests: int | None = None,
) -> pd.DataFrame:
    """Lädt FMP-Historien für alle Ticker der Map (resumefähig).

    ``max_requests`` zählt nur echte API-Calls (Cache-Hits gehen nicht drauf).
    Die Map sollte bereits priorisiert sein (häufigste CIKs zuerst).
    """
    if ticker_map.empty:
        return pd.DataFrame()
    work = ticker_map.copy()
    work["ticker"] = work["ticker"].fillna("").astype(str).str.strip().str.upper()
    work = work.loc[work["ticker"] != ""].drop_duplicates("ticker")
    if limit is not None:
        work = work.head(int(limit))
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    n_ok = n_empty = n_err = n_net = 0
    stop_net = bool(cache_only)
    for i, row in enumerate(work.itertuples(index=False), start=1):
        ticker = str(row.ticker)
        cik = int(row.cik)
        from_cache = True
        try:
            if stop_net:
                cache_file = Path(cache_dir) / f"{fmp_symbol(ticker)}.json"
                if not cache_file.exists():
                    continue
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                payload = payload if isinstance(payload, list) else []
            else:
                payload, from_cache = fetch_fmp_historical(
                    ticker,
                    api_key=api_key,
                    cache_dir=cache_dir,
                    force=force,
                    session=session,
                )
                if not from_cache:
                    n_net += 1
                    if max_requests is not None and n_net >= int(max_requests):
                        stop_net = True
                        logger.info(
                            "FMP max-requests=%d erreicht — Rest nur aus Cache.",
                            int(max_requests),
                        )
        except RatingsFetchError:
            raise
        except requests.RequestException as exc:
            n_err += 1
            logger.warning("FMP %s fehlgeschlagen: %s", ticker, exc)
            continue
        parsed = parse_fmp_payload(payload, ticker=ticker)
        if parsed.empty:
            n_empty += 1
        else:
            parsed["cik"] = cik
            frames.append(parsed)
            n_ok += 1
        if i % 50 == 0:
            logger.info(
                "FMP-Fortschritt: %d/%d Ticker (%d mit Historie, %d API-Calls).",
                i, len(work), n_ok, n_net,
            )
        if not stop_net and not from_cache and sleep_s:
            time.sleep(sleep_s)
    if not frames:
        logger.warning(
            "Kein FMP-Rating geladen (ok=%d leer=%d err=%d net=%d).",
            n_ok, n_empty, n_err, n_net,
        )
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "FMP-Panel: %d Zeilen, %d CIKs (%d Ticker leer, %d Fehler, %d API-Calls).",
        len(out),
        out["cik"].nunique(),
        n_empty,
        n_err,
        n_net,
    )
    return out


# --------------------------------------------------------------------------- #
# ratingshistory.info (SEC 17g-7 NRSRO-Historien)
# --------------------------------------------------------------------------- #


_HREF_RE = re.compile(
    r'href="https://ratingshistory\.info/api/public/([^"]+\.csv)"',
    re.IGNORECASE,
)
_FILE_RE = re.compile(r"^(\d{8})\s+(.+)\s+(\S+)\.csv$", re.IGNORECASE)

_LEGAL_SUFFIXES = (
    "INCORPORATED", "INCORPORATION", "CORPORATION", "COMPANY", "LIMITED",
    "INC", "CORP", "CO", "LTD", "LLC", "PLC", "LP", "LLP", "SA", "AG",
    "NV", "PLC", "THE", "HOLDINGS", "HOLDING", "GROUP", "PLC",
)


# --------------------------------------------------------------------------- #
# NRSRO / ratingshistory.info (17g-7 Bulk)
# --------------------------------------------------------------------------- #


def canonicalize_agency(raw: Any) -> str:
    """``Moody's Investors Service`` → ``moodys``."""
    s = str(raw or "").strip().lower()
    if "moody" in s:
        return "moodys"
    if "fitch" in s:
        return "fitch"
    if "egan" in s:
        return "egan-jones"
    if "dbrs" in s or "morningstar" in s:
        return "dbrs"
    if "fmp" in s:
        return "fmp"
    return s.replace(" ", "-")[:32] or "unknown"


def normalize_issuer_name(raw: Any) -> str:
    """Grobe Namensnormalisierung für den CIK-Join."""
    s = str(raw or "").upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    for tok in _LEGAL_SUFFIXES:
        s = re.sub(rf"\b{tok}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _term_priority(term: Any, rating_type: Any = None) -> int | None:
    """0 = Issuer/CFR, 1 = Long-term, 2 = Senior Unsecured, None = drop."""
    t = f"{term or ''} {rating_type or ''}".strip().lower()
    if any(n in t for n in _SHORT_TERM_NEEDLES):
        return None
    if any(n in t for n in ("corporate family", "lt issuer", "issuer default", "issuer rating")):
        return 0
    if "organization" in t:
        return 0
    if "long term" in t or "long-term" in t:
        return 1
    if "senior unsecured" in t and "backed" not in t and "mtn" not in t:
        return 2
    if "senior unsecured" in t:
        return 3
    return None


def list_ratingshistory_files(
    *,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Index von ratingshistory.info → date, agency, category, filename."""
    sess = session or requests.Session()
    resp = sess.get(
        RATINGSHISTORY_INDEX,
        headers={"User-Agent": _NRSRO_UA, "Accept": "text/html"},
        timeout=timeout,
    )
    resp.raise_for_status()
    rows: list[dict[str, Any]] = []
    for fname in _HREF_RE.findall(resp.text):
        fname = unquote(fname).replace("&amp;", "&")
        m = _FILE_RE.match(fname)
        if not m:
            continue
        date_s, agency_raw, category = m.group(1), m.group(2), m.group(3)
        rows.append(
            {
                "file_date": date_s,
                "agency_raw": agency_raw,
                "agency": canonicalize_agency(agency_raw),
                "category": category,
                "filename": fname,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["agency", "category", "file_date"]).reset_index(drop=True)


def latest_nrsro_files(
    catalog: pd.DataFrame,
    *,
    agencies: Iterable[str] = DEFAULT_NRSRO_AGENCIES,
    categories: Iterable[str] = DEFAULT_NRSRO_CATEGORIES,
) -> pd.DataFrame:
    """Je Agency×Kategorie die neueste Datei."""
    if catalog.empty:
        return catalog
    want_ag = {canonicalize_agency(a) for a in agencies}
    want_cat = {str(c).strip().lower() for c in categories}
    sub = catalog.loc[
        catalog["agency"].isin(want_ag)
        & catalog["category"].str.lower().isin(want_cat)
    ].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("file_date")
    return sub.drop_duplicates(["agency", "category"], keep="last").reset_index(drop=True)


def download_ratingshistory_file(
    filename: str,
    cache_dir: Path | str,
    *,
    force: bool = False,
    timeout: int = 180,
    session: requests.Session | None = None,
) -> tuple[Path, bool]:
    """Lädt eine 17g-7-CSV in den Cache. Rückgabe: ``(path, from_cache)``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = filename.replace("'", "").replace(" ", "_")
    dest = cache_dir / safe
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest, True
    url = RATINGSHISTORY_API + quote(filename)
    sess = session or requests.Session()
    logger.info("Download %s …", filename)
    resp = sess.get(
        url,
        headers={"User-Agent": _NRSRO_UA, "Accept": "text/csv"},
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)
    logger.info("Gespeichert: %s (%.1f MB)", dest, dest.stat().st_size / (1 << 20))
    return dest, False


def parse_nrsro_csv(path: Path | str) -> pd.DataFrame:
    """17g-7-CSV → Long-term-Issuer-/Senior-Unsecured-Aktionen."""
    path = Path(path)
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return raw
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    term = raw["rating_type_term"] if "rating_type_term" in raw.columns else ""
    rtype = raw["rating_type"] if "rating_type" in raw.columns else ""
    if isinstance(term, str):
        prio = pd.Series([_term_priority(term, rtype)] * len(raw), index=raw.index)
    else:
        prio = [
            _term_priority(t, rt)
            for t, rt in zip(term.fillna(""), rtype.fillna("") if not isinstance(rtype, str) else [rtype] * len(raw))
        ]
        prio = pd.Series(prio, index=raw.index)
    keep = prio.notna()
    out = raw.loc[keep].copy()
    out["_prio"] = prio.loc[keep]
    if out.empty:
        return out

    agency_col = out["rating_agency_name"] if "rating_agency_name" in out.columns else "unknown"
    out["agency"] = (
        agency_col.map(canonicalize_agency)
        if isinstance(agency_col, pd.Series)
        else canonicalize_agency(agency_col)
    )
    letter = out["rating"] if "rating" in out.columns else None
    notches: list[int | None] = []
    for i in range(len(out)):
        let = letter.iloc[i] if isinstance(letter, pd.Series) else None
        ag = out["agency"].iloc[i]
        notches.append(notch_from_rating(let, agency=ag))
    out["notch"] = pd.array(notches, dtype="Int64")
    out = out.loc[out["notch"].notna()].copy()
    if out.empty:
        return out

    date_col = "rating_action_date" if "rating_action_date" in out.columns else "rating_date"
    out["rating_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["rating_date"])

    cik_raw = out["central_index_key"] if "central_index_key" in out.columns else None
    if cik_raw is not None:
        cik_num = pd.to_numeric(cik_raw, errors="coerce")
        out["cik"] = cik_num.astype("Int64")
    else:
        out["cik"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    lei = out["legal_entity_identifier"] if "legal_entity_identifier" in out.columns else ""
    out["lei"] = (
        lei.fillna("").astype(str).str.strip().str.upper()
        if isinstance(lei, pd.Series)
        else ""
    )
    name_col = "issuer_name" if "issuer_name" in out.columns else "obligor_name"
    if name_col in out.columns:
        out["issuer_name"] = out[name_col].fillna("").astype(str)
    else:
        out["issuer_name"] = ""
    out["name_key"] = out["issuer_name"].map(normalize_issuer_name)
    out["source"] = "nrsro"
    keep_cols = [
        c for c in (
            "cik", "lei", "issuer_name", "name_key", "rating_date", "rating",
            "notch", "agency", "source", "_prio",
        ) if c in out.columns
    ]
    return out[keep_cols].reset_index(drop=True)


def assign_cik_to_ratings(
    panel: pd.DataFrame,
    ticker_map: pd.DataFrame,
) -> pd.DataFrame:
    """CIK über vorhandene CIK-Spalte, dann LEI, dann eindeutigen Namen."""
    out = panel.copy()
    if "cik" not in out.columns:
        out["cik"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")

    mp = ticker_map.copy() if ticker_map is not None and not ticker_map.empty else pd.DataFrame()
    if not mp.empty:
        mp["cik"] = pd.to_numeric(mp["cik"], errors="coerce")
        mp = mp.dropna(subset=["cik"])
        mp["cik"] = mp["cik"].astype(int)
        if "lei" in mp.columns:
            mp["lei"] = mp["lei"].fillna("").astype(str).str.strip().str.upper()
            lei_map = (
                mp.loc[mp["lei"].str.len() >= 10, ["lei", "cik"]]
                .drop_duplicates("lei", keep="first")
                .set_index("lei")["cik"]
            )
        else:
            lei_map = pd.Series(dtype="int64")
        mp["name_key"] = mp.get("name", pd.Series("", index=mp.index)).map(normalize_issuer_name)
        name_counts = mp.loc[mp["name_key"] != "", "name_key"].value_counts()
        unique_names = set(name_counts[name_counts == 1].index)
        name_map = (
            mp.loc[mp["name_key"].isin(unique_names), ["name_key", "cik"]]
            .drop_duplicates("name_key", keep="first")
            .set_index("name_key")["cik"]
        )
    else:
        lei_map = pd.Series(dtype="int64")
        name_map = pd.Series(dtype="int64")

    missing = out["cik"].isna()
    if missing.any() and "lei" in out.columns and len(lei_map):
        mapped = out.loc[missing, "lei"].map(lei_map)
        out.loc[missing, "cik"] = mapped
    missing = out["cik"].isna()
    if missing.any() and "name_key" in out.columns and len(name_map):
        mapped = out.loc[missing, "name_key"].map(name_map)
        out.loc[missing, "cik"] = mapped

    n_before = len(out)
    out = out.loc[out["cik"].notna()].copy()
    out["cik"] = out["cik"].astype(int)
    logger.info(
        "CIK-Join: %d → %d Zeilen mit CIK (%d Emittenten).",
        n_before, len(out), out["cik"].nunique() if len(out) else 0,
    )
    return out


def collapse_ratings_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je CIK×Datum: Issuer-Term vor Senior Unsecured, Moody vor Fitch."""
    if df.empty:
        return df
    out = df.copy()
    if "_prio" not in out.columns:
        out["_prio"] = 2
    out["_ag_prio"] = out.get("agency", pd.Series("", index=out.index)).map(
        lambda a: _AGENCY_PRIORITY.get(str(a).lower(), 8)
    )
    out = out.sort_values(["cik", "rating_date", "_prio", "_ag_prio"])
    out = out.drop_duplicates(subset=["cik", "rating_date"], keep="first")
    return out.drop(columns=["_prio", "_ag_prio"], errors="ignore").reset_index(drop=True)


def build_nrsro_ratings_panel(
    ticker_map: pd.DataFrame,
    *,
    cache_dir: Path | str = "data/raw/nrsro_ratings",
    agencies: Iterable[str] = DEFAULT_NRSRO_AGENCIES,
    categories: Iterable[str] = DEFAULT_NRSRO_CATEGORIES,
    force: bool = False,
    cache_only: bool = False,
    timeout: int = 180,
) -> pd.DataFrame:
    """Lädt Moody/Fitch/Egan-Jones Corporate(+Financial) und joined auf CIK."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    if cache_only:
        files = sorted(p for p in cache_dir.glob("*.csv") if p.is_file())
        if not files:
            logger.warning("NRSRO-Cache leer (%s).", cache_dir)
            return pd.DataFrame()
        frames = [parse_nrsro_csv(p) for p in files]
    else:
        try:
            catalog = list_ratingshistory_files(session=session)
        except requests.RequestException as exc:
            logger.warning("ratingshistory.info Index fehlgeschlagen: %s — nutze Cache.", exc)
            files = sorted(p for p in cache_dir.glob("*.csv") if p.is_file())
            if not files:
                raise RatingsFetchError(
                    f"NRSRO-Index nicht erreichbar und Cache leer: {exc}"
                ) from exc
            frames = [parse_nrsro_csv(p) for p in files]
        else:
            chosen = latest_nrsro_files(catalog, agencies=agencies, categories=categories)
            if chosen.empty:
                logger.warning("Keine passenden NRSRO-Dateien im Index.")
                return pd.DataFrame()
            for rec in chosen.itertuples(index=False):
                try:
                    path, _hit = download_ratingshistory_file(
                        rec.filename,
                        cache_dir,
                        force=force,
                        timeout=timeout,
                        session=session,
                    )
                except requests.RequestException as exc:
                    logger.warning("Download %s fehlgeschlagen: %s", rec.filename, exc)
                    continue
                parsed = parse_nrsro_csv(path)
                logger.info(
                    "NRSRO %s %s: %d Long-term-Zeilen.",
                    rec.agency, rec.category, len(parsed),
                )
                frames.append(parsed)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        logger.warning("Kein NRSRO-Rating geladen.")
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = assign_cik_to_ratings(panel, ticker_map)
    if panel.empty:
        return panel
    if ticker_map is not None and not ticker_map.empty and "ticker" in ticker_map.columns:
        tmap = ticker_map.drop_duplicates("cik").set_index("cik")["ticker"]
        panel["ticker"] = panel["cik"].map(tmap).fillna("")
    panel = collapse_ratings_panel(panel)
    panel = normalize_ratings_panel(panel)
    logger.info(
        "NRSRO-Panel: %d Zeilen, %d CIKs, %s–%s.",
        len(panel),
        int(panel["cik"].nunique()) if len(panel) else 0,
        panel["rating_date"].min().date() if len(panel) else "—",
        panel["rating_date"].max().date() if len(panel) else "—",
    )
    return panel


# --------------------------------------------------------------------------- #
# Panel laden / normalisieren / Labels
# --------------------------------------------------------------------------- #


def normalize_ratings_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Spalten kanonisieren, Notch berechnen, Duplikate droppen."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    rename = {
        "date": "rating_date",
        "ratingdate": "rating_date",
        "symbol": "ticker",
        "grade": "rating",
        "ratings": "rating",
        "agency_name": "agency",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "rating_date" not in out.columns:
        raise ValueError("Ratings-Panel braucht eine Datumsspalte (rating_date).")
    if "rating" not in out.columns and "rating_score" not in out.columns:
        raise ValueError("Ratings-Panel braucht rating oder rating_score.")
    if "cik" in out.columns:
        out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].fillna("").map(fmp_symbol)
    out["rating_date"] = pd.to_datetime(out["rating_date"], errors="coerce")
    agency = out["agency"] if "agency" in out.columns else "fmp"
    score = out["rating_score"] if "rating_score" in out.columns else None
    letter = out["rating"] if "rating" in out.columns else None
    notches: list[int | None] = []
    n = len(out)
    for i in range(n):
        ag = agency.iloc[i] if isinstance(agency, pd.Series) else agency
        sc = score.iloc[i] if isinstance(score, pd.Series) else None
        let = letter.iloc[i] if isinstance(letter, pd.Series) else None
        notches.append(notch_from_rating(let, agency=ag, rating_score=sc))
    out["notch"] = pd.array(notches, dtype="Int64")
    if "source" not in out.columns:
        out["source"] = out["agency"] if "agency" in out.columns else "unknown"
    out = out.dropna(subset=["rating_date"])
    keep = [
        c for c in (
            "cik", "ticker", "rating_date", "rating", "notch", "agency",
            "source", "rating_score", "issuer_name", "lei",
        ) if c in out.columns
    ]
    dup_keys = [c for c in ("cik", "ticker", "rating_date", "agency") if c in out.columns]
    out = out[keep].drop_duplicates(subset=dup_keys or None)
    return out.sort_values("rating_date").reset_index(drop=True)


def load_ratings(path: Path | str) -> pd.DataFrame:
    """Liest das kanonische Ratings-Panel."""
    df = pd.read_csv(path)
    out = normalize_ratings_panel(df)
    logger.info(
        "Ratings geladen: %d Zeilen, %d CIKs, %s–%s.",
        len(out),
        int(out["cik"].nunique()) if "cik" in out.columns else 0,
        out["rating_date"].min().date() if len(out) else "—",
        out["rating_date"].max().date() if len(out) else "—",
    )
    return out


def attach_rating_labels(
    df: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    target: str = "speculative",
    horizon_months: int = 12,
    drop_unrated: bool = True,
    drop_censored: bool = True,
    asof_col: str = "reporting_date",
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Hängt ``label_rating`` PIT-sauber an Firm-Years.

    ``target='ordinal'``: Notch 1–21 zum Stichtag (Regressions-Label).
    ``target='speculative'``: HY-ähnlich zum Stichtag.
    ``target='downgrade'``: Notch fällt in ``(reporting_date, reporting_date+H]``.
    Unrated Firm-Years werden standardmäßig gedroppt (kein Target).
    """
    tgt = str(target).strip().lower()
    if tgt not in {"ordinal", "speculative", "downgrade", "hy", "ig_hy"}:
        raise ValueError(f"Unbekanntes rating-target: {target!r}")
    if tgt in {"hy", "ig_hy"}:
        tgt = "speculative"

    out = df.copy()
    if "cik" not in out.columns:
        raise ValueError("Firm-Years brauchen eine cik-Spalte für Rating-Labels.")
    asof = pd.to_datetime(out.get(asof_col), errors="coerce")
    n_no_date = int(asof.isna().sum())
    if n_no_date:
        logger.warning("%d Zeilen ohne %s — für Rating-Labels gedroppt.", n_no_date, asof_col)
    out = out.loc[asof.notna()].copy()
    asof = asof.loc[asof.notna()]
    out["_asof"] = asof
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce")
    out = out.dropna(subset=["cik"])
    out["cik"] = out["cik"].astype(int)

    panel = normalize_ratings_panel(ratings)
    if "cik" not in panel.columns:
        raise ValueError("Ratings-Panel braucht cik (Ticker vorher über die Map joinen).")
    panel["cik"] = pd.to_numeric(panel["cik"], errors="coerce")
    panel = panel.dropna(subset=["cik", "rating_date", "notch"])
    panel["cik"] = panel["cik"].astype(int)
    panel["notch"] = pd.to_numeric(panel["notch"], errors="coerce")
    panel = panel.dropna(subset=["notch"])
    panel = collapse_ratings_panel(panel)

    left = out.reset_index(drop=True)
    left["_row"] = range(len(left))
    right_cols = [c for c in ("cik", "rating_date", "rating", "notch", "agency") if c in panel.columns]
    right = panel[right_cols]
    right_by = {int(c): g for c, g in right.groupby("cik")}
    pieces: list[pd.DataFrame] = []
    for cik, gleft in left.groupby("cik", sort=False):
        gright = right_by.get(int(cik))
        gleft = gleft.sort_values("_asof")
        if gright is None or gright.empty:
            g = gleft.copy()
            g["rating_asof"] = pd.NA
            g["notch_asof"] = pd.NA
            g["rating_asof_date"] = pd.NaT
            g["agency_asof"] = pd.NA
            pieces.append(g)
            continue
        asof_cols = [c for c in ("rating_date", "rating", "notch", "agency") if c in gright.columns]
        merged = pd.merge_asof(
            gleft,
            gright[asof_cols].sort_values("rating_date"),
            left_on="_asof",
            right_on="rating_date",
            direction="backward",
        )
        pieces.append(
            merged.rename(
                columns={
                    "rating": "rating_asof",
                    "notch": "notch_asof",
                    "rating_date": "rating_asof_date",
                    "agency": "agency_asof",
                }
            )
        )
    joined = pd.concat(pieces, ignore_index=True)

    n_unrated = int(joined["notch_asof"].isna().sum())
    if drop_unrated:
        joined = joined.loc[joined["notch_asof"].notna()].copy()
        logger.info("Unrated gedroppt: %d Zeilen.", n_unrated)
    else:
        logger.info("Unrated behalten: %d Zeilen ohne as-of-Rating.", n_unrated)

    if tgt == "ordinal":
        joined[label_col] = pd.to_numeric(joined["notch_asof"], errors="coerce")
        joined = joined.drop(columns=["_asof", "_row"], errors="ignore")
        logger.info(
            "Rating-Label (ordinal): n=%d, mean_notch=%.2f, CIKs=%d.",
            len(joined),
            float(joined[label_col].mean()) if len(joined) else float("nan"),
            joined["cik"].nunique() if len(joined) else 0,
        )
        return joined.reset_index(drop=True)

    if tgt == "speculative":
        joined[label_col] = joined["notch_asof"].map(lambda x: int(is_speculative(x)) if pd.notna(x) else 0)
    else:
        horizon_end = joined["_asof"] + pd.DateOffset(months=int(horizon_months))
        joined["_horizon_end"] = horizon_end
        fut = joined[["_row", "cik", "_asof", "_horizon_end"]].merge(
            panel[["cik", "rating_date", "notch"]],
            on="cik",
            how="left",
        )
        in_win = (fut["rating_date"] > fut["_asof"]) & (fut["rating_date"] <= fut["_horizon_end"])
        min_fut = fut.loc[in_win].groupby("_row")["notch"].min()
        joined["notch_future_min"] = joined["_row"].map(min_fut)
        down = (
            joined["notch_asof"].notna()
            & joined["notch_future_min"].notna()
            & (joined["notch_future_min"] < joined["notch_asof"])
        )
        joined[label_col] = down.fillna(False).astype(int)
        if drop_censored and not panel.empty:
            cik_max = panel.groupby("cik")["rating_date"].max()
            joined["_cik_max"] = joined["cik"].map(cik_max)
            censored = (
                joined["_cik_max"].notna()
                & (joined["_horizon_end"] > joined["_cik_max"])
                & (joined[label_col] == 0)
            )
            n_cens = int(censored.sum())
            joined = joined.loc[~censored].copy()
            logger.info("Rechtszensierung (Rating-Coverage): %d Zeilen gedroppt.", n_cens)
            joined = joined.drop(columns=["_cik_max"], errors="ignore")
        joined = joined.drop(columns=["_horizon_end", "notch_future_min"], errors="ignore")

    joined = joined.drop(columns=["_asof", "_row"], errors="ignore")
    pos = float(joined[label_col].mean()) if len(joined) else 0.0
    logger.info(
        "Rating-Label (%s): n=%d, positiv=%.2f%%, CIKs=%d.",
        tgt,
        len(joined),
        100 * pos,
        joined["cik"].nunique() if len(joined) else 0,
    )
    return joined.reset_index(drop=True)
