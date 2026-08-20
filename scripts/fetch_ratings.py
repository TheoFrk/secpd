#!/usr/bin/env python3
"""Lädt Issuer-Ratings und schreibt ein PIT-fähiges Panel.

Default-Quelle: **ratingshistory.info** (SEC 17g-7: Moody's, Fitch,
Egan-Jones) — Bulk-CSV, kein API-Limit. Optional FMP-Fundamentalnoten
(``--source fmp`` / ``both``, Free-Tier ~250 Calls/Tag).

Ticker/Namen/LEI kommen aus dem EDGAR-Submissions-Cache, optional ergänzt
um SEC ``company_tickers.json``.

Beispiele
---------
python scripts/fetch_ratings.py \\
    --dataset data/processed/zenodo_labeled.csv.gz \\
    --out data/raw/ratings_panel.csv

# FMP ergänzen (nur Netz-Hits zählen):
python scripts/fetch_ratings.py --source both --max-requests 250

# Nur Cache zusammensetzen (kein Netz):
python scripts/fetch_ratings.py --cache-only --out data/raw/ratings_panel.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from secpd.data.ratings import (  # noqa: E402
    RatingsFetchError,
    build_fmp_ratings_panel,
    build_nrsro_ratings_panel,
    fetch_sec_company_tickers,
    merge_ticker_maps,
    normalize_ratings_panel,
    ticker_map_from_submissions,
)

logger = logging.getLogger("fetch_ratings")
SECRETS_FILE = ROOT / ".secpd.env"


def _load_secrets() -> None:
    if not SECRETS_FILE.exists():
        return
    for raw in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issuer-Ratings via ratingshistory.info (NRSRO) und/oder FMP"
    )
    p.add_argument(
        "--source",
        choices=["nrsro", "fmp", "both"],
        default="nrsro",
        help="nrsro = 17g-7 Bulk (Default) | fmp = Fundamentalnote | both",
    )
    p.add_argument("--dataset", default=None, help="Datensatz mit cik (filtert/priorisiert die Map)")
    p.add_argument("--out", default="data/raw/ratings_panel.csv")
    p.add_argument("--tickers-out", default="data/raw/company_tickers.csv")
    p.add_argument("--submissions-cache", default="data/raw/edgar_submissions")
    p.add_argument("--nrsro-cache", default="data/raw/nrsro_ratings")
    p.add_argument("--fmp-cache", default="data/raw/fmp_ratings")
    p.add_argument("--sec-tickers-cache", default="data/raw/company_tickers.json")
    p.add_argument("--no-sec-tickers", action="store_true", help="SEC company_tickers.json nicht laden")
    p.add_argument("--agencies", default="moodys,fitch,egan-jones")
    p.add_argument("--categories", default="Corporate,Financial")
    p.add_argument("--limit", type=int, default=None, help="Nur die ersten N Ticker (FMP-Smoke-Test)")
    p.add_argument(
        "--max-requests",
        type=int,
        default=250,
        help="FMP: max. echte API-Calls (Cache-Hits zählen nicht). Default 250.",
    )
    p.add_argument("--sleep", type=float, default=0.25, help="Pause nach FMP-Netz-Request (s)")
    p.add_argument("--force", action="store_true", help="Caches ignorieren")
    p.add_argument("--cache-only", action="store_true", help="Nur vorhandene Caches, kein Netz")
    p.add_argument("--api-key", default=None, help="FMP-Key (sonst SECPD_FMP_API_KEY)")
    p.add_argument("--ua", default=None, help="SEC User-Agent (sonst SECPD_SEC_UA)")
    return p.parse_args()


def _prioritize_map(ticker_map: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    """Filtert auf Dataset-CIKs und sortiert nach Firm-Year-Häufigkeit."""
    if not dataset:
        return ticker_map
    ds = pd.read_csv(dataset, usecols=lambda c: str(c).lower() == "cik")
    ds.columns = [c.lower() for c in ds.columns]
    ciks = pd.to_numeric(ds["cik"], errors="coerce").dropna().astype(int)
    want = set(ciks)
    counts = ciks.value_counts()
    before = len(ticker_map)
    out = ticker_map.loc[ticker_map["cik"].isin(want)].copy()
    out["_n"] = out["cik"].map(counts).fillna(0)
    out = out.sort_values("_n", ascending=False).drop(columns=["_n"])
    logger.info("Dataset-Filter: %d → %d CIKs (häufigste zuerst).", before, len(out))
    return out


def _combine_panels(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return normalize_ratings_panel(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _load_secrets()
    args = parse_args()

    ticker_map = ticker_map_from_submissions(args.submissions_cache)
    ua = args.ua or os.getenv("SECPD_SEC_UA", "")
    if not args.no_sec_tickers and ua:
        try:
            sec_map = fetch_sec_company_tickers(
                user_agent=ua,
                cache_path=args.sec_tickers_cache,
            )
            ticker_map = merge_ticker_maps(ticker_map, sec_map)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SEC company_tickers übersprungen: %s", exc)
    elif not args.no_sec_tickers and not ua:
        logger.info("Kein SECPD_SEC_UA — SEC company_tickers.json wird übersprungen.")

    ticker_map = _prioritize_map(ticker_map, args.dataset)

    tickers_out = Path(args.tickers_out)
    tickers_out.parent.mkdir(parents=True, exist_ok=True)
    ticker_map.to_csv(tickers_out, index=False)
    logger.info("Ticker-Map: %s (%d CIKs).", tickers_out, len(ticker_map))

    want_nrsro = args.source in {"nrsro", "both"}
    want_fmp = args.source in {"fmp", "both"}
    nrsro = pd.DataFrame()
    fmp = pd.DataFrame()

    if want_nrsro:
        agencies = tuple(a.strip() for a in args.agencies.split(",") if a.strip())
        categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
        try:
            nrsro = build_nrsro_ratings_panel(
                ticker_map,
                cache_dir=args.nrsro_cache,
                agencies=agencies,
                categories=categories,
                force=args.force,
                cache_only=bool(args.cache_only),
            )
        except RatingsFetchError as exc:
            logger.error("%s", exc)
            if not want_fmp:
                return 2

    if want_fmp:
        api_key = args.api_key or os.getenv("SECPD_FMP_API_KEY", "")
        if not args.cache_only and not api_key:
            logger.error(
                "SECPD_FMP_API_KEY fehlt. Kostenlosen Key anlegen und in .secpd.env "
                "setzen, oder --cache-only / --source nrsro."
            )
            if nrsro.empty:
                return 2
            logger.warning("FMP übersprungen — NRSRO-Panel bleibt.")
        else:
            try:
                fmp = build_fmp_ratings_panel(
                    ticker_map,
                    api_key=api_key,
                    cache_dir=args.fmp_cache,
                    sleep_s=args.sleep,
                    force=args.force,
                    cache_only=bool(args.cache_only),
                    limit=args.limit,
                    max_requests=args.max_requests,
                )
            except RatingsFetchError as exc:
                logger.error("%s", exc)
                if nrsro.empty:
                    return 2

    panel = _combine_panels(nrsro, fmp)
    if panel.empty:
        logger.error("Ratings-Panel ist leer — Quelle, Cache oder Ticker-Map prüfen.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    logger.info(
        "Geschrieben: %s (%d Zeilen, %d CIKs, %s–%s)",
        out,
        len(panel),
        panel["cik"].nunique(),
        panel["rating_date"].min().date(),
        panel["rating_date"].max().date(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
