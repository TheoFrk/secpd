#!/usr/bin/env python3
"""Lädt 8-K-Events (Metadaten) via EDGAR-Submissions-API. Nur Home-Setup.

Das Insolvenz-Signal (Item 1.03 bzw. altes Item 3) steht bereits in den
Filing-Metadaten — Volltexte sind für Label und Event-Features nicht nötig.
Roh-JSONs werden unter ``data/raw/edgar_submissions/`` gecacht; der Lauf ist
damit unterbrech- und wiederaufnehmbar (bereits geladene CIKs werden
übersprungen).

Beispiele
---------
# CIKs aus dem konvertierten Zenodo-Datensatz:
export SECPD_SEC_UA="Commerzbank Praktikum vorname.nachname@example.com"
python scripts/fetch_edgar_events.py \\
    --dataset data/processed/zenodo_labeled.csv.gz \\
    --out data/raw/edgar_8k_events.csv

# Gezielte CIK-Liste (z. B. Funktionstest):
python scripts/fetch_edgar_events.py --ciks 806085 320193 \\
    --out data/raw/edgar_8k_events.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from secpd.data.events import build_events_table, log_item_coverage  # noqa: E402

logger = logging.getLogger("fetch_edgar_events")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="8-K-Events via EDGAR-Submissions-API")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Datensatz mit cik-Spalte (CSV/CSV.GZ/Parquet)")
    src.add_argument("--ciks", nargs="+", type=int, help="Explizite CIK-Liste")
    p.add_argument("--out", default="data/raw/edgar_8k_events.csv")
    p.add_argument("--ua", default=None, help="SEC User-Agent (sonst SECPD_SEC_UA)")
    p.add_argument("--cache-dir", default="data/raw/edgar_submissions")
    p.add_argument("--limit", type=int, default=None, help="Nur die ersten N CIKs")
    p.add_argument("--sleep", type=float, default=0.15, help="Pause zwischen Requests (s)")
    p.add_argument("--force", action="store_true", help="Cache ignorieren, neu laden")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    ua = args.ua or os.getenv("SECPD_SEC_UA", "")
    if not ua:
        logger.error(
            "SEC verlangt einen User-Agent — SECPD_SEC_UA setzen oder --ua nutzen, "
            'z. B. "Commerzbank Praktikum vorname.nachname@example.com".'
        )
        return 2

    if args.ciks:
        ciks = args.ciks
    else:
        df = pd.read_csv(args.dataset, usecols=lambda c: str(c).lower() == "cik")
        df.columns = [c.lower() for c in df.columns]
        ciks = (
            pd.to_numeric(df["cik"], errors="coerce").dropna().astype(int).unique().tolist()
        )
    if args.limit:
        ciks = sorted(set(ciks))[: args.limit]
    logger.info("Starte Fetch für %d CIKs (Cache: %s)", len(set(ciks)), args.cache_dir)

    events = build_events_table(
        ciks,
        user_agent=ua,
        cache_dir=args.cache_dir,
        sleep_s=args.sleep,
        force=args.force,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    logger.info("Geschrieben: %s (%d 8-Ks, %d CIKs)", out, len(events),
                events["cik"].nunique() if len(events) else 0)
    log_item_coverage(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
