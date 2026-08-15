#!/usr/bin/env python3
"""Baut das Finanz-Panel via SEC-XBRL companyfacts. Nur Home-Setup (Internet).

Liest die CIKs aus dem konvertierten Zenodo-Datensatz und schreibt ein CSV im
kanonischen Schema (cik, fyear, total_assets, …), das committet und auf dem
Bank-Server per ``train.py --financials …`` gemerged wird.

Beispiel:
    export SECPD_SEC_UA="Commerzbank Praktikum vorname.nachname@example.com"
    python scripts/fetch_edgar_financials.py \
        --dataset data/processed/zenodo_labeled.csv.gz \
        --out data/raw/financials_panel.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from secpd.config import load_settings  # noqa: E402
from secpd.data.edgar import build_financials_panel  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="EDGAR-Finanzpanel (companyfacts)")
    p.add_argument("--dataset", required=True, help="Datensatz mit cik-Spalte")
    p.add_argument("--out", default="data/raw/financials_panel.csv")
    p.add_argument("--max-ciks", type=int, default=None, help="Obergrenze (Testläufe)")
    p.add_argument("--ua", default=None, help="SEC User-Agent (sonst SECPD_SEC_UA)")
    p.add_argument(
        "--cache-dir",
        default="data/raw/edgar_companyfacts",
        help="JSON-Cache je CIK (Restart ohne erneuten Download)",
    )
    p.add_argument(
        "--existing",
        default=None,
        help="Vorhandenes Panel: diese CIKs nicht erneut abfragen, ans Out mergen",
    )
    args = p.parse_args()

    ua = args.ua or load_settings().sec_user_agent
    if not ua:
        logging.error("SECPD_SEC_UA fehlt — SEC verlangt einen User-Agent.")
        return 2
    df = pd.read_csv(args.dataset, usecols=lambda c: c.lower() == "cik")
    ciks = sorted(pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int).unique())
    if args.max_ciks:
        ciks = ciks[: args.max_ciks]

    existing = pd.DataFrame()
    if args.existing:
        existing = pd.read_csv(args.existing)
        existing.columns = [c.lower() for c in existing.columns]
        have = set(pd.to_numeric(existing["cik"], errors="coerce").dropna().astype(int))
        before = len(ciks)
        ciks = [c for c in ciks if c not in have]
        logging.info("Überspringe %d CIKs aus %s — noch %d neu.", before - len(ciks), args.existing, len(ciks))

    logging.info("Frage companyfacts für %d CIKs ab …", len(ciks))
    panel = build_financials_panel(ciks, user_agent=ua, cache_dir=args.cache_dir) if ciks else pd.DataFrame()
    if not existing.empty:
        panel = pd.concat([existing, panel], ignore_index=True)
        panel = panel.drop_duplicates(subset=["cik", "fyear"], keep="first").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    logging.info("Panel geschrieben: %s (%d Zeilen, %d CIKs)", out, len(panel), panel["cik"].nunique() if len(panel) else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
