#!/usr/bin/env python3
"""Exportiert SEC-PD als Aktienwert-Panel für die Bachelor-Praxis.

Schreibt ein PIT-sicheres Firm-Year-CSV, das über ``filing_date`` (plus 1 Tag
Lag, analog conservative_lag1) auf Daily-Preispanels gejoint werden kann.

Beispiel
--------
python scripts/export_aktienwert.py \\
  --data docs/demo/apple_10k.csv \\
  --ticker AAPL \\
  --model models/combined_default_h36.joblib \\
  --events data/raw/edgar_8k_events_full.csv \\
  --out docs/demo/apple_aktienwert.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from secpd.equity.overlay import attach_equity_scores  # noqa: E402
from secpd.equity.scoring import score_firm_years  # noqa: E402

logger = logging.getLogger("export_aktienwert")

EXPORT_COLS = [
    "doc_id",
    "cik",
    "name",
    "yahoo_ticker",
    "fyear",
    "filing_date",
    "reporting_date",
    "asof_date",
    "pd_score",
    "equity_quality",
    "z_secpd_quality",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD → Aktienwert-Panel (S5)")
    p.add_argument("--data", required=True, help="Firm-Years CSV/JSON/CSV.GZ")
    p.add_argument("--model", default="models/combined_default_h36.joblib")
    p.add_argument("--events", default="data/raw/edgar_8k_events_full.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--ticker", default=None, help="Yahoo-Ticker für alle Zeilen (Demo: AAPL)")
    p.add_argument("--llm", default="openai")
    p.add_argument("--min-fyear", type=int, default=2009)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    events = args.events if Path(args.events).exists() else None
    if events is None:
        logger.warning("Events-Datei fehlt (%s) — Scoring ohne evt_*-Spalten.", args.events)

    df = score_firm_years(args.data, args.model, events=events, llm=args.llm)
    if args.min_fyear is not None and "fyear" in df.columns:
        df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(args.min_fyear)].copy()
    df = attach_equity_scores(df)

    filing = pd.to_datetime(df.get("filing_date"), errors="coerce")
    df["asof_date"] = (filing + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    if args.ticker:
        df["yahoo_ticker"] = str(args.ticker).strip().upper()
    elif "yahoo_ticker" not in df.columns:
        df["yahoo_ticker"] = pd.NA

    keep = [c for c in EXPORT_COLS if c in df.columns]
    extra = [c for c in ("total_assets", "fin_leverage", "fin_altman_z") if c in df.columns]
    out = df[keep + extra].sort_values(["cik", "fyear"] if "fyear" in df.columns else keep[:1])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("Aktienwert-Panel: %s (%d Zeilen)", path, len(out))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
