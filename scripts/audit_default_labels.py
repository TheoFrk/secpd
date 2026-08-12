#!/usr/bin/env python3
"""Audit der Default-Labels: Regime, Accession, Überschneidung mit Alt-Regime.

Beispiel:
  python scripts/audit_default_labels.py \\
    --data data/processed/zenodo_labeled.csv.gz \\
    --events data/raw/edgar_8k_events.csv \\
    --out data/processed/default_label_audit.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from secpd.data.events import (  # noqa: E402
    REGIME_SWITCH,
    attach_default_labels,
    bankruptcy_dates,
    load_events,
    mark_concepts,
)
from secpd.data.zenodo import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Default-Labels / Bankruptcy-Events")
    p.add_argument("--data", required=True, help="Firm-Year-Datensatz")
    p.add_argument("--events", required=True, help="edgar_8k_events.csv")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--out", default="data/processed/default_label_audit.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    events = load_events(args.events)
    marked = mark_concepts(events, trust_legacy_regime=True)
    bk_all = marked.loc[marked["is_bankruptcy"]].copy()
    bk_all["regime"] = bk_all["filing_date_8k"].map(
        lambda d: "new" if d >= REGIME_SWITCH else "legacy"
    )

    n_legacy_evt = int((bk_all["regime"] == "legacy").sum())
    n_new_evt = int((bk_all["regime"] == "new").sum())
    print(f"Bankruptcy-Events: {len(bk_all)}  (legacy={n_legacy_evt}, new={n_new_evt})")
    print(f"  unique CIKs legacy={bk_all.loc[bk_all['regime']=='legacy','cik'].nunique()}  "
          f"new={bk_all.loc[bk_all['regime']=='new','cik'].nunique()}")

    df = load_dataset(args.data)
    labeled_legacy = attach_default_labels(
        df, events, horizon_months=args.horizon, trust_legacy_regime=True
    )
    labeled_clean = attach_default_labels(
        df, events, horizon_months=args.horizon, trust_legacy_regime=False
    )
    pos_legacy = int(labeled_legacy["label_default"].sum())
    pos_clean = int(labeled_clean["label_default"].sum())
    print(f"Positive Labels (Horizont {args.horizon}M):")
    print(f"  trust_legacy=True : {pos_legacy}")
    print(f"  trust_legacy=False: {pos_clean}  "
          f"(Δ={pos_legacy - pos_clean}, "
          f"{100 * (pos_legacy - pos_clean) / pos_legacy if pos_legacy else 0:.1f}% der alten Positives)")

    bk_dates_legacy = bankruptcy_dates(events, trust_legacy_regime=True)
    bk_dates_clean = bankruptcy_dates(events, trust_legacy_regime=False)

    pos = labeled_legacy.loc[labeled_legacy["label_default"] == 1].copy()
    pos["bankruptcy_date_legacy"] = pos["cik"].map(bk_dates_legacy)
    pos["bankruptcy_date_clean"] = pos["cik"].map(bk_dates_clean)
    pos["legacy_only"] = pos["bankruptcy_date_clean"].isna() & pos["bankruptcy_date_legacy"].notna()
    # Enrich with first matching bankruptcy accession (legacy-aware)
    first_bk = (
        bk_all.sort_values("filing_date_8k")
        .groupby("cik", as_index=False)
        .first()[["cik", "filing_date_8k", "items", "accession", "regime"]]
        .rename(columns={
            "filing_date_8k": "first_bk_date",
            "items": "first_bk_items",
            "accession": "first_bk_accession",
            "regime": "first_bk_regime",
        })
    )
    audit = pos.merge(first_bk, on="cik", how="left")
    keep = [
        c for c in (
            "cik", "name", "fyear", "doc_id", "reporting_date", "filing_date",
            "label_default", "bankruptcy_date_legacy", "bankruptcy_date_clean",
            "legacy_only", "first_bk_date", "first_bk_regime",
            "first_bk_items", "first_bk_accession",
        )
        if c in audit.columns
    ]
    audit = audit[keep].sort_values(["legacy_only", "cik", "fyear"], ascending=[False, True, True])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    print(f"Audit geschrieben: {out}  ({len(audit)} positive Zeilen, "
          f"{int(audit['legacy_only'].sum())} legacy-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
