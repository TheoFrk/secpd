#!/usr/bin/env python3
"""Konvertiert die Zenodo-Rohdaten in einen kompakten Modellierungs-Datensatz.

Streamt ``firm_years_labels.json`` (bzw. ``firm_years.json``), trunkiert die
MD&A-Texte, konstruiert das Fraud-Label über den AAER-Join und schreibt ein
``.csv.gz`` — klein genug, um es (ggf. via git-lfs) zu committen und offline
auf dem Bank-Server zu verwenden.

Beispiel
--------
python scripts/convert_zenodo.py \
    --firm-years data/raw/firm_years_labels.json \
    --aaer data/raw/aaer_mark5.csv \
    --out data/processed/zenodo_labeled.csv.gz \
    --truncate-chars 40000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.data.zenodo import attach_fraud_labels, firm_years_to_frame, load_aaer  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Zenodo-Konverter")
    p.add_argument("--firm-years", required=True, help="firm_years[_labels].json")
    p.add_argument("--aaer", required=True, help="aaer_mark5.csv")
    p.add_argument("--out", required=True, help="Ziel, z. B. data/processed/zenodo_labeled.csv.gz")
    p.add_argument("--max-records", type=int, default=None, help="Nur die ersten N Records (Testläufe)")
    p.add_argument("--truncate-chars", type=int, default=40_000)
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--include-non-fsf", action="store_true",
                   help="Auch AAERs ohne Financial-Statement-Fraud-Flag als positiv werten")
    p.add_argument("--keep-revoked", action="store_true")
    args = p.parse_args()

    firm_years = firm_years_to_frame(
        args.firm_years,
        max_records=args.max_records,
        min_text_chars=args.min_text_chars,
        truncate_chars=args.truncate_chars,
    )
    if firm_years.empty:
        logging.error("Keine verwertbaren Firm-Year-Records gefunden.")
        return 2

    aaer = load_aaer(args.aaer)
    labeled = attach_fraud_labels(
        firm_years,
        aaer,
        require_fsf=not args.include_non_fsf,
        drop_revoked=not args.keep_revoked,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out, index=False)  # pandas erkennt .gz an der Endung
    logging.info(
        "Geschrieben: %s — %d Zeilen, Basisrate %.2f%%",
        out, len(labeled), 100 * labeled["label"].mean(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
