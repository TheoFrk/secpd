#!/usr/bin/env python3
"""Erzeugt den synthetischen Demo-Datensatz (kanonisches Schema + MD&A-Texte).

Beispiel:
    python scripts/make_synthetic_data.py --n 1200 --out data/processed/synthetic.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.data.synthetic import make_synthetic_dataset, make_synthetic_events, make_synthetic_ratings  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Synthetische Demo-Daten")
    p.add_argument("--n", type=int, default=1_200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed/synthetic.csv")
    p.add_argument("--events-out", default=None,
                   help="Optional: passende synthetische 8-K-Eventliste (CSV) für "
                        "den Default-Label-Workflow schreiben")
    p.add_argument("--ratings-out", default=None,
                   help="Optional: synthetische Rating-Historie für --label-source rating")
    args = p.parse_args()

    df = make_synthetic_dataset(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("Geschrieben: %s — %d Zeilen, Basisrate %.2f%%",
                 out, len(df), 100 * df["label"].mean())

    if args.events_out:
        events = make_synthetic_events(df, seed=args.seed)
        ev_out = Path(args.events_out)
        ev_out.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(ev_out, index=False)
        logging.info("Geschrieben: %s — %d 8-Ks für %d CIKs",
                     ev_out, len(events), events["cik"].nunique())

    if args.ratings_out:
        ratings = make_synthetic_ratings(df, seed=args.seed)
        rt_out = Path(args.ratings_out)
        rt_out.parent.mkdir(parents=True, exist_ok=True)
        ratings.to_csv(rt_out, index=False)
        logging.info("Geschrieben: %s — %d Ratings für %d CIKs",
                     rt_out, len(ratings), ratings["cik"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
