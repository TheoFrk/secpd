#!/usr/bin/env python3
"""Inference-Entry-Point: gespeicherte Bundles auf neue Daten anwenden.

Rekonstruiert dieselben Features wie ``train.py`` (Finanzkennzahlen,
bei Bedarf LLM-Text-Features via Cache/Client) und schreibt eine
Score-Datei ``doc_id,pd_score``.

Beispiel
--------
python predict.py --model models/combined_model.joblib \
    --data data/processed/synthetic.csv --out scores.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from secpd.data.events import add_event_features, load_events  # noqa: E402
from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import extract_text_features, text_feature_names  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.ensemble import EnsembleWeights, combine_probabilities  # noqa: E402
from secpd.models.persistence import (  # noqa: E402
    BUNDLE_KIND_ENSEMBLE,
    BUNDLE_KIND_SINGLE,
    load_any,
)

logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD Scoring")
    p.add_argument("--model", required=True, help="Pfad zum .joblib-Bundle")
    p.add_argument("--data", required=True, help="Datensatz (.csv/.csv.gz/.parquet)")
    p.add_argument("--out", default="scores.csv")
    p.add_argument("--llm", choices=["mock", "bank", "lmstudio"], default=None)
    p.add_argument("--events", default=None,
                   help="8-K-Eventliste (CSV) — nötig, wenn das Bundle evt_*-Features erwartet")
    p.add_argument("--id-col", default=None)
    p.add_argument("--text-col", default=None)
    p.add_argument("--max-chars", type=int, default=12_000)
    return p.parse_args()


def _needs_llm(feature_cols: list[str]) -> bool:
    return any(c in feature_cols for c in text_feature_names())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    payload = load_any(args.model)
    df = load_dataset(args.data)

    # Label ist beim Scoring optional — Dummy einsetzen, falls abwesend.
    if not any(c in df.columns for c in ("label", "misstate", "fraud", "target", "y")):
        df["label"] = 0
    cols = resolve_columns(df, id_col=args.id_col, text_col=args.text_col)
    df = df.reset_index(drop=True)

    df, _ = add_financial_features(df)

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        feature_cols = list(payload["feature_cols"])
        needs_llm = _needs_llm(feature_cols)
    elif payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        feature_cols = list(payload["financial"]["feature_cols"])
        needs_llm = True
    else:
        raise ValueError(f"Unbekannter Bundle-Typ: {payload['kind']!r}")

    needs_events = any(str(c).startswith("evt_") for c in feature_cols)
    if needs_events:
        if not args.events:
            logger.error(
                "Bundle erwartet evt_*-Features — bitte --events angeben "
                "(s. scripts/fetch_edgar_events.py)."
            )
            return 2
        df, _ = add_event_features(df, load_events(args.events))

    if needs_llm:
        if cols.text_col is None:
            logger.error("Bundle benötigt Text-Features, aber keine Textspalte gefunden.")
            return 2
        client = get_llm_client(args.llm)
        text_feats = extract_text_features(
            df, client=client, text_col=cols.text_col, id_col=cols.id_col,
            max_chars=args.max_chars,
        )
        df = df.merge(text_feats, on=cols.id_col, how="left")

    all_needed: list[str] = list(feature_cols)
    if payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        all_needed = list(
            dict.fromkeys(
                list(payload["financial"]["feature_cols"])
                + list(payload["text"]["feature_cols"])
            )
        )
    for c in all_needed:
        if c not in df.columns:
            df[c] = float("nan")

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        scores = payload["pipeline"].predict_proba(df)[:, 1]
    else:
        p_fin = payload["financial"]["pipeline"].predict_proba(df)[:, 1]
        p_txt = payload["text"]["pipeline"].predict_proba(df)[:, 1]
        w = payload.get("weights", {})
        weights = EnsembleWeights(
            w_financial=float(w.get("w_financial", 0.6)),
            w_text=float(w.get("w_text", 0.4)),
        )
        scores = combine_probabilities(p_fin, p_txt, weights, method="logit")

    out = pd.DataFrame({cols.id_col: df[cols.id_col], "pd_score": scores})
    out.to_csv(args.out, index=False)
    logger.info("Scores geschrieben: %s (%d Zeilen)", args.out, len(out))
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
