#!/usr/bin/env python3
"""Trainings-Entry-Point für SEC-PD.

Ablauf: Datensatz laden → Spalten auflösen → Finanz-Features → (optional)
LLM-Text-Features → leakage-bewusster Split → Training → Evaluation →
joblib-Persistierung. Die Financial-Baseline wird IMMER mittrainiert, damit
der Mehrwert der Text-Features quantifizierbar bleibt.

Beispiele
---------
# Nur Finanz-Baseline (synthetische Daten):
python train.py --data data/processed/synthetic.csv --mode financial

# Option A — Text-Features als Spalten im selben Modell (Mock-LLM):
python train.py --data data/processed/synthetic.csv --mode combined --llm mock

# Option B — separates Text-Modell + Logit-Ensemble:
python train.py --data data/processed/synthetic.csv --mode ensemble --w-fin 0.6

# Auf dem Bank-Server identisch, nur: SECPD_LLM_MODE=bank (plus Endpoint/Key).
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

# src/-Layout ohne Installation nutzbar machen (pip install -e . ist der saubere Weg).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from secpd.data.events import (  # noqa: E402
    attach_default_labels,
    add_event_features,
    load_events,
)
from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.evaluation import evaluate_probs, print_report  # noqa: E402
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import extract_text_features, text_feature_names  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.ensemble import EnsembleWeights, combine_probabilities  # noqa: E402
from secpd.models.persistence import (  # noqa: E402
    ModelBundle,
    bundle_filename,
    runtime_metadata,
    save_ensemble,
    save_single,
)
from secpd.models.pipeline import build_pipeline, build_text_pipeline  # noqa: E402
from secpd.splitting import smart_split  # noqa: E402

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD Training")
    p.add_argument("--data", required=True, help="Pfad zum Datensatz (.csv/.csv.gz/.parquet)")
    p.add_argument("--financials", default=None,
                   help="Optionales Finanz-Panel (CSV mit cik,fyear,…) zum Mergen, s. scripts/fetch_edgar_financials.py")
    p.add_argument("--events", default=None,
                   help="Optionale 8-K-Eventliste (CSV), s. scripts/fetch_edgar_events.py — "
                        "ergänzt PIT-saubere evt_*-Features")
    p.add_argument("--label-source", choices=["fraud", "default"], default="fraud",
                   help="fraud = AAER-Label des Datensatzes | default = Insolvenz-Label "
                        "aus 8-K Item 1.03 (erfordert --events)")
    p.add_argument("--default-horizon", type=int, default=12,
                   help="Prognosehorizont des Default-Labels in Monaten")
    p.add_argument("--mode", choices=["financial", "combined", "ensemble"], default="financial")
    p.add_argument("--llm", choices=["mock", "bank"], default=None,
                   help="Überschreibt SECPD_LLM_MODE (Default: ENV bzw. mock)")
    p.add_argument("--out", default="models", help="Ausgabeverzeichnis für .joblib-Bundles")
    p.add_argument("--label-col", default=None)
    p.add_argument("--id-col", default=None)
    p.add_argument("--text-col", default=None)
    p.add_argument("--year-col", default=None)
    p.add_argument("--group-col", default="cik")
    p.add_argument("--split", choices=["auto", "temporal", "group", "random"], default="auto")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--calibrate", action="store_true",
                   help="Isotonic-Kalibrierung (CV=3) für kalibrierte PD-Niveaus")
    p.add_argument("--w-fin", type=float, default=0.6, help="Ensemble-Gewicht Finanzmodell")
    p.add_argument("--max-chars", type=int, default=12_000, help="LLM-Input-Trunkierung")
    p.add_argument("--sample", type=int, default=None,
                   help="Nur die ersten N Zeilen (schnelle Iteration / LLM-Kosten begrenzen)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out_dir = Path(args.out)

    # ------------------------------------------------------------------ #
    # 1) Daten laden & Spalten auflösen
    # ------------------------------------------------------------------ #
    df = load_dataset(args.data)
    if args.sample:
        df = df.head(args.sample).copy()
    if args.financials:
        panel = pd.read_csv(args.financials)
        panel.columns = [c.lower() for c in panel.columns]
        df = df.merge(panel, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
        logger.info("Finanz-Panel gemerged: %d Zeilen nach Merge.", len(df))

    events_df = load_events(args.events) if args.events else None
    if args.label_source == "default":
        if events_df is None:
            logger.error("--label-source default erfordert --events "
                         "(s. scripts/fetch_edgar_events.py).")
            return 2
        df = attach_default_labels(df, events_df, horizon_months=args.default_horizon)
        if args.label_col is None:
            args.label_col = "label_default"

    cols = resolve_columns(
        df,
        label_col=args.label_col,
        id_col=args.id_col,
        text_col=args.text_col,
        year_col=args.year_col,
    )
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2) Feature Engineering
    # ------------------------------------------------------------------ #
    df, fin_features = add_financial_features(df)
    if not fin_features:
        logger.error("Ohne Finanz-Features kein Training — ggf. --financials angeben.")
        return 2

    evt_features: list[str] = []
    if events_df is not None:
        df, evt_features = add_event_features(df, events_df)
    numeric_features = fin_features + evt_features

    llm_features: list[str] = []
    if args.mode in ("combined", "ensemble"):
        if cols.text_col is None:
            logger.error("Modus %r benötigt eine Textspalte (--text-col).", args.mode)
            return 2
        client = get_llm_client(args.llm)
        text_feats = extract_text_features(
            df,
            client=client,
            text_col=cols.text_col,
            id_col=cols.id_col,
            max_chars=args.max_chars,
        )
        df = df.merge(text_feats, on=cols.id_col, how="left")
        llm_features = text_feature_names()

    # ------------------------------------------------------------------ #
    # 3) Leakage-bewusster Split
    # ------------------------------------------------------------------ #
    group_col = args.group_col if args.group_col in df.columns else None
    train_idx, test_idx, used_strategy = smart_split(
        df,
        label_col=cols.label_col,
        group_col=group_col,
        year_col=cols.year_col,
        test_size=args.test_size,
        strategy=args.split,
        random_state=args.seed,
    )
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train = df_train[cols.label_col].astype(int).to_numpy()
    y_test = df_test[cols.label_col].astype(int).to_numpy()
    logger.info(
        "Split (%s): Train n=%d (%.1f%% pos) | Test n=%d (%.1f%% pos)",
        used_strategy, len(df_train), 100 * y_train.mean(), len(df_test), 100 * y_test.mean(),
    )

    # ------------------------------------------------------------------ #
    # 4) Training & Evaluation
    # ------------------------------------------------------------------ #
    results: dict[str, dict[str, float]] = {}
    train_run_id = uuid.uuid4().hex[:12]
    base_meta = {
        "mode": args.mode,
        "split_strategy": used_strategy,
        "label_col": cols.label_col,
        "label_source": args.label_source,
        "default_horizon_months": args.default_horizon if args.label_source == "default" else None,
        "event_features": evt_features,
        "calibrated": bool(args.calibrate),
        "data": str(args.data),
        "train_run_id": train_run_id,
    }

    # 4a) Financial-Baseline — immer, als Referenzpunkt für den Text-Mehrwert.
    fin_pipe = build_pipeline(
        numeric_features,
        n_estimators=args.n_estimators,
        calibrate=args.calibrate,
        random_state=args.seed,
    )
    fin_pipe.fit(df_train, y_train)
    p_fin_test = fin_pipe.predict_proba(df_test)[:, 1]
    results["financial_baseline"] = evaluate_probs(y_test, p_fin_test)
    fin_bundle = ModelBundle(
        pipeline=fin_pipe,
        feature_cols=numeric_features,
        metadata=runtime_metadata({**base_meta, "component": "financial",
                                   "metrics": results["financial_baseline"]}),
    )
    fin_name = bundle_filename(
        "financial",
        label_source=args.label_source,
        horizon_months=args.default_horizon if args.label_source == "default" else None,
    )
    save_single(fin_bundle, out_dir / fin_name)

    report_focus = ("financial_baseline", p_fin_test)

    if args.mode == "combined":
        # 4b) Option A: Finanz- + LLM-Features im selben Modell.
        combined_features = numeric_features + llm_features
        comb_pipe = build_pipeline(
            combined_features,
            n_estimators=args.n_estimators,
            calibrate=args.calibrate,
            random_state=args.seed,
        )
        comb_pipe.fit(df_train, y_train)
        p_comb_test = comb_pipe.predict_proba(df_test)[:, 1]
        results["combined_model"] = evaluate_probs(y_test, p_comb_test)
        comb_name = bundle_filename(
            "combined",
            label_source=args.label_source,
            horizon_months=args.default_horizon if args.label_source == "default" else None,
        )
        save_single(
            ModelBundle(
                pipeline=comb_pipe,
                feature_cols=combined_features,
                metadata=runtime_metadata({**base_meta, "component": "combined",
                                           "llm_features": llm_features,
                                           "metrics": results["combined_model"]}),
            ),
            out_dir / comb_name,
        )
        report_focus = ("combined_model", p_comb_test)

    elif args.mode == "ensemble":
        # 4b') Option B: separates Text-Modell + Logit-Ensemble.
        text_pipe = build_text_pipeline(llm_features, random_state=args.seed)
        text_pipe.fit(df_train, y_train)
        p_text_test = text_pipe.predict_proba(df_test)[:, 1]
        results["text_only"] = evaluate_probs(y_test, p_text_test)

        weights = EnsembleWeights(w_financial=args.w_fin, w_text=1.0 - args.w_fin)
        p_ens_test = combine_probabilities(p_fin_test, p_text_test, weights, method="logit")
        results[f"ensemble(w_fin={args.w_fin:.2f})"] = evaluate_probs(y_test, p_ens_test)

        text_bundle = ModelBundle(pipeline=text_pipe, feature_cols=llm_features, metadata={})
        ens_name = bundle_filename(
            "ensemble",
            label_source=args.label_source,
            horizon_months=args.default_horizon if args.label_source == "default" else None,
        )
        save_ensemble(
            fin_bundle,
            text_bundle,
            weights={"w_financial": weights.w_financial, "w_text": weights.w_text},
            path=out_dir / ens_name,
            metadata=runtime_metadata({**base_meta,
                                       "metrics": results[f"ensemble(w_fin={args.w_fin:.2f})"]}),
        )
        report_focus = (f"ensemble(w_fin={args.w_fin:.2f})", p_ens_test)

    # ------------------------------------------------------------------ #
    # 5) Report
    # ------------------------------------------------------------------ #
    focus_name, focus_probs = report_focus
    print_report(results, show_deciles_for=focus_name, y_true=y_test, p=focus_probs)
    logger.info("Fertig. Artefakte unter: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
