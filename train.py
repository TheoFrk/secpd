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
import os
import sys
import uuid
from pathlib import Path

# src/-Layout ohne Installation nutzbar machen (pip install -e . ist der saubere Weg).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from secpd.data.events import (  # noqa: E402
    MIN_FYEAR_WITH_FINANCIALS,
    attach_default_labels,
    add_event_features,
    feature_exclusions_for_labels,
    load_events,
    parse_label_concepts,
)
from secpd.data.ratings import attach_rating_labels, load_ratings  # noqa: E402
from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.evaluation import evaluate_ordinal, evaluate_probs, firm_overlap_stats, print_report  # noqa: E402
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import attach_text_features  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.ensemble import EnsembleWeights, combine_probabilities  # noqa: E402
from secpd.models.persistence import (  # noqa: E402
    ModelBundle,
    bundle_filename,
    runtime_metadata,
    save_ensemble,
    save_single,
)
from secpd.models.pipeline import (  # noqa: E402
    build_pipeline,
    build_regression_pipeline,
    build_text_pipeline,
    fit_pipeline,
    make_calibration_cv,
    predict_output,
)
from secpd.splitting import smart_split  # noqa: E402

logger = logging.getLogger("train")
SECRETS_FILE = Path(__file__).resolve().parent / ".secpd.env"


def _load_secrets() -> None:
    """Übernimmt Keys/Modus aus ``.secpd.env``, ohne vorhandene ENV zu überschreiben."""
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
    """CLI: Datenquellen, Label, Split, LLM, Kalibrierung, Persistenz."""
    p = argparse.ArgumentParser(description="SEC-PD Training")
    # Datenquellen
    p.add_argument("--data", required=True, help="Pfad zum Datensatz (.csv/.csv.gz/.parquet)")
    p.add_argument("--financials", default=None,
                   help="Optionales Finanz-Panel (CSV mit cik,fyear,…) zum Mergen, s. scripts/fetch_edgar_financials.py")
    p.add_argument("--events", default=None,
                   help="Optionale 8-K-Eventliste (CSV), s. scripts/fetch_edgar_events.py — "
                        "ergänzt PIT-saubere evt_*-Features")
    p.add_argument("--label-source", choices=["fraud", "default", "rating"], default="fraud",
                   help="fraud = AAER-Label | default = 8-K Item 1.03 | "
                        "rating = Agency/FMP-Panel (erfordert --ratings)")
    p.add_argument("--default-horizon", type=int, default=12,
                   help="Prognosehorizont in Monaten (default-Label und rating-downgrade)")
    p.add_argument(
        "--ratings",
        default=None,
        help="Ratings-Panel (CSV), s. scripts/fetch_ratings.py — für --label-source rating",
    )
    p.add_argument(
        "--rating-target",
        choices=["ordinal", "speculative", "downgrade"],
        default="ordinal",
        help="ordinal = Notch 1–21 (Default) | speculative = HY-ähnlich | "
             "downgrade = Notch fällt im Horizont",
    )
    p.add_argument(
        "--label-concepts",
        default="bankruptcy",
        help="Komma-Liste der 8-K-Konzepte fürs Default-Label "
             "(bankruptcy und/oder delisting). Label-Konzepte sind keine Features.",
    )
    p.add_argument(
        "--trust-legacy-regime",
        action="store_true",
        help="Auch vor 2004-08-23 Item-3/Alt-Codes als Bankruptcy/Events zählen "
             "(Default: aus — Submissions-Items vor dem Regimewechsel unzuverlässig)",
    )
    p.add_argument(
        "--min-fyear",
        type=int,
        default=None,
        help="Nur Firm-Years ab diesem GJ (Default bei --label-source default/rating: "
             f"{MIN_FYEAR_WITH_FINANCIALS}, sonst kein Cut)",
    )
    p.add_argument(
        "--require-financials",
        action="store_true",
        help="Zeilen ohne total_assets droppen (sinnvoll nach --financials-Merge)",
    )
    p.add_argument("--mode", choices=["financial", "combined", "ensemble"], default="financial")
    p.add_argument("--llm", choices=["mock", "bank", "lmstudio", "openai", "chatgpt"], default=None,
                   help="Überschreibt SECPD_LLM_MODE (Default: ENV bzw. mock)")
    p.add_argument(
        "--llm-refresh",
        action="store_true",
        help="Textfeatures neu via LLM bewerten (Cache überschreiben); "
             "Default: vorhandene Cache-Treffer nutzen",
    )
    p.add_argument(
        "--llm-cache-only",
        action="store_true",
        help="Keine LLM-API bei Cache-Miss (Fallback statt Request)",
    )
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
                   help="Wahrscheinlichkeits-Kalibrierung (Default-Methode: sigmoid)")
    p.add_argument("--calibrate-method", choices=["sigmoid", "isotonic", "auto"], default="auto",
                   help="auto: sigmoid bei <100 Trainings-Positiven, sonst isotonic")
    p.add_argument("--w-fin", type=float, default=0.6, help="Ensemble-Gewicht Finanzmodell")
    p.add_argument("--max-chars", type=int, default=12_000, help="LLM-Input-Trunkierung")
    p.add_argument("--sample", type=int, default=None,
                   help="Nur die ersten N Zeilen (schnelle Iteration / LLM-Kosten begrenzen)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _load_secrets()
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
    label_concepts: tuple[str, ...] = ("bankruptcy",)
    if args.label_source == "default":
        if events_df is None:
            logger.error("--label-source default erfordert --events "
                         "(s. scripts/fetch_edgar_events.py).")
            return 2
        label_concepts = parse_label_concepts(args.label_concepts)
        df = attach_default_labels(
            df,
            events_df,
            horizon_months=args.default_horizon,
            trust_legacy_regime=bool(args.trust_legacy_regime),
            label_concepts=label_concepts,
        )
        if args.label_col is None:
            args.label_col = "label_default"

    if args.label_source == "rating":
        if not args.ratings:
            logger.error("--label-source rating erfordert --ratings "
                         "(s. scripts/fetch_ratings.py).")
            return 2
        ratings_df = load_ratings(args.ratings)
        df = attach_rating_labels(
            df,
            ratings_df,
            target=args.rating_target,
            horizon_months=args.default_horizon,
        )
        if args.label_col is None:
            args.label_col = "label_rating"

    min_fyear = args.min_fyear
    if min_fyear is None and args.label_source in {"default", "rating"}:
        min_fyear = MIN_FYEAR_WITH_FINANCIALS
    if min_fyear is not None and "fyear" in df.columns:
        before = len(df)
        df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(min_fyear)].copy()
        logger.info(
            "min-fyear=%d: %d → %d Zeilen (%.1f%% behalten).",
            int(min_fyear), before, len(df),
            100 * len(df) / before if before else 0.0,
        )
    if args.require_financials:
        if "total_assets" not in df.columns:
            logger.error("--require-financials braucht --financials (Spalte total_assets).")
            return 2
        before = len(df)
        df = df.loc[df["total_assets"].notna()].copy()
        logger.info(
            "require-financials: %d → %d Zeilen mit total_assets.",
            before, len(df),
        )
    if df.empty:
        logger.error("Datensatz nach Filtern leer — min-fyear / Labels prüfen.")
        return 2

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
        exclude = (
            feature_exclusions_for_labels(label_concepts)
            if args.label_source == "default"
            else ()
        )
        df, evt_features = add_event_features(
            df,
            events_df,
            trust_legacy_regime=bool(args.trust_legacy_regime),
            exclude_concepts=exclude,
        )
    numeric_features = fin_features + evt_features

    llm_features: list[str] = []
    if args.mode in ("combined", "ensemble"):
        if cols.text_col is None:
            logger.error("Modus %r benötigt eine Textspalte (--text-col).", args.mode)
            return 2
        client = get_llm_client(
            args.llm,
            cached=True,
            force_refresh=bool(args.llm_refresh),
            cache_only=bool(args.llm_cache_only),
        )
        cache_dir = getattr(client, "cache_dir", None)
        if args.llm_refresh:
            logger.info("LLM: Force-Refresh aktiv — Cache wird überschrieben (%s)", client.name)
        elif args.llm_cache_only:
            logger.info(
                "LLM: Cache-only Replay (%s)%s",
                client.name,
                f" dir={cache_dir}" if cache_dir else "",
            )
        else:
            logger.info("LLM: Cache bevorzugt, keine neuen API-Calls bei Hits (%s)", client.name)
        df, llm_features = attach_text_features(
            df,
            client=client,
            text_col=cols.text_col,
            id_col=cols.id_col,
            max_chars=args.max_chars,
        )
        logger.info("Text-Features fürs Modell: %s", llm_features)

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
    is_ordinal = args.label_source == "rating" and str(args.rating_target).lower() == "ordinal"
    if is_ordinal:
        y_train = df_train[cols.label_col].astype(float).to_numpy()
        y_test = df_test[cols.label_col].astype(float).to_numpy()
        logger.info(
            "Split (%s): Train n=%d (mean_notch=%.2f) | Test n=%d (mean_notch=%.2f)",
            used_strategy, len(df_train), float(y_train.mean()),
            len(df_test), float(y_test.mean()),
        )
    else:
        y_train = df_train[cols.label_col].astype(int).to_numpy()
        y_test = df_test[cols.label_col].astype(int).to_numpy()
        logger.info(
            "Split (%s): Train n=%d (%.1f%% pos) | Test n=%d (%.1f%% pos)",
            used_strategy, len(df_train), 100 * y_train.mean(),
            len(df_test), 100 * y_test.mean(),
        )
    overlap = {"overlap_rate": float("nan")}
    if group_col is not None:
        overlap = firm_overlap_stats(df_train[group_col], df_test[group_col])
        logger.info(
            "Firm-Overlap: %.1f%% der Testfirmen auch im Training "
            "(%d/%d; neu=%d)",
            100 * overlap["overlap_rate"],
            int(overlap["n_overlap_firms"]),
            int(overlap["n_test_firms"]),
            int(overlap["n_new_test_firms"]),
        )

    # ------------------------------------------------------------------ #
    # 4) Training & Evaluation
    # ------------------------------------------------------------------ #
    results: dict[str, dict[str, float]] = {}
    train_run_id = uuid.uuid4().hex[:12]
    rating_horizon = (
        args.default_horizon
        if args.label_source == "rating" and args.rating_target == "downgrade"
        else None
    )
    bundle_horizon = args.default_horizon if args.label_source == "default" else rating_horizon
    if is_ordinal and args.calibrate:
        logger.info("--calibrate wird beim ordinalen Rating-Regressor ignoriert.")
    do_calibrate = bool(args.calibrate) and not is_ordinal
    task = "regression" if is_ordinal else "classification"
    base_meta = {
        "mode": args.mode,
        "split_strategy": used_strategy,
        "label_col": cols.label_col,
        "label_source": args.label_source,
        "task": task,
        "default_horizon_months": args.default_horizon if args.label_source == "default" else rating_horizon,
        "label_concepts": list(label_concepts) if args.label_source == "default" else None,
        "rating_target": args.rating_target if args.label_source == "rating" else None,
        "trust_legacy_regime": bool(args.trust_legacy_regime),
        "min_fyear": int(min_fyear) if min_fyear is not None else None,
        "require_financials": bool(args.require_financials),
        "event_features": evt_features,
        "calibrated": do_calibrate,
        "firm_overlap": overlap,
        "data": str(args.data),
        "train_run_id": train_run_id,
        # Fraud-Bundles sind Screening-Experimente (wenige Positive, oft negativer Skill).
        "status": "experimental" if args.label_source == "fraud" else "production",
    }

    # 4a) Financial-Baseline — immer, als Referenzpunkt für den Text-Mehrwert.
    n_pos_train = int((y_train > 0).sum()) if not is_ordinal else 0
    if args.calibrate_method == "auto":
        cal_method = "sigmoid" if n_pos_train < 100 else "isotonic"
    else:
        cal_method = args.calibrate_method
    groups_train = df_train[group_col].to_numpy() if group_col else None
    cal_cv = make_calibration_cv(groups_train, y_train) if do_calibrate else 3
    if do_calibrate:
        logger.info(
            "Kalibrierung: method=%s cv=%s (train positives=%d)",
            cal_method,
            f"GroupSplits({len(cal_cv)})" if isinstance(cal_cv, list) else cal_cv,
            n_pos_train,
        )

    def _eval(y, pred) -> dict[str, float]:
        return evaluate_ordinal(y, pred) if is_ordinal else evaluate_probs(y, pred)

    def _fit_tabular(features: list[str]):
        if is_ordinal:
            pipe = build_regression_pipeline(
                features,
                n_estimators=args.n_estimators,
                random_state=args.seed,
            )
        else:
            pipe = build_pipeline(
                features,
                n_estimators=args.n_estimators,
                calibrate=do_calibrate,
                calibration_method=cal_method,
                cv=cal_cv,
                random_state=args.seed,
            )
        fit_pipeline(pipe, df_train, y_train, groups=groups_train)
        pred = predict_output(pipe, df_test, task=task)
        return pipe, pred

    fin_pipe, p_fin_test = _fit_tabular(numeric_features)
    results["financial_baseline"] = _eval(y_test, p_fin_test)
    # New-firm Subset (streng): nur Test-CIKs ohne Trainingsjahre
    if group_col is not None:
        train_ciks = set(df_train[group_col].dropna().astype(int))
        new_mask = ~df_test[group_col].astype("Int64").isin(train_ciks)
        if new_mask.any() and df_test.loc[new_mask, cols.label_col].nunique() >= 2:
            y_new = df_test.loc[new_mask, cols.label_col].to_numpy()
            if not is_ordinal:
                y_new = y_new.astype(int)
            results["financial_new_firms"] = _eval(y_new, p_fin_test[new_mask.to_numpy()])
            extra = (
                f"MAE={results['financial_new_firms']['mae']:.3f}"
                if is_ordinal
                else f"ROC={results['financial_new_firms']['roc_auc']:.3f}"
            )
            logger.info(
                "New-firm Subset: n=%d %s",
                int(results["financial_new_firms"]["n"]),
                extra,
            )
    fin_bundle = ModelBundle(
        pipeline=fin_pipe,
        feature_cols=numeric_features,
        metadata=runtime_metadata({**base_meta, "component": "financial",
                                   "calibration_method": cal_method if do_calibrate else None,
                                   "calibration_cv": (
                                       f"GroupSplits({len(cal_cv)})"
                                       if isinstance(cal_cv, list)
                                       else cal_cv
                                   ) if do_calibrate else None,
                                   "metrics": results["financial_baseline"]}),
    )
    fin_name = bundle_filename(
        "financial",
        label_source=args.label_source,
        horizon_months=bundle_horizon,
        rating_target=args.rating_target if args.label_source == "rating" else None,
    )
    save_single(fin_bundle, out_dir / fin_name)

    report_focus = ("financial_baseline", p_fin_test)
    train_mode = args.mode
    if is_ordinal and train_mode == "ensemble":
        logger.info("Ordinales Rating: ensemble → combined (kein Logit-Mix).")
        train_mode = "combined"

    if train_mode == "combined":
        # 4b) Option A: Finanz- + LLM-Features im selben Modell.
        combined_features = numeric_features + llm_features
        comb_pipe, p_comb_test = _fit_tabular(combined_features)
        results["combined_model"] = _eval(y_test, p_comb_test)
        comb_name = bundle_filename(
            "combined",
            label_source=args.label_source,
            horizon_months=bundle_horizon,
            rating_target=args.rating_target if args.label_source == "rating" else None,
        )
        save_single(
            ModelBundle(
                pipeline=comb_pipe,
                feature_cols=combined_features,
                metadata=runtime_metadata({**base_meta, "component": "combined",
                                           "llm_features": llm_features,
                                           "calibration_method": cal_method if do_calibrate else None,
                                           "metrics": results["combined_model"]}),
            ),
            out_dir / comb_name,
        )
        report_focus = ("combined_model", p_comb_test)

    elif train_mode == "ensemble":
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
            horizon_months=bundle_horizon,
            rating_target=args.rating_target if args.label_source == "rating" else None,
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
