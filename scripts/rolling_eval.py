#!/usr/bin/env python3
"""Rolling-Origin-Evaluation: Financial vs Combined (Clean-Policy).

Trainiert je Cutoff neu auf ``fyear ≤ cutoff``, testet auf dem folgenden
Fenster. Combined nutzt den LLM-Cache (keine API bei ``--llm-cache-only``).

Beispiel
--------
python scripts/rolling_eval.py \\
  --data data/processed/zenodo_labeled.csv.gz \\
  --financials data/raw/financials_panel.csv \\
  --events data/raw/edgar_8k_events.csv \\
  --llm openai --llm-cache-only \\
  --out benchmarks/rolling_default_h12
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from secpd.data.events import (  # noqa: E402
    MIN_FYEAR_WITH_FINANCIALS,
    add_event_features,
    attach_default_labels,
    feature_exclusions_for_labels,
    load_events,
    parse_label_concepts,
)
from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.evaluation import (  # noqa: E402
    cluster_bootstrap_ci,
    evaluate_probs,
    firm_overlap_stats,
    top_k_capture,
)
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import attach_text_features  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.pipeline import (  # noqa: E402
    build_pipeline,
    fit_pipeline,
    make_calibration_cv,
)
from secpd.splitting import rolling_origin_splits, smart_split  # noqa: E402

logger = logging.getLogger("rolling_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD Rolling-Origin Eval")
    p.add_argument("--data", required=True)
    p.add_argument("--financials", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--out", default="benchmarks/rolling_default_h12")
    p.add_argument("--default-horizon", type=int, default=12)
    p.add_argument("--label-concepts", default="bankruptcy")
    p.add_argument("--min-fyear", type=int, default=MIN_FYEAR_WITH_FINANCIALS)
    p.add_argument("--allow-missing-financials", action="store_true")
    p.add_argument("--trust-legacy-regime", action="store_true")
    p.add_argument("--test-window-years", type=int, default=2)
    p.add_argument("--step-years", type=int, default=2)
    p.add_argument("--min-train-years", type=int, default=4)
    p.add_argument("--min-train-positives", type=int, default=5)
    p.add_argument("--min-test-positives", type=int, default=1)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--calibrate", action="store_true", default=True)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--llm", choices=["mock", "bank", "lmstudio", "openai", "chatgpt"], default="openai")
    p.add_argument("--llm-cache-only", action="store_true", default=True)
    p.add_argument("--no-llm-cache-only", action="store_true")
    p.add_argument("--max-chars", type=int, default=12_000)
    p.add_argument("--skip-combined", action="store_true")
    p.add_argument("--group-split", action="store_true", help="Zusätzlich Group-Split (keine Firm-Wiederholung)")
    return p.parse_args()


def _prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str], str]:
    df = load_dataset(args.data)
    panel = pd.read_csv(args.financials)
    panel.columns = [c.lower() for c in panel.columns]
    df = df.merge(panel, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
    events = load_events(args.events)
    concepts = parse_label_concepts(args.label_concepts)
    exclude = feature_exclusions_for_labels(concepts)
    df = attach_default_labels(
        df,
        events,
        horizon_months=args.default_horizon,
        trust_legacy_regime=bool(args.trust_legacy_regime),
        label_concepts=concepts,
    )
    df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(args.min_fyear)].copy()
    if not args.allow_missing_financials:
        df = df.loc[df["total_assets"].notna()].copy()
    cols = resolve_columns(df, label_col="label_default")
    df = df.reset_index(drop=True)
    df, fin = add_financial_features(df)
    df, evt = add_event_features(
        df,
        events,
        trust_legacy_regime=bool(args.trust_legacy_regime),
        exclude_concepts=exclude,
    )
    numeric = fin + evt
    text_cols: list[str] = []
    if not args.skip_combined:
        cache_only = bool(args.llm_cache_only) and not bool(args.no_llm_cache_only)
        client = get_llm_client(args.llm, cached=True, cache_only=cache_only)
        logger.info("Text-Features via %s", client.name)
        df, text_cols = attach_text_features(
            df,
            client=client,
            text_col=cols.text_col,
            id_col=cols.id_col,
            max_chars=args.max_chars,
            progress_every=250,
        )
    return df, numeric, text_cols, cols.label_col


def _fit_score(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    y_tr: np.ndarray,
    features: list[str],
    *,
    calibrate: bool,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    cal_method = "sigmoid" if int(y_tr.sum()) < 100 else "isotonic"
    cv = make_calibration_cv(df_tr["cik"].to_numpy(), y_tr) if calibrate else 3
    pipe = build_pipeline(
        features,
        n_estimators=n_estimators,
        calibrate=calibrate,
        calibration_method=cal_method,
        cv=cv,
        random_state=seed,
    )
    fit_pipeline(pipe, df_tr, y_tr)
    return pipe.predict_proba(df_te)[:, 1]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = evaluate_probs(y, p)
    m.update(top_k_capture(y, p, fractions=(0.1, 0.3)))
    return m


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    calibrate = bool(args.calibrate) and not bool(args.no_calibrate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, numeric, text_cols, label_col = _prepare(args)
    combined_features = numeric + text_cols
    folds = rolling_origin_splits(
        df,
        label_col=label_col,
        year_col="fyear",
        test_window_years=args.test_window_years,
        step_years=args.step_years,
        min_train_years=args.min_train_years,
        min_train_positives=args.min_train_positives,
        min_test_positives=args.min_test_positives,
    )
    if not folds:
        logger.error("Keine brauchbaren Rolling-Folds.")
        return 2

    fold_rows: list[dict] = []
    pooled: dict[str, list[np.ndarray]] = {
        "y": [], "g": [], "p_fin": [], "p_comb": [],
    }

    for tr, te, meta in folds:
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_tr = df_tr[label_col].astype(int).to_numpy()
        y_te = df_te[label_col].astype(int).to_numpy()
        overlap = firm_overlap_stats(df_tr["cik"], df_te["cik"])
        p_fin = _fit_score(
            df_tr, df_te, y_tr, numeric,
            calibrate=calibrate, n_estimators=args.n_estimators, seed=args.seed,
        )
        m_fin = _metrics(y_te, p_fin)
        row: dict = {
            **meta,
            **{f"fin_{k}": v for k, v in m_fin.items()},
            "overlap_rate": overlap["overlap_rate"],
            "n_new_test_firms": overlap["n_new_test_firms"],
        }
        pooled["y"].append(y_te)
        pooled["g"].append(df_te["cik"].to_numpy())
        pooled["p_fin"].append(p_fin)
        if text_cols:
            p_comb = _fit_score(
                df_tr, df_te, y_tr, combined_features,
                calibrate=calibrate, n_estimators=args.n_estimators, seed=args.seed,
            )
            m_comb = _metrics(y_te, p_comb)
            row.update({f"comb_{k}": v for k, v in m_comb.items()})
            pooled["p_comb"].append(p_comb)
            logger.info(
                "cutoff=%d n=%d pos=%d ROC fin=%.3f comb=%.3f PR fin=%.3f comb=%.3f",
                meta["cutoff"], meta["n_test"], meta["positives_test"],
                m_fin["roc_auc"], m_comb["roc_auc"], m_fin["pr_auc"], m_comb["pr_auc"],
            )
        else:
            logger.info(
                "cutoff=%d n=%d pos=%d ROC fin=%.3f",
                meta["cutoff"], meta["n_test"], meta["positives_test"], m_fin["roc_auc"],
            )
        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_dir / "folds.csv", index=False)

    y_all = np.concatenate(pooled["y"])
    p_fin_all = np.concatenate(pooled["p_fin"])
    g_all = np.concatenate(pooled["g"])
    pooled_fin = _metrics(y_all, p_fin_all)
    boot_fin = cluster_bootstrap_ci(y_all, p_fin_all, g_all, n_boot=args.n_boot, seed=args.seed)

    summary: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_folds": len(folds),
        "text_features": text_cols,
        "policy": {
            "min_fyear": args.min_fyear,
            "trust_legacy_regime": bool(args.trust_legacy_regime),
            "require_financials": not bool(args.allow_missing_financials),
            "test_window_years": args.test_window_years,
            "step_years": args.step_years,
            "calibrate": calibrate,
            "llm": args.llm,
            "default_horizon_months": args.default_horizon,
            "label_concepts": list(parse_label_concepts(args.label_concepts)),
        },
        "mean_fold_roc_fin": float(folds_df["fin_roc_auc"].mean()),
        "mean_fold_pr_fin": float(folds_df["fin_pr_auc"].mean()),
        "pooled_financial": pooled_fin,
        "bootstrap_financial": boot_fin,
        "folds": fold_rows,
    }
    if text_cols and pooled["p_comb"]:
        p_comb_all = np.concatenate(pooled["p_comb"])
        pooled_comb = _metrics(y_all, p_comb_all)
        boot_comb = cluster_bootstrap_ci(
            y_all, p_comb_all, g_all, n_boot=args.n_boot, seed=args.seed
        )
        summary["mean_fold_roc_comb"] = float(folds_df["comb_roc_auc"].mean())
        summary["mean_fold_pr_comb"] = float(folds_df["comb_pr_auc"].mean())
        summary["pooled_combined"] = pooled_comb
        summary["bootstrap_combined"] = boot_comb

    group_eval = None
    if args.group_split:
        tr, te, strat = smart_split(
            df, label_col=label_col, group_col="cik", year_col="fyear",
            test_size=0.2, strategy="group", random_state=args.seed,
        )
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_tr = df_tr[label_col].astype(int).to_numpy()
        y_te = df_te[label_col].astype(int).to_numpy()
        p_fin = _fit_score(
            df_tr, df_te, y_tr, numeric,
            calibrate=calibrate, n_estimators=args.n_estimators, seed=args.seed,
        )
        group_eval = {"split": strat, "financial": _metrics(y_te, p_fin)}
        if text_cols:
            p_comb = _fit_score(
                df_tr, df_te, y_tr, combined_features,
                calibrate=calibrate, n_estimators=args.n_estimators, seed=args.seed,
            )
            group_eval["combined"] = _metrics(y_te, p_comb)
        summary["group_split"] = group_eval

    (out_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    has_comb = bool(text_cols)
    lines = [
        "# SEC-PD Rolling-Origin (Financial vs Combined)",
        "",
        f"- folds: {len(folds)} · pooled n={int(pooled_fin['n'])} · "
        f"positives={int(pooled_fin['positives'])}",
        f"- text features: {', '.join(text_cols) if text_cols else '(none)'}",
        f"- mean fold ROC financial={summary['mean_fold_roc_fin']:.3f}"
        + (f" · combined={summary['mean_fold_roc_comb']:.3f}" if has_comb else ""),
        f"- mean fold PR  financial={summary['mean_fold_pr_fin']:.3f}"
        + (f" · combined={summary['mean_fold_pr_comb']:.3f}" if has_comb else ""),
        f"- pooled ROC financial={pooled_fin['roc_auc']:.3f} "
        f"[{boot_fin['roc_auc']['ci_low']:.3f}, {boot_fin['roc_auc']['ci_high']:.3f}]",
    ]
    if has_comb:
        pc = summary["pooled_combined"]
        bc = summary["bootstrap_combined"]
        lines.append(
            f"- pooled ROC combined={pc['roc_auc']:.3f} "
            f"[{bc['roc_auc']['ci_low']:.3f}, {bc['roc_auc']['ci_high']:.3f}]"
        )
        lines.append(
            f"- pooled PR financial={pooled_fin['pr_auc']:.3f} · "
            f"combined={pc['pr_auc']:.3f} · "
            f"Top10% fin={pooled_fin['top_10pct_capture']:.1%} "
            f"comb={pc['top_10pct_capture']:.1%}"
        )
    lines += [
        "",
        "| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |",
        "|--------|---|-----|---------|----------|--------|---------|------------|-------------|",
    ]
    for r in fold_rows:
        rc = r.get("comb_roc_auc", float("nan"))
        pc_ = r.get("comb_pr_auc", float("nan"))
        t10f = r.get("fin_top_10pct_capture", float("nan"))
        t10c = r.get("comb_top_10pct_capture", float("nan"))
        lines.append(
            f"| {r['cutoff']} | {r['n_test']} | {r['positives_test']} | "
            f"{r['fin_roc_auc']:.3f} | {rc:.3f} | {r['fin_pr_auc']:.3f} | "
            f"{pc_:.3f} | {t10f:.2f} | {t10c:.2f} |"
        )
    if group_eval:
        gf, gc = group_eval["financial"], group_eval.get("combined")
        lines += [
            "",
            "## Group-Split (keine Firm-Wiederholung)",
            "",
            f"- n={int(gf['n'])} pos={int(gf['positives'])}",
            f"- Financial ROC={gf['roc_auc']:.3f} PR={gf['pr_auc']:.3f} "
            f"Top10%={gf['top_10pct_capture']:.2f}",
        ]
        if gc:
            lines.append(
                f"- Combined  ROC={gc['roc_auc']:.3f} PR={gc['pr_auc']:.3f} "
                f"Top10%={gc['top_10pct_capture']:.2f}"
            )
    lines.append("")
    report = "\n".join(lines)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    logger.info("Rolling-Eval geschrieben: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
