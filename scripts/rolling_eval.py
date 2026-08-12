#!/usr/bin/env python3
"""Rolling-Origin-Evaluation der Financial-Baseline (Clean-Policy).

Trainiert je Cutoff neu auf ``fyear ≤ cutoff``, testet auf dem folgenden
Fenster. Fokus Financial — Combined/Mock-Text bringt laut Freeze keinen
belastbaren Mehrwert und wäre hier unnötig teuer.

Beispiel
--------
python scripts/rolling_eval.py \\
  --data data/processed/zenodo_labeled.csv.gz \\
  --financials data/raw/financials_panel.csv \\
  --events data/raw/edgar_8k_events.csv \\
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
    load_events,
)
from secpd.data.zenodo import load_dataset, resolve_columns  # noqa: E402
from secpd.evaluation import (  # noqa: E402
    cluster_bootstrap_ci,
    evaluate_probs,
    firm_overlap_stats,
    top_k_capture,
)
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.models.pipeline import (  # noqa: E402
    build_pipeline,
    fit_pipeline,
    make_calibration_cv,
)
from secpd.splitting import rolling_origin_splits  # noqa: E402

logger = logging.getLogger("rolling_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD Rolling-Origin Eval")
    p.add_argument("--data", required=True)
    p.add_argument("--financials", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--out", default="benchmarks/rolling_default_h12")
    p.add_argument("--default-horizon", type=int, default=12)
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
    return p.parse_args()


def _prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str]:
    df = load_dataset(args.data)
    panel = pd.read_csv(args.financials)
    panel.columns = [c.lower() for c in panel.columns]
    df = df.merge(panel, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
    events = load_events(args.events)
    df = attach_default_labels(
        df,
        events,
        horizon_months=args.default_horizon,
        trust_legacy_regime=bool(args.trust_legacy_regime),
    )
    df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(args.min_fyear)].copy()
    if not args.allow_missing_financials:
        df = df.loc[df["total_assets"].notna()].copy()
    cols = resolve_columns(df, label_col="label_default")
    df = df.reset_index(drop=True)
    df, fin = add_financial_features(df)
    df, evt = add_event_features(
        df, events, trust_legacy_regime=bool(args.trust_legacy_regime)
    )
    return df, fin + evt, cols.label_col


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    calibrate = bool(args.calibrate) and not bool(args.no_calibrate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, features, label_col = _prepare(args)
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
    pooled_y: list[np.ndarray] = []
    pooled_p: list[np.ndarray] = []
    pooled_g: list[np.ndarray] = []

    for tr, te, meta in folds:
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_tr = df_tr[label_col].astype(int).to_numpy()
        y_te = df_te[label_col].astype(int).to_numpy()
        groups_tr = df_tr["cik"].to_numpy()
        overlap = firm_overlap_stats(df_tr["cik"], df_te["cik"])
        cal_method = "sigmoid" if int(y_tr.sum()) < 100 else "isotonic"
        cv = make_calibration_cv(groups_tr, y_tr) if calibrate else 3
        pipe = build_pipeline(
            features,
            n_estimators=args.n_estimators,
            calibrate=calibrate,
            calibration_method=cal_method,
            cv=cv,
            random_state=args.seed,
        )
        fit_pipeline(pipe, df_tr, y_tr, groups=groups_tr if calibrate else None)
        p = pipe.predict_proba(df_te)[:, 1]
        metrics = evaluate_probs(y_te, p)
        metrics.update(top_k_capture(y_te, p, fractions=(0.1, 0.3)))
        row = {
            **meta,
            **{f"m_{k}": v for k, v in metrics.items()},
            "overlap_rate": overlap["overlap_rate"],
            "n_new_test_firms": overlap["n_new_test_firms"],
        }
        # New-firm subset
        train_ciks = set(df_tr["cik"].dropna().astype(int))
        new_mask = ~df_te["cik"].astype("Int64").isin(train_ciks)
        if int(new_mask.sum()) > 0 and df_te.loc[new_mask, label_col].nunique() >= 2:
            nm = evaluate_probs(
                df_te.loc[new_mask, label_col].astype(int).to_numpy(),
                p[new_mask.to_numpy()],
            )
            row["new_firm_n"] = nm["n"]
            row["new_firm_positives"] = nm["positives"]
            row["new_firm_roc_auc"] = nm["roc_auc"]
        else:
            row["new_firm_n"] = float(new_mask.sum())
            row["new_firm_positives"] = float(
                df_te.loc[new_mask, label_col].astype(int).sum()
            ) if new_mask.any() else 0.0
            row["new_firm_roc_auc"] = float("nan")
        fold_rows.append(row)
        pooled_y.append(y_te)
        pooled_p.append(p)
        pooled_g.append(df_te["cik"].to_numpy())
        logger.info(
            "cutoff=%d test=%s n=%d pos=%d ROC=%.3f Skill=%+.3f overlap=%.0f%%",
            meta["cutoff"],
            meta["test_years"],
            meta["n_test"],
            meta["positives_test"],
            metrics["roc_auc"],
            metrics["brier_skill"],
            100 * overlap["overlap_rate"],
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_dir / "folds.csv", index=False)

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_p)
    g_all = np.concatenate(pooled_g)
    pooled = evaluate_probs(y_all, p_all)
    pooled.update(top_k_capture(y_all, p_all, fractions=(0.1, 0.3)))
    boot = cluster_bootstrap_ci(y_all, p_all, g_all, n_boot=args.n_boot, seed=args.seed)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_folds": len(folds),
        "policy": {
            "min_fyear": args.min_fyear,
            "trust_legacy_regime": bool(args.trust_legacy_regime),
            "require_financials": not bool(args.allow_missing_financials),
            "test_window_years": args.test_window_years,
            "step_years": args.step_years,
            "calibrate": calibrate,
        },
        "mean_fold_roc_auc": float(folds_df["m_roc_auc"].mean()),
        "mean_fold_pr_auc": float(folds_df["m_pr_auc"].mean()),
        "mean_fold_brier_skill": float(folds_df["m_brier_skill"].mean()),
        "mean_overlap_rate": float(folds_df["overlap_rate"].mean()),
        "pooled": pooled,
        "bootstrap": boot,
        "folds": fold_rows,
    }
    (out_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# SEC-PD Rolling-Origin (Financial)",
        "",
        f"- folds: {len(folds)} · pooled n={int(pooled['n'])} · "
        f"positives={int(pooled['positives'])}",
        f"- mean fold ROC={summary['mean_fold_roc_auc']:.3f} · "
        f"PR={summary['mean_fold_pr_auc']:.3f} · "
        f"Skill={summary['mean_fold_brier_skill']:+.3f}",
        f"- mean firm-overlap: {summary['mean_overlap_rate']:.1%}",
        f"- pooled ROC={pooled['roc_auc']:.3f} "
        f"[{boot['roc_auc']['ci_low']:.3f}, {boot['roc_auc']['ci_high']:.3f}]",
        f"- pooled PR={pooled['pr_auc']:.3f} · Skill={pooled['brier_skill']:+.3f} · "
        f"Top10% capture={pooled['top_10pct_capture']:.1%}",
        "",
        "| cutoff | test years | n | pos | ROC | PR | Skill | overlap | new-firm ROC |",
        "|--------|------------|---|-----|-----|----|-------|---------|--------------|",
    ]
    for r in fold_rows:
        yrs = ",".join(str(x) for x in r["test_years"])
        nf = r["new_firm_roc_auc"]
        nf_s = f"{nf:.3f}" if nf == nf else "n/a"
        lines.append(
            f"| {r['cutoff']} | {yrs} | {r['n_test']} | {r['positives_test']} | "
            f"{r['m_roc_auc']:.3f} | {r['m_pr_auc']:.3f} | {r['m_brier_skill']:+.3f} | "
            f"{r['overlap_rate']:.0%} | {nf_s} |"
        )
    lines += [
        "",
        "Financial-only Rolling-Eval. Combined absichtlich ausgelassen "
        "(Freeze: kein signifikanter Mehrwert).",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    logger.info("Rolling-Eval geschrieben: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
