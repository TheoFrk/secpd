#!/usr/bin/env python3
"""Friert ein Clean-Sample-Benchmark ein und vergleicht Null / Financial / Combined.

Rekonstruiert dieselbe Daten-Policy wie das Default-Training
(Legacy-Regime aus, min-fyear, optional require-financials), schreibt
Test-IDs + Vorhersagen + CIK-Bootstrap-Report.

Beispiel
--------
python scripts/freeze_benchmark.py \\
  --data data/processed/zenodo_full.csv.gz \\
  --financials data/raw/financials_panel_full.csv \\
  --events data/raw/edgar_8k_events_full.csv \\
  --financial-model models/financial_default_h12.joblib \\
  --combined-model models/combined_default_h12.joblib \\
  --out benchmarks/default_h12_full
"""
from __future__ import annotations

import argparse
import hashlib
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
    cluster_bootstrap_delta,
    decile_table,
    evaluate_probs,
    firm_overlap_stats,
    reliability_table,
    top_k_capture,
)
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import attach_text_features  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.persistence import load_any  # noqa: E402
from secpd.splitting import smart_split  # noqa: E402

logger = logging.getLogger("freeze_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC-PD Frozen Benchmark")
    p.add_argument("--data", required=True)
    p.add_argument("--financials", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--financial-model", required=True)
    p.add_argument("--combined-model", required=True)
    p.add_argument("--out", default="benchmarks/default_h12_full")
    p.add_argument("--default-horizon", type=int, default=12)
    p.add_argument("--min-fyear", type=int, default=MIN_FYEAR_WITH_FINANCIALS)
    p.add_argument("--require-financials", action="store_true", default=True)
    p.add_argument("--allow-missing-financials", action="store_true",
                   help="require-financials abschalten")
    p.add_argument("--trust-legacy-regime", action="store_true")
    p.add_argument("--split", default="temporal", choices=["auto", "temporal", "group", "random"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1_000)
    p.add_argument("--llm", choices=["mock", "bank", "lmstudio", "openai", "chatgpt"], default="openai")
    p.add_argument("--max-chars", type=int, default=12_000)
    return p.parse_args()


def _sha1_ids(ids: pd.Series) -> str:
    blob = "\n".join(sorted(ids.astype(str).tolist())).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _score_bundle(df: pd.DataFrame, model_path: Path, *, llm: str, max_chars: int) -> np.ndarray:
    payload = load_any(model_path)
    work = df.copy()
    feature_cols = list(payload["feature_cols"])
    needs_text = any(str(c).startswith(("llm_", "txt_")) for c in feature_cols)
    if needs_text:
        cols = resolve_columns(work)
        if cols.text_col is None:
            raise RuntimeError(f"{model_path.name}: Textspalte fehlt")
        client = get_llm_client(llm, cache_only=True)
        work, _ = attach_text_features(
            work,
            client=client,
            text_col=cols.text_col,
            id_col=cols.id_col,
            max_chars=max_chars,
            progress_every=10_000,
        )
    for c in feature_cols:
        if c not in work.columns:
            work[c] = float("nan")
    return payload["pipeline"].predict_proba(work)[:, 1]


def _enrich(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = evaluate_probs(y, p)
    m.update(top_k_capture(y, p, fractions=(0.1, 0.3)))
    return m


def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.1%}"


def _fmt_x(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}×"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    require_fin = bool(args.require_financials) and not bool(args.allow_missing_financials)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Daten wie Training (Clean-Policy) ---
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
    before = len(df)
    df = df.loc[pd.to_numeric(df["fyear"], errors="coerce") >= int(args.min_fyear)].copy()
    logger.info("min-fyear=%d: %d → %d", args.min_fyear, before, len(df))
    if require_fin:
        before = len(df)
        df = df.loc[df["total_assets"].notna()].copy()
        logger.info("require-financials: %d → %d", before, len(df))
    if df.empty:
        logger.error("Datensatz nach Filtern leer.")
        return 2

    cols = resolve_columns(df, label_col="label_default")
    df = df.reset_index(drop=True)
    df, _ = add_financial_features(df)
    df, _ = add_event_features(
        df, events, trust_legacy_regime=bool(args.trust_legacy_regime)
    )

    train_idx, test_idx, strategy = smart_split(
        df,
        label_col=cols.label_col,
        group_col="cik" if "cik" in df.columns else None,
        year_col=cols.year_col,
        test_size=args.test_size,
        strategy=args.split,
        random_state=args.seed,
    )
    df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_test = df_test[cols.label_col].astype(int).to_numpy()
    groups = df_test["cik"].to_numpy()
    overlap = firm_overlap_stats(df_train["cik"], df_test["cik"])
    logger.info(
        "Split=%s train=%d test=%d pos_test=%d overlap=%.1f%%",
        strategy, len(df_train), len(df_test), int(y_test.sum()),
        100 * overlap["overlap_rate"],
    )

    # --- Scores ---
    base_rate = float(y_test.mean()) if len(y_test) else float("nan")
    p_null = np.full(len(df_test), base_rate)
    logger.info("Score Financial …")
    p_fin = _score_bundle(
        df_test, Path(args.financial_model), llm=args.llm, max_chars=args.max_chars
    )
    logger.info("Score Combined …")
    p_comb = _score_bundle(
        df_test, Path(args.combined_model), llm=args.llm, max_chars=args.max_chars
    )

    results = {
        "null_base_rate": _enrich(y_test, p_null),
        "financial": _enrich(y_test, p_fin),
        "combined": _enrich(y_test, p_comb),
    }

    logger.info("CIK-Bootstrap (n=%d) …", args.n_boot)
    boot = {
        "financial": cluster_bootstrap_ci(
            y_test, p_fin, groups, n_boot=args.n_boot, seed=args.seed
        ),
        "combined": cluster_bootstrap_ci(
            y_test, p_comb, groups, n_boot=args.n_boot, seed=args.seed
        ),
        "delta_combined_minus_financial_roc": cluster_bootstrap_delta(
            y_test, p_comb, p_fin, groups, n_boot=args.n_boot, seed=args.seed,
            metric="roc_auc",
        ),
        "delta_combined_minus_financial_pr": cluster_bootstrap_delta(
            y_test, p_comb, p_fin, groups, n_boot=args.n_boot, seed=args.seed,
            metric="pr_auc",
        ),
    }

    # --- Artefakte ---
    ids = df_test[cols.id_col].astype(str)
    freeze_hash = _sha1_ids(ids)
    test_ids = df_test[[cols.id_col, "cik", "fyear", cols.label_col]].copy()
    test_ids.to_csv(out_dir / "test_ids.csv", index=False)

    preds = test_ids.copy()
    preds["p_null"] = p_null
    preds["p_financial"] = p_fin
    preds["p_combined"] = p_comb
    preds.to_csv(out_dir / "predictions.csv", index=False)

    decile_table(y_test, p_comb).to_csv(out_dir / "deciles_combined.csv", index=False)
    reliability_table(y_test, p_comb).to_csv(out_dir / "reliability_combined.csv", index=False)

    fin_meta = (load_any(args.financial_model).get("metadata") or {})
    comb_meta = (load_any(args.combined_model).get("metadata") or {})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freeze_hash": freeze_hash,
        "policy": {
            "label_source": "default",
            "default_horizon_months": args.default_horizon,
            "trust_legacy_regime": bool(args.trust_legacy_regime),
            "min_fyear": int(args.min_fyear),
            "require_financials": require_fin,
            "split": strategy,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "data": str(args.data),
        "financials": str(args.financials),
        "events": str(args.events),
        "models": {
            "financial": str(args.financial_model),
            "combined": str(args.combined_model),
            "financial_train_run_id": fin_meta.get("train_run_id"),
            "combined_train_run_id": comb_meta.get("train_run_id"),
        },
        "n_train": len(df_train),
        "n_test": len(df_test),
        "positives_test": int(y_test.sum()),
        "firm_overlap": overlap,
        "metrics": results,
        "bootstrap": boot,
    }
    (out_dir / "report.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Kurzbericht
    lines = [
        "# SEC-PD Frozen Benchmark",
        "",
        f"- freeze_hash: `{freeze_hash}`",
        f"- split: {strategy} · test n={len(df_test)} · positives={int(y_test.sum())} · "
        f"base={base_rate:.2%}",
        f"- firm overlap: {overlap['overlap_rate']:.1%} "
        f"({int(overlap['n_overlap_firms'])}/{int(overlap['n_test_firms'])})",
        f"- policy: legacy={args.trust_legacy_regime} · min_fyear={args.min_fyear} · "
        f"require_financials={require_fin}",
        "",
        "| Modell | ROC-AUC | PR-AUC | Brier | Skill | Top10% Capture | Top10% Lift |",
        "|--------|---------|--------|-------|-------|----------------|-------------|",
    ]
    for name, key in (
        ("Null (Test-Basisrate)", "null_base_rate"),
        ("Financial", "financial"),
        ("Combined", "combined"),
    ):
        m = results[key]
        lines.append(
            f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.4f} | "
            f"{m['brier_skill']:+.3f} | {_fmt_pct(m['top_10pct_capture'])} | "
            f"{_fmt_x(m['top_10pct_lift'])} |"
        )
    d_roc = boot["delta_combined_minus_financial_roc"]
    d_pr = boot["delta_combined_minus_financial_pr"]
    lines += [
        "",
        "## CIK-Bootstrap 95%-CI",
        "",
        f"- Financial ROC: {boot['financial']['roc_auc']['ci_low']:.3f}–"
        f"{boot['financial']['roc_auc']['ci_high']:.3f}",
        f"- Combined ROC: {boot['combined']['roc_auc']['ci_low']:.3f}–"
        f"{boot['combined']['roc_auc']['ci_high']:.3f}",
        f"- Δ Combined−Financial ROC: {d_roc['point']:+.3f} "
        f"[{d_roc['ci_low']:+.3f}, {d_roc['ci_high']:+.3f}]",
        f"- Δ Combined−Financial PR: {d_pr['point']:+.3f} "
        f"[{d_pr['ci_low']:+.3f}, {d_pr['ci_high']:+.3f}]",
        "",
        "**Hinweis:** Bei wenigen Positives sind AUC-CIs breit — Top-k und Skill mitlesen.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    logger.info("Benchmark geschrieben: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
