#!/usr/bin/env python3
"""Vergleicht Label-Politiken über Rolling-PR / Top-10%-Capture / Group-Split.

Leitmetrik laut Modellplan: nicht der Temporal-Split mit 6 Events, sondern
Rolling-PR und Top-10%-Capture; Group-Split als Firm-Holdout.

Keine LLM-API: ``--llm-cache-only``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

POLICIES = [
    {"horizon": 12, "concepts": "bankruptcy"},
    {"horizon": 24, "concepts": "bankruptcy"},
    {"horizon": 36, "concepts": "bankruptcy"},
    {"horizon": 12, "concepts": "bankruptcy,delisting"},
    {"horizon": 24, "concepts": "bankruptcy,delisting"},
]


def _slug(p: dict) -> str:
    c = p["concepts"].replace(",", "-")
    return f"h{p['horizon']}_{c}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Label-Policy-Vergleich (Rolling-PR)")
    ap.add_argument("--data", default=str(ROOT / "data/processed/zenodo_full.csv.gz"))
    ap.add_argument("--financials", default=str(ROOT / "data/raw/financials_panel_full.csv"))
    ap.add_argument("--events", default=str(ROOT / "data/raw/edgar_8k_events_full.csv"))
    ap.add_argument("--tag", default="full", help="Ordnerpräfix: rolling_<tag>_<slug>")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for pol in POLICIES:
        slug = _slug(pol)
        out = ROOT / "benchmarks" / f"rolling_{args.tag}_{slug}"
        cmd = [
            PY, str(ROOT / "scripts" / "rolling_eval.py"),
            "--data", args.data,
            "--financials", args.financials,
            "--events", args.events,
            "--default-horizon", str(pol["horizon"]),
            "--label-concepts", pol["concepts"],
            "--llm", "openai",
            "--llm-cache-only",
            "--group-split",
            "--n-boot", str(args.n_boot),
            "--out", str(out),
        ]
        print("==>", " ".join(cmd[-12:]), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAILED {slug} rc={rc}", flush=True)
            return rc
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        pc = report.get("pooled_combined") or {}
        gf = (report.get("group_split") or {}).get("financial") or {}
        gc = (report.get("group_split") or {}).get("combined") or {}
        rows.append({
            "slug": slug,
            "horizon": pol["horizon"],
            "concepts": pol["concepts"],
            "mean_pr_comb": report.get("mean_fold_pr_comb"),
            "mean_roc_comb": report.get("mean_fold_roc_comb"),
            "pooled_pr_comb": pc.get("pr_auc"),
            "pooled_top10_comb": pc.get("top_10pct_capture"),
            "pooled_pos": pc.get("positives"),
            "group_pr_comb": gc.get("pr_auc"),
            "group_roc_comb": gc.get("roc_auc"),
            "group_n_pos": gc.get("positives"),
            "mean_pr_fin": report.get("mean_fold_pr_fin"),
        })

    def key(r: dict) -> tuple:
        pr = r["mean_pr_comb"] if r["mean_pr_comb"] is not None else -1
        top = r["pooled_top10_comb"] if r["pooled_top10_comb"] is not None else -1
        return (pr, top)

    ranked = sorted(rows, key=key, reverse=True)
    winner = ranked[0]
    lines = [
        "# Label-Policy-Vergleich (Leitmetrik: Rolling-PR Combined)",
        "",
        f"Universum: `{Path(args.data).name}` · tag=`{args.tag}`",
        "",
        f"**Gewinner:** `{winner['slug']}` "
        f"(mean fold PR={winner['mean_pr_comb']:.3f}, "
        f"Top10%={winner['pooled_top10_comb']:.1%})",
        "",
        "| policy | mean PR comb | mean ROC comb | pooled Top10% | pooled pos | group PR comb | mean PR fin |",
        "|--------|--------------|---------------|---------------|------------|---------------|-------------|",
    ]
    for r in ranked:
        def fmt(x, pct=False):
            if x is None:
                return "n/a"
            return f"{100*x:.1f}%" if pct else f"{x:.3f}"
        lines.append(
            f"| {r['slug']} | {fmt(r['mean_pr_comb'])} | {fmt(r['mean_roc_comb'])} | "
            f"{fmt(r['pooled_top10_comb'], pct=True)} | {r['pooled_pos']} | "
            f"{fmt(r['group_pr_comb'])} | {fmt(r['mean_pr_fin'])} |"
        )
    lines.append("")
    out_md = ROOT / "benchmarks" / f"policy_compare_{args.tag}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "benchmarks" / f"policy_compare_{args.tag}.json").write_text(
        json.dumps({"winner": winner, "tag": args.tag, "rows": ranked}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\nWrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
