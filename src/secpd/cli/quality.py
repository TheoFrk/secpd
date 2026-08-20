"""Modellgüte-Anzeige: Bundle-Metriken plus Frozen/Rolling-Reports."""
from __future__ import annotations

from pathlib import Path

from secpd.cli.catalog import is_experimental, list_model_catalog, warn_model_coherence
from secpd.cli.paths import FREEZE_REPORT, ROLLING_REPORT, ROOT
from secpd.cli.ui import C, banner, clear, fmt_pct, hr, pause


def _fmt_metric(val: object, spec: str = ".3f") -> str:
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if x != x:
        return "n/a"
    return format(x, spec)


def _print_overlap_and_unseen(md: dict, *, ordinal: bool) -> None:
    ov = md.get("firm_overlap") or {}
    if ov.get("n_test_firms"):
        rate = ov.get("overlap_rate", float("nan"))
        rate_s = f"{100 * float(rate):5.1f} %" if rate == rate else "n/a"
        print(
            f"    Overlap   {rate_s} Testfirmen auch im Training "
            f"(neu={int(ov.get('n_new_test_firms') or 0)})"
        )
    unseen = md.get("metrics_unseen_cik") or {}
    if not unseen.get("n"):
        return
    n_u = int(unseen["n"])
    firms = unseen.get("n_test_firms")
    firms_s = f", Firmen={int(firms)}" if firms is not None and firms == firms else ""
    if ordinal or "mae" in unseen:
        print(
            f"    Unseen-CIK MAE={_fmt_metric(unseen.get('mae'))}  "
            f"Spearman={_fmt_metric(unseen.get('spearman'))}  "
            f"±1={_fmt_metric(100 * float(unseen.get('hit_pm1') or 0), '.1f')} %  "
            f"n={n_u}{firms_s}"
        )
    else:
        print(
            f"    Unseen-CIK ROC={_fmt_metric(unseen.get('roc_auc'))}  "
            f"PR={_fmt_metric(unseen.get('pr_auc'))}  "
            f"n={n_u}{firms_s}"
        )
    print(
        f"    {C.DIM}Group-Holdout: ganze CIKs außerhalb des Trainingsuniversums.{C.RESET}"
    )


def _experimental_note(row: dict) -> str:
    """Kurzer Warnhinweis aus Bundle-Metriken (Fraud: wenige Positive, Skill)."""
    md = row.get("metadata") or {}
    metrics = md.get("metrics") or {}
    pos = metrics.get("positives")
    skill = metrics.get("brier_skill")
    bits: list[str] = []
    if pos is not None:
        try:
            bits.append(f"{int(pos)} Positive")
        except (TypeError, ValueError):
            pass
    if skill is None:
        brier = metrics.get("brier")
        base = metrics.get("base_rate")
        if brier is not None and base is not None and 0 < float(base) < 1:
            skill = 1.0 - float(brier) / (float(base) * (1.0 - float(base)))
    if skill is not None and skill == skill:
        bits.append(f"Skill {float(skill):+.3f}")
    extra = ", ".join(bits) if bits else "zu wenig Evidenz"
    return f"nicht entscheidungsfähig ({extra})"


def show_model_quality() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Modellgüte (Bundles unter models/){C.RESET}")
    hr()
    catalog = list_model_catalog()
    if not catalog:
        print(f"\n  {C.YELLOW}Keine Modelle gefunden.{C.RESET}")
        pause()
        return
    warn_model_coherence(catalog)
    print()

    for row in catalog:
        path = Path(row["path"])
        md = row.get("metadata") or {}
        metrics = md.get("metrics") or {}
        title = f"{row.get('component') or '?'} / {row.get('label_source') or '?'}"
        if row.get("horizon_months"):
            title += f" h{row['horizon_months']}"
        if row.get("rating_target"):
            title += f" {row['rating_target']}"
        if row.get("legacy"):
            title += "  [LEGACY]"
        if is_experimental(row):
            title += "  [experimentell]"
        print(f"\n  {C.BOLD}{C.BLUE}{title}{C.RESET}  {C.DIM}{path.name}{C.RESET}")
        print(f"    trained   {row.get('trained_at') or md.get('created_at_utc') or '—'}")
        print(f"    data      {row.get('data') or '—'}")
        cal = md.get("calibration_method")
        cal_s = f"/{cal}" if cal else ""
        task_s = f"  ·  task={md.get('task')}" if md.get("task") else ""
        print(
            f"    split     {md.get('split_strategy', '?')}  ·  "
            f"kalibriert={md.get('calibrated', '?')}{cal_s}  ·  "
            f"mode={md.get('mode', '?')}{task_s}"
        )
        if is_experimental(row):
            print(f"    {C.YELLOW}Warnung   {_experimental_note(row)}{C.RESET}")
        if not metrics:
            print(f"  {C.DIM}Keine Metriken im Bundle.{C.RESET}")
            continue

        if "mae" in metrics:
            mae = metrics.get("mae", float("nan"))
            sp = metrics.get("spearman", float("nan"))
            hit = metrics.get("hit_pm1", float("nan"))
            n = int(metrics.get("n", 0))
            print(f"    MAE       {mae:6.3f}  {C.DIM}Notches (niedriger = besser){C.RESET}")
            print(f"    Spearman  {sp:6.3f}  {C.DIM}Rangkorrelation{C.RESET}")
            print(f"    ±1-Notch  {100 * hit:5.1f} %  {C.DIM}Treffer auf eine Note genau{C.RESET}")
            print(f"    Testset   n={n}")
            _print_overlap_and_unseen(md, ordinal=True)
            continue

        roc = metrics.get("roc_auc", float("nan"))
        pr = metrics.get("pr_auc", float("nan"))
        brier = metrics.get("brier", float("nan"))
        base = metrics.get("base_rate", float("nan"))
        n = int(metrics.get("n", 0))
        pos = int(metrics.get("positives", 0))

        def bar(val: float, lo: float = 0.5, hi: float = 0.8) -> str:
            if val != val:  # NaN
                return "—"
            t = (val - lo) / (hi - lo)
            t = min(1.0, max(0.0, t))
            nfill = int(round(t * 16))
            return "█" * nfill + "░" * (16 - nfill)

        print(f"    ROC-AUC   {roc:6.3f}  {C.CYAN}{bar(roc)}{C.RESET}  {C.DIM}Ranking-Güte{C.RESET}")
        print(f"    PR-AUC    {pr:6.3f}  {C.CYAN}{bar(pr, 0.05, 0.25)}{C.RESET}  {C.DIM}bei seltenen Events{C.RESET}")
        print(f"    Brier     {brier:6.3f}  {C.DIM}Kalibrierung (niedriger = besser){C.RESET}")
        skill = metrics.get("brier_skill")
        if skill is None and base == base and brier == brier and 0 < base < 1:
            skill = 1.0 - float(brier) / (float(base) * (1.0 - float(base)))
        if skill is not None and skill == skill:
            print(
                f"    Skill     {skill:+6.3f}  {C.DIM}"
                f"vs. konstanter Basisrate (>0 = besser){C.RESET}"
            )
        print(f"    Testset   n={n}, Positive={pos}, Basisrate={fmt_pct(base)}")
        _print_overlap_and_unseen(md, ordinal=False)
        if md.get("min_fyear") is not None:
            print(
                f"    policy    min_fyear={md.get('min_fyear')}  ·  "
                f"legacy={md.get('trust_legacy_regime', '?')}  ·  "
                f"require_fin={md.get('require_financials', '?')}"
            )

    freeze = FREEZE_REPORT
    if freeze.exists():
        print()
        print(f"  {C.BOLD}Temporaler Holdout (Freeze){C.RESET}  {C.DIM}{freeze.relative_to(ROOT)}{C.RESET}")
        print(
            f"  {C.YELLOW}Optimistisch: Firm-Overlap hoch, Krisenjahre (z. B. 2019) "
            f"liegen oft im Trainingsset — RF kann Finanz-Kombinationen memorieren.{C.RESET}"
        )
        for line in freeze.read_text(encoding="utf-8").splitlines():
            if line.startswith("| ") or line.startswith("- "):
                print(f"  {line}")

    rolling = ROLLING_REPORT
    if rolling.exists():
        print()
        print(
            f"  {C.BOLD}Ehrlicher Maßstab: Rolling-Origin{C.RESET}  "
            f"{C.DIM}{rolling.relative_to(ROOT)}{C.RESET}"
        )
        print(
            f"  {C.DIM}Walk-forward: jedes Cutoff-Jahr ist strikt out-of-sample. "
            f"Pooled Combined: ROC 0,79 · PR-AUC 0,055 · Top-10 % fängt ~60 % der Defaults.{C.RESET}"
        )
        for line in rolling.read_text(encoding="utf-8").splitlines()[:12]:
            if line.startswith("- ") or line.startswith("| cut") or (
                line.startswith("| ") and line[2].isdigit()
            ):
                print(f"  {line}")

    print()
    print(f"  {C.DIM}Faustregel: ROC-AUC ~0.5 = Zufall, >0.6 brauchbar, >0.7 stark.{C.RESET}")
    print(
        f"  {C.DIM}Unseen-CIK ist der ehrliche Firm-Holdout; temporaler Test "
        f"enthält oft dieselben Emittenten in späteren Jahren.{C.RESET}"
    )
    print(
        f"  {C.DIM}Vor GJ 2009 ohne XBRL zeigt die UI „nicht scorebar“, "
        f"keine Pseudo-PD aus Median-Imputation.{C.RESET}"
    )
    print(f"  {C.DIM}Nur Bundles mit gleichem trained_at / data vergleichen.{C.RESET}")
    pause()
