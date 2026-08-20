"""Terminal-UX ohne Extra-Dependencies: Farben, Banner, Eingabe, Score-Format."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from secpd.cli import state
from secpd.cli.paths import NATIVE_DEFAULT_HORIZONS, ROOT

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    RESET = "\033[0m" if USE_COLOR else ""
    BOLD = "\033[1m" if USE_COLOR else ""
    DIM = "\033[2m" if USE_COLOR else ""
    CYAN = "\033[36m" if USE_COLOR else ""
    GREEN = "\033[32m" if USE_COLOR else ""
    YELLOW = "\033[33m" if USE_COLOR else ""
    RED = "\033[31m" if USE_COLOR else ""
    MAGENTA = "\033[35m" if USE_COLOR else ""
    BLUE = "\033[34m" if USE_COLOR else ""


def clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def banner() -> None:
    """Kopfzeile inkl. aktivem Ziel (Default / Rating / Fraud)."""
    from secpd.cli.catalog import active_model_meta  # lazy: catalog nutzt C

    print()
    print(f"  {C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}║{C.RESET}  {C.BOLD}SEC-PD{C.RESET}  ·  Risiko-Score aus 10-K MD&A + Finanzen  {C.BOLD}{C.CYAN}║{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════╝{C.RESET}")
    meta = active_model_meta()
    if meta.get("label_source") == "default":
        h = int(meta.get("default_horizon_months") or 12)
        print(f"  {C.DIM}Ziel: {h}-Monats-Ausfallwahrscheinlichkeit je 10-K (Insolvenz-Proxy).{C.RESET}")
    elif meta.get("label_source") == "rating":
        tgt = meta.get("rating_target") or "ordinal"
        if tgt == "ordinal":
            print(
                f"  {C.DIM}Ziel: Shadow-Rating (Notch 1–21). "
                f"PD mit wählbarem Horizont sekundär.{C.RESET}"
            )
        else:
            print(f"  {C.DIM}Ziel: Rating-Label ({tgt}) — Agentur-Note, keine regulatorische PD.{C.RESET}")
    else:
        print(f"  {C.DIM}Misconduct-/Fraud-Risiko (AAER), experimentell — keine regulatorische PD.{C.RESET}")
    print()


def hr() -> None:
    print(f"  {C.DIM}{'─' * 54}{C.RESET}")


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"  {C.BOLD}?{C.RESET} {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return raw if raw else (default or "")


def pause() -> None:
    try:
        input(f"\n  {C.DIM}Enter zum Weiter …{C.RESET}")
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)


def scale_pd(p: float, *, from_months: int, to_months: int) -> float:
    """Termstruktur unter konstanter Hazard-Rate.

    ``PD_t = 1 − (1 − PD_base)^(t / base)``. Nur Approximation, wenn das
    Modell nicht direkt auf ``to_months`` trainiert wurde.
    """
    p = float(min(max(p, 0.0), 1.0 - 1e-15))
    if from_months <= 0 or to_months <= 0:
        raise ValueError("Horizont muss > 0 Monate sein")
    if from_months == to_months:
        return p
    return float(1.0 - (1.0 - p) ** (to_months / from_months))


def _base_rate_12m_from_meta(meta: dict, model_horizon: int) -> float:
    metrics = meta.get("metrics") or {}
    br_raw = metrics.get("base_rate")
    try:
        br = (
            float(br_raw)
            if br_raw is not None and br_raw == br_raw
            else state.DEFAULT_BASE_RATE_12M
        )
    except (TypeError, ValueError):
        br = state.DEFAULT_BASE_RATE_12M
    if model_horizon != 12 and br > 0:
        try:
            return scale_pd(br, from_months=model_horizon, to_months=12)
        except ValueError:
            return br
    return br


def risk_band(
    score: float,
    *,
    label_source: str = "fraud",
    horizon_months: int = 12,
    base_rate_12m: float | None = None,
    mid_mult: float | None = None,
    high_mult: float | None = None,
) -> tuple[str, str]:
    """(Label, Farbe) — bei Default relativ zur erwarteten Basisrate im Horizont."""
    if label_source == "default":
        br = float(base_rate_12m if base_rate_12m is not None else state.DEFAULT_BASE_RATE_12M)
        mid = float(mid_mult if mid_mult is not None else state.RISK_BAND_MID_MULT)
        high = float(high_mult if high_mult is not None else state.RISK_BAND_HIGH_MULT)
        base_h = scale_pd(br, from_months=12, to_months=max(1, horizon_months))
        if score >= high * base_h:
            return "über Basisrate", C.RED
        if score >= mid * base_h:
            return "um Basisrate", C.YELLOW
        return "unter Basisrate", C.GREEN
    if score >= 0.22:
        return "erhöht", C.RED
    if score >= 0.15:
        return "mittel", C.YELLOW
    return "niedrig", C.GREEN


def fmt_score(
    x: float,
    *,
    label_source: str = "fraud",
    horizon_months: int = 12,
    base_rate_12m: float | None = None,
) -> str:
    br = float(base_rate_12m if base_rate_12m is not None else state.DEFAULT_BASE_RATE_12M)
    band, color = risk_band(
        x,
        label_source=label_source,
        horizon_months=horizon_months,
        base_rate_12m=br,
    )
    bar_len = 20
    if label_source == "default":
        base_h = scale_pd(br, from_months=12, to_months=max(1, horizon_months))
        scale = max(3.0 * base_h, 0.02)
        filled = int(round(min(max(x / scale, 0.0), 1.0) * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"{color}{100 * x:.2f} %{C.RESET}  ({x:.4f})  {color}{bar}{C.RESET}  {color}{band}{C.RESET}"
    filled = int(round(min(max(x, 0.0), 1.0) * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"{color}{x:.3f}{C.RESET}  {color}{bar}{C.RESET}  {color}{band}{C.RESET}"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def horizon_label(months: int) -> str:
    if months % 12 == 0 and months >= 12:
        years = months // 12
        return f"{months} M (~{years} J)" if years != 1 else f"{months} M (~1 J)"
    return f"{months} M"


def ask_forecast_horizon(model_horizon: int = 12) -> int:
    """Fragt den Vorausschau-Horizont in Monaten ab."""
    default = state.FORECAST_HORIZON_MONTHS or model_horizon
    print()
    print(f"  {C.BOLD}Prognosehorizont{C.RESET}  {C.DIM}(Monate ab Bilanzstichtag){C.RESET}")
    print(f"  {C.DIM}Modell trainiert auf {model_horizon} M — andere Horizonte via "
          f"konstanter Hazard-Termstruktur.{C.RESET}")
    print(f"  {C.DIM}12 / 24 / 36 M nutzen ein natives Default-Modell, falls vorhanden.{C.RESET}")
    presets = [
        (12, "1 Jahr"),
        (24, "2 Jahre"),
        (36, "3 Jahre"),
        (60, "5 Jahre"),
        (120, "10 Jahre"),
    ]
    for m, label in presets:
        mark = " ← Modell" if m == model_horizon else ""
        print(f"    {C.CYAN}{m:>3}{C.RESET}  {label}{C.DIM}{mark}{C.RESET}")
    print(f"  {C.DIM}oder beliebige monatszahl, z. B. 18 / 84{C.RESET}")
    raw = ask("Horizont in Monaten", str(default))
    try:
        months = int(raw)
        if months < 1 or months > 600:
            raise ValueError
    except ValueError:
        print(f"  {C.YELLOW}Ungültig — nutze {default} M.{C.RESET}")
        months = default
    state.FORECAST_HORIZON_MONTHS = months
    return months


def file_status(path: Path) -> str:
    if not path.exists():
        return f"{C.RED}fehlt{C.RESET}"
    if path.is_dir():
        n = sum(1 for _ in path.glob("*") if _.is_file())
        return f"{C.GREEN}ok{C.RESET} ({n} Dateien)"
    size = path.stat().st_size
    if size >= 1 << 20:
        s = f"{size / (1 << 20):.1f} MB"
    elif size >= 1 << 10:
        s = f"{size / (1 << 10):.0f} KB"
    else:
        s = f"{size} B"
    return f"{C.GREEN}ok{C.RESET} ({s})"


def rel_to_root(path: Path) -> Path | str:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path
