#!/usr/bin/env python3
"""Interaktive Start-Oberfläche für SEC-PD.

Bedienung::

    source .venv/bin/activate
    python start.py

Menü: Unternehmen scoren, Modellgüte, Hilfe, Einstellungen, Beenden.
Automatisiert Suche → Finanz-Join → Textfeatures → Score → lesbare Ausgabe.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
SECRETS_FILE = ROOT / ".secpd.env"

# --------------------------------------------------------------------------- #
# Abhängigkeiten / Umgebung
# --------------------------------------------------------------------------- #


def _bootstrap() -> None:
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print()
        print("  Pandas/sklearn fehlen. Bitte zuerst die virtuelle Umgebung aktivieren:")
        print()
        print("      cd", ROOT)
        print("      source .venv/bin/activate")
        print("      python start.py")
        print()
        raise SystemExit(1)


_bootstrap()

import pandas as pd  # noqa: E402

from secpd.data.edgar import build_financials_panel  # noqa: E402
from secpd.data.events import (  # noqa: E402
    MIN_FYEAR_WITH_FINANCIALS,
    add_event_features,
    annotate_default_labels,
    build_events_table,
    load_events,
)
from secpd.llm.bank import DEFAULT_OPENAI_ENDPOINT, DEFAULT_OPENAI_MODEL  # noqa: E402
from secpd.data.zenodo import resolve_columns  # noqa: E402
from secpd.features.financial import add_financial_features  # noqa: E402
from secpd.features.textual import extract_text_features, text_feature_names  # noqa: E402
from secpd.llm import get_llm_client  # noqa: E402
from secpd.models.ensemble import EnsembleWeights, combine_probabilities  # noqa: E402
from secpd.models.persistence import (  # noqa: E402
    BUNDLE_KIND_ENSEMBLE,
    BUNDLE_KIND_SINGLE,
    bundle_filename,
    bundle_metadata,
    discover_model_paths,
    load_any,
    parse_bundle_name,
)

# --------------------------------------------------------------------------- #
# Pfade & Defaults
# --------------------------------------------------------------------------- #

LABELED = ROOT / "data" / "processed" / "zenodo_labeled.csv.gz"
PANEL = ROOT / "data" / "raw" / "financials_panel.csv"
EVENTS = ROOT / "data" / "raw" / "edgar_8k_events.csv"
FIRM_YEARS = ROOT / "data" / "raw" / "firm_years.json"
FIRM_YEARS_LABELS = ROOT / "data" / "raw" / "firm_years_labels.json"
AAER = ROOT / "data" / "raw" / "aaer_mark5.csv"
SUBMISSIONS_CACHE = ROOT / "data" / "raw" / "edgar_submissions"
MODEL_DIR = ROOT / "models"
SCORES_DIR = ROOT / "data" / "processed"


def load_secrets_env() -> None:
    """Lädt lokale Secrets aus ``.secpd.env`` in os.environ (ohne Überschreiben)."""
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


def save_secrets_env(updates: dict[str, str]) -> None:
    """Schreibt/merged Key-Value in ``.secpd.env`` (gitignored)."""
    existing: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for raw in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()
    existing.update({k: str(v) for k, v in updates.items() if v})
    lines = ["# Lokale SEC-PD Secrets — nicht committen"]
    lines.extend(f"{k}={v}" for k, v in sorted(existing.items()))
    lines.append("")
    SECRETS_FILE.write_text("\n".join(lines), encoding="utf-8")
    try:
        SECRETS_FILE.chmod(0o600)
    except OSError:
        pass


logging.disable(logging.WARNING)

# --------------------------------------------------------------------------- #
# Terminal-UX (ohne Extra-Dependencies)
# --------------------------------------------------------------------------- #

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
    print()
    print(f"  {C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}║{C.RESET}  {C.BOLD}SEC-PD{C.RESET}  ·  Risiko-Score aus 10-K MD&A + Finanzen  {C.BOLD}{C.CYAN}║{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════╝{C.RESET}")
    meta = active_model_meta()
    if meta.get("label_source") == "default":
        h = int(meta.get("default_horizon_months") or 12)
        print(f"  {C.DIM}Ziel: {h}-Monats-Ausfallwahrscheinlichkeit je 10-K (Insolvenz-Proxy).{C.RESET}")
    else:
        print(f"  {C.DIM}Misconduct-/Fraud-Risiko (AAER), keine regulatorische PD.{C.RESET}")
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


_MODEL_META_CACHE: dict | None = None
_ACTIVE_MODEL_PATH: Path | None = None


def list_model_catalog() -> list[dict[str, Any]]:
    """Alle Bundles mit Metadaten (kanonisch + Legacy)."""
    out: list[dict[str, Any]] = []
    for path in discover_model_paths(MODEL_DIR):
        try:
            payload = load_any(path)
            md = dict(bundle_metadata(payload))
            kind = payload.get("kind")
        except Exception as exc:  # noqa: BLE001
            md, kind = {"_load_error": str(exc)}, None
        parsed = parse_bundle_name(path)
        label = md.get("label_source") or parsed.get("label_source")
        horizon = md.get("default_horizon_months")
        if horizon is None:
            horizon = parsed.get("horizon_months")
        component = md.get("component") or md.get("mode") or parsed.get("component")
        out.append(
            {
                "path": path,
                "name": path.name,
                "kind": kind,
                "metadata": md,
                "component": component,
                "label_source": label,
                "horizon_months": int(horizon) if horizon is not None else None,
                "legacy": bool(parsed.get("legacy")),
                "trained_at": md.get("trained_at_utc") or md.get("created_at_utc"),
                "data": md.get("data"),
            }
        )
    return out


def warn_model_coherence(catalog: list[dict] | None = None) -> None:
    """Warnt, wenn Financial/Combined nicht aus demselben Trainingslauf stammen."""
    cat = catalog if catalog is not None else list_model_catalog()
    # Legacy-Dateien in der Warnung ignorieren, wenn kanonische Pendants existieren
    canon = [r for r in cat if not r.get("legacy")]
    use = canon or cat
    by_comp: dict[str, list[dict]] = {}
    for row in use:
        comp = str(row.get("component") or "")
        if comp in {"financial", "combined", "ensemble"}:
            by_comp.setdefault(comp, []).append(row)
    fins = by_comp.get("financial") or []
    combs = by_comp.get("combined") or []
    if not fins or not combs:
        return

    def _same_run(a: dict, b: dict) -> bool:
        ma, mb = a.get("metadata") or {}, b.get("metadata") or {}
        ra, rb = ma.get("train_run_id"), mb.get("train_run_id")
        if ra and rb:
            return ra == rb
        # Fallback: gleicher data/label/horizon und timestamps ≤ 2 Min auseinander
        if str(a.get("data") or "") != str(b.get("data") or ""):
            return False
        if a.get("label_source") != b.get("label_source"):
            return False
        if a.get("horizon_months") != b.get("horizon_months"):
            return False
        ta, tb = a.get("trained_at") or "", b.get("trained_at") or ""
        if not ta or not tb:
            return False
        try:
            import pandas as pd

            delta = abs(pd.Timestamp(ta) - pd.Timestamp(tb)).total_seconds()
            return delta <= 120
        except Exception:  # noqa: BLE001
            return ta == tb

    for c in combs:
        peers = [
            f
            for f in fins
            if f.get("label_source") == c.get("label_source")
            and f.get("horizon_months") == c.get("horizon_months")
        ]
        if not peers:
            print(
                f"  {C.YELLOW}Hinweis: {c['name']} ohne passendes financial-Pendant "
                f"(label={c.get('label_source')}, h={c.get('horizon_months')}) — "
                f"Gütevergleiche über Bundles sind unsicher.{C.RESET}"
            )
            continue
        for f in peers:
            if not _same_run(c, f):
                print(
                    f"  {C.YELLOW}Warnung: {c['name']} und {f['name']} stammen "
                    f"nicht aus demselben Lauf "
                    f"(trained_at {c.get('trained_at')} vs {f.get('trained_at')}).{C.RESET}"
                )


def select_model_path(
    *,
    prefer: str = "combined",
    label_source: str | None = None,
    horizon_months: int | None = None,
    has_text: bool = True,
) -> Path | None:
    """Wählt das beste passende Bundle; Legacy nur als Fallback."""
    cat = list_model_catalog()
    if not cat:
        return None

    want_label = label_source
    want_h = horizon_months
    prefer_comp = "combined" if has_text and prefer == "combined" else "financial"
    if not has_text:
        prefer_comp = "financial"

    def score(row: dict) -> tuple:
        # higher is better
        comp = str(row.get("component") or "")
        lab = row.get("label_source")
        hor = row.get("horizon_months")
        legacy = 1 if row.get("legacy") else 0
        s = 0
        if comp == prefer_comp:
            s += 100
        elif comp == "financial" and prefer_comp == "combined":
            s += 40
        elif comp == "combined" and prefer_comp == "financial":
            s += 20
        if want_label and lab == want_label:
            s += 50
        elif lab == "default":
            s += 10
        if want_h is not None and hor == want_h:
            s += 30
        elif want_h is None and hor == 12:
            s += 5
        s -= legacy * 80  # stark gegen Legacy
        # neuere Modelle bevorzugen
        ts = str(row.get("trained_at") or "")
        return (s, ts)

    ranked = sorted(cat, key=score, reverse=True)
    best = ranked[0]
    # Wenn Legacy gewinnt, aber kanonische existieren: kanonische bevorzugen
    canon = [r for r in cat if not r.get("legacy")]
    if best.get("legacy") and canon:
        best = sorted(canon, key=score, reverse=True)[0]
    return Path(best["path"])


def active_model_path(*, refresh: bool = False) -> Path | None:
    global _ACTIVE_MODEL_PATH
    if _ACTIVE_MODEL_PATH is not None and not refresh and _ACTIVE_MODEL_PATH.exists():
        return _ACTIVE_MODEL_PATH
    path = select_model_path(prefer="combined", label_source="default", horizon_months=12)
    if path is None:
        path = select_model_path(prefer="financial")
    _ACTIVE_MODEL_PATH = path
    return path


def active_model_meta(*, refresh: bool = False) -> dict:
    global _MODEL_META_CACHE
    if _MODEL_META_CACHE is not None and not refresh:
        return _MODEL_META_CACHE
    path = active_model_path(refresh=refresh)
    if path is None:
        _MODEL_META_CACHE = {}
        return _MODEL_META_CACHE
    try:
        _MODEL_META_CACHE = dict(bundle_metadata(load_any(path)))
    except Exception:  # noqa: BLE001
        _MODEL_META_CACHE = {}
    return _MODEL_META_CACHE


def _describe_model(path: Path) -> str:
    try:
        md = bundle_metadata(load_any(path))
    except Exception as exc:  # noqa: BLE001
        return f"{path.name} (Ladefehler: {exc})"
    ts = md.get("trained_at_utc") or md.get("created_at_utc") or "?"
    lab = md.get("label_source", "?")
    mode = md.get("mode") or md.get("component") or "?"
    h = md.get("default_horizon_months")
    data = md.get("data") or "?"
    h_s = f", h={h}M" if h is not None else ""
    return f"{path.name}  [{mode}/{lab}{h_s}]  trained={ts}  data={data}"


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


# Session-Default für Vorausschau-Horizont (Monate); None = Modellhorizont.
FORECAST_HORIZON_MONTHS: int | None = None

# Risiko-Bänder relativ zur Basisrate im Horizont (Multiplikatoren).
# unter < mid_mult × Basis | mid_mult…high_mult × Basis = um Basisrate | ≥ high = über
RISK_BAND_MID_MULT: float = 0.85
RISK_BAND_HIGH_MULT: float = 2.5
# Fallback-Basisrate 12M, falls Modell-Metadaten keine liefern
DEFAULT_BASE_RATE_12M: float = 0.012


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
        br = float(base_rate_12m if base_rate_12m is not None else DEFAULT_BASE_RATE_12M)
        mid = float(mid_mult if mid_mult is not None else RISK_BAND_MID_MULT)
        high = float(high_mult if high_mult is not None else RISK_BAND_HIGH_MULT)
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
    br = float(base_rate_12m if base_rate_12m is not None else DEFAULT_BASE_RATE_12M)
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
    global FORECAST_HORIZON_MONTHS
    default = FORECAST_HORIZON_MONTHS or model_horizon
    print()
    print(f"  {C.BOLD}Prognosehorizont{C.RESET}  {C.DIM}(Monate ab Bilanzstichtag){C.RESET}")
    print(f"  {C.DIM}Modell trainiert auf {model_horizon} M — andere Horizonte via "
          f"konstanter Hazard-Termstruktur.{C.RESET}")
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
    FORECAST_HORIZON_MONTHS = months
    return months


def _file_status(path: Path) -> str:
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


def _run_script(argv: list[str], *, title: str) -> int:
    """Führt ein Skript/CLI im Projektroot aus und streamt die Ausgabe."""
    hr()
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"  {C.DIM}$ {' '.join(argv)}{C.RESET}")
    hr()
    print()
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), check=False)
    except KeyboardInterrupt:
        print(f"\n  {C.YELLOW}Abgebrochen.{C.RESET}")
        return 130
    code = int(proc.returncode)
    print()
    if code == 0:
        print(f"  {C.GREEN}Fertig (Exit {code}).{C.RESET}")
    else:
        print(f"  {C.RED}Fehlgeschlagen (Exit {code}).{C.RESET}")
    return code


def _ensure_sec_ua() -> str | None:
    ua = os.environ.get("SECPD_SEC_UA", "").strip()
    if ua:
        return ua
    print(f"  {C.YELLOW}SECPD_SEC_UA ist nicht gesetzt (SEC-Pflicht).{C.RESET}")
    ua = ask(
        "User-Agent (Firma email@…)",
        "Commerzbank Praktikum vorname.nachname@example.com",
    ).strip()
    if not ua:
        print(f"  {C.RED}Ohne User-Agent kein EDGAR-Abruf.{C.RESET}")
        return None
    os.environ["SECPD_SEC_UA"] = ua
    return ua


# --------------------------------------------------------------------------- #
# Daten & Scoring
# --------------------------------------------------------------------------- #


def _require_files() -> list[str]:
    missing = []
    if not LABELED.exists():
        missing.append(str(LABELED.relative_to(ROOT)))
    if not PANEL.exists():
        missing.append(str(PANEL.relative_to(ROOT)))
    if not list_model_catalog():
        missing.append("models/*.joblib (bitte zuerst trainieren)")
    return missing


def load_company_index() -> pd.DataFrame:
    df = pd.read_csv(LABELED, usecols=["cik", "name"])
    return df.drop_duplicates(subset=["cik", "name"]).sort_values("name")


def search_companies(query: str, index: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    q = query.strip()
    if not q:
        return index.head(0)
    if q.isdigit():
        cik = int(q)
        return index[index["cik"] == cik].head(limit)
    mask = index["name"].astype(str).str.contains(q, case=False, na=False, regex=False)
    return index.loc[mask].head(limit)


def load_firm_years_for_cik(cik: int, max_chars: int = 40_000) -> pd.DataFrame:
    """Lädt Firm-Years aus dem Label-Set; Fallback firm_years.json."""
    labeled = pd.read_csv(LABELED)
    sub = labeled[labeled["cik"] == cik].copy()
    source = "label-set"
    if sub.empty and FIRM_YEARS.exists():
        source = "firm_years.json"
        rows: list[dict] = []
        with FIRM_YEARS.open("r", encoding="utf-8") as fh:
            fh.readline()  # [
            needle = f'"cik": {cik}'
            needle2 = f'"cik":{cik}'
            for line in fh:
                if needle not in line and needle2 not in line:
                    continue
                line = line.strip().rstrip(",")
                if not line or line == "]":
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(rec.get("cik") or 0) != cik:
                    continue
                text = str(rec.get("mda") or "")
                if len(text) < 200:
                    continue
                rows.append(
                    {
                        "cik": rec.get("cik"),
                        "name": rec.get("name"),
                        "sic": rec.get("sic"),
                        "filing_type": rec.get("filing_type"),
                        "filing_date": rec.get("filing_date"),
                        "reporting_date": rec.get("reporting_date"),
                        "mda": text[:max_chars],
                        "text_chars": min(len(text), max_chars),
                    }
                )
        if not rows:
            return pd.DataFrame()
        sub = pd.DataFrame(rows)
        sub["filing_date"] = pd.to_datetime(sub["filing_date"], format="%d-%m-%Y", errors="coerce")
        sub["reporting_date"] = pd.to_datetime(
            sub["reporting_date"], format="%d-%m-%Y", errors="coerce"
        )
        sub["cik"] = pd.to_numeric(sub["cik"], errors="coerce").astype("Int64")
        sub = sub.dropna(subset=["cik", "reporting_date"]).reset_index(drop=True)
        sub["fyear"] = sub["reporting_date"].dt.year.astype(int)
        sub["doc_id"] = (
            sub["cik"].astype(int).astype(str)
            + "_"
            + sub["reporting_date"].dt.strftime("%Y-%m-%d")
        )
        sub = sub.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
        if "label" not in sub.columns:
            sub["label"] = 0
    elif not sub.empty:
        for col in ("filing_date", "reporting_date"):
            if col in sub.columns and not pd.api.types.is_datetime64_any_dtype(sub[col]):
                sub[col] = pd.to_datetime(sub[col], errors="coerce")
    if sub.empty:
        return sub
    sub.attrs["source"] = source
    return sub


def attach_financials(df: pd.DataFrame, *, fetch_if_missing: bool = True) -> pd.DataFrame:
    panel = pd.read_csv(PANEL)
    panel.columns = [c.lower() for c in panel.columns]
    out = df.merge(panel, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
    coverage = float(out["total_assets"].notna().mean()) if "total_assets" in out.columns else 0.0
    if coverage < 0.3 and fetch_if_missing:
        ciks = sorted({int(c) for c in out["cik"].dropna().unique()})
        print(f"  {C.YELLOW}Wenige Finanzdaten im Panel ({coverage:.0%}).{C.RESET}")
        if ask("EDGAR jetzt nachladen? (j/n)", "j").lower().startswith("j"):
            ua = os.environ.get(
                "SECPD_SEC_UA",
                "Commerzbank Praktikum vorname.nachname@example.com",
            )
            print(f"  {C.DIM}Lade companyfacts für CIK {ciks} …{C.RESET}")
            try:
                fresh = build_financials_panel(ciks, user_agent=ua)
                if not fresh.empty:
                    out = df.merge(fresh, on=["cik", "fyear"], how="left", suffixes=("", "_fin"))
                    print(f"  {C.GREEN}EDGAR-Panel: {len(fresh)} Firm-Years{C.RESET}")
            except Exception as exc:  # noqa: BLE001 — UX: Fehler anzeigen, nicht crashen
                print(f"  {C.RED}EDGAR-Abruf fehlgeschlagen: {exc}{C.RESET}")
    return out


def score_frame(
    df: pd.DataFrame,
    model_path: Path,
    *,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    payload = load_any(model_path)
    work = df.copy()
    if "label" not in work.columns:
        work["label"] = 0
    cols = resolve_columns(work)
    work = work.reset_index(drop=True)
    work, _ = add_financial_features(work)

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        feature_cols = list(payload["feature_cols"])
        needs_llm = any(c in feature_cols for c in text_feature_names())
    elif payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        feature_cols = list(payload["financial"]["feature_cols"])
        needs_llm = True
    else:
        raise ValueError(f"Unbekannter Bundle-Typ: {payload['kind']!r}")

    needs_events = any(str(c).startswith("evt_") for c in feature_cols)
    if needs_events:
        ev = events_df
        if ev is None and EVENTS.exists():
            ev = load_events(EVENTS)
        if ev is None or ev.empty:
            print(f"  {C.YELLOW}Keine 8-K-Events — Event-Features werden 0/NaN "
                  f"(Imputation).{C.RESET}")
            ev = pd.DataFrame(columns=["cik", "filing_date_8k", "items", "accession"])
        else:
            print(f"  {C.DIM}8-K-Event-Features ({len(ev)} Events) …{C.RESET}")
        work, _ = add_event_features(work, ev)

    if needs_llm:
        if cols.text_col is None or cols.text_col not in work.columns:
            raise RuntimeError("Modell braucht Text (MD&A), aber keine Textspalte gefunden.")
        if work[cols.text_col].fillna("").astype(str).str.len().max() < 50:
            raise RuntimeError("MD&A-Text fehlt oder ist zu kurz für das Combined-Modell.")
        llm_mode = os.environ.get("SECPD_LLM_MODE", "mock")
        client = get_llm_client(llm_mode)
        print(f"  {C.DIM}Textanalyse ({len(work)} Dokumente, LLM={llm_mode}) …{C.RESET}")
        text_feats = extract_text_features(
            work,
            client=client,
            text_col=cols.text_col,
            id_col=cols.id_col,
            progress_every=10_000,
        )
        work = work.merge(text_feats, on=cols.id_col, how="left")

    # Fehlende Feature-Spalten (z. B. fin_interest_coverage ohne ebit in EDGAR)
    # als NaN anlegen — SimpleImputer in der Pipeline füllt sie.
    all_needed: list[str] = list(feature_cols)
    if payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        all_needed = list(
            dict.fromkeys(
                list(payload["financial"]["feature_cols"])
                + list(payload["text"]["feature_cols"])
            )
        )
    missing = [c for c in all_needed if c not in work.columns]
    if missing:
        print(f"  {C.DIM}Fehlende Features als NaN: {', '.join(missing)}{C.RESET}")
        for c in missing:
            work[c] = float("nan")

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        scores = payload["pipeline"].predict_proba(work)[:, 1]
    else:
        p_fin = payload["financial"]["pipeline"].predict_proba(work)[:, 1]
        p_txt = payload["text"]["pipeline"].predict_proba(work)[:, 1]
        w = payload.get("weights", {})
        weights = EnsembleWeights(
            w_financial=float(w.get("w_financial", 0.6)),
            w_text=float(w.get("w_text", 0.4)),
        )
        scores = combine_probabilities(p_fin, p_txt, weights, method="logit")

    out = work[[cols.id_col]].copy()
    out["pd_score"] = scores
    if "fyear" in work.columns:
        out["fyear"] = work["fyear"].values
    if "name" in work.columns:
        out["name"] = work["name"].values
    if "label" in work.columns:
        out["label"] = work["label"].values
    meta = dict(payload.get("metadata") or {})
    if meta.get("label_source") == "default":
        ev = events_df
        if ev is None and EVENTS.exists():
            ev = load_events(EVENTS)
        if ev is not None and not ev.empty and "reporting_date" in work.columns:
            horizon = int(meta.get("default_horizon_months") or 12)
            trust_legacy = bool(meta.get("trust_legacy_regime", False))
            annotated = annotate_default_labels(
                work,
                ev,
                horizon_months=horizon,
                trust_legacy_regime=trust_legacy,
            )
            out["label_default"] = annotated["label_default"].values
            out["bankruptcy_date"] = annotated["bankruptcy_date"].values
        elif "label_default" in work.columns:
            out["label_default"] = work["label_default"].values
    if "total_assets" in work.columns:
        out["has_financials"] = work["total_assets"].notna().values
    for col in ("filing_date", "reporting_date"):
        if col in work.columns:
            out[col] = work[col].values
    return out.sort_values("fyear" if "fyear" in out.columns else cols.id_col)


def _fmt_date(val: object) -> str:
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d")


def pick_tenk(firm: pd.DataFrame) -> pd.Series | None:
    """Interaktive Auswahl eines 10-K (Firm-Year). Default: neuestes."""
    view = firm.sort_values("fyear", ascending=False).reset_index(drop=True)
    print()
    print(f"  {C.BOLD}Verfügbare 10-Ks{C.RESET}  {C.DIM}(je ein Geschäftsjahr){C.RESET}")
    show_n = min(12, len(view))
    for i in range(show_n):
        row = view.iloc[i]
        year = int(row["fyear"]) if pd.notna(row.get("fyear")) else "?"
        mark = "  ← neuestes" if i == 0 else ""
        filing = _fmt_date(row.get("filing_date"))
        report = _fmt_date(row.get("reporting_date"))
        print(
            f"    {C.CYAN}{i + 1:2d}{C.RESET}  GJ {year}  ·  "
            f"Filing {filing}  ·  Bilanz {report}{C.DIM}{mark}{C.RESET}"
        )
    if len(view) > show_n:
        print(f"    {C.DIM}… +{len(view) - show_n} ältere (Nummer oder GJ-Jahr eingeben){C.RESET}")
    print(f"    {C.CYAN} 0{C.RESET}  Abbrechen")
    print()
    choice = ask("10-K wählen (Nr. oder GJ)", "1")
    if choice in {"0", "q"}:
        return None
    try:
        if len(choice) == 4 and choice.isdigit():
            year = int(choice)
            hits = view[view["fyear"] == year]
            if hits.empty:
                print(f"  {C.RED}Kein 10-K für GJ {year}.{C.RESET}")
                return None
            return hits.iloc[0]
        idx = int(choice) - 1
        if idx < 0 or idx >= len(view):
            raise IndexError
        return view.iloc[idx]
    except (ValueError, IndexError):
        print(f"  {C.RED}Ungültige Auswahl.{C.RESET}")
        return None


def print_score_table(
    result: pd.DataFrame,
    company_name: str,
    *,
    focus_doc_id: str | None = None,
    meta: dict | None = None,
    forecast_horizon_months: int | None = None,
) -> None:
    meta = meta or {}
    label_source = str(meta.get("label_source") or "fraud")
    model_horizon = int(meta.get("default_horizon_months") or 12)
    horizon = int(forecast_horizon_months or model_horizon)
    metrics = meta.get("metrics") or {}
    br_raw = metrics.get("base_rate")
    try:
        base_rate_12m = (
            float(br_raw) if br_raw is not None and br_raw == br_raw else DEFAULT_BASE_RATE_12M
        )
    except (TypeError, ValueError):
        base_rate_12m = DEFAULT_BASE_RATE_12M
    # Metrik = Basisrate des Trainings-Horizonts → auf 12M zurückrechnen
    if model_horizon != 12 and base_rate_12m > 0:
        try:
            base_rate_12m = scale_pd(base_rate_12m, from_months=model_horizon, to_months=12)
        except ValueError:
            pass

    hr()
    print(f"  {C.BOLD}Ergebnis · {company_name}{C.RESET}")
    hr()
    if result.empty:
        print(f"  {C.YELLOW}Keine Scores.{C.RESET}")
        return

    focus = result.iloc[-1]
    if focus_doc_id and "doc_id" in result.columns:
        hit = result[result["doc_id"].astype(str) == str(focus_doc_id)]
        if not hit.empty:
            focus = hit.iloc[0]

    score_model = float(focus["pd_score"])
    score = scale_pd(score_model, from_months=model_horizon, to_months=horizon)
    scaled = horizon != model_horizon
    year = focus.get("fyear", "?")
    year_s = str(int(year)) if pd.notna(year) and str(year) != "?" else "?"
    filing = _fmt_date(focus.get("filing_date")) if "filing_date" in focus.index else "—"
    report = _fmt_date(focus.get("reporting_date")) if "reporting_date" in focus.index else "—"
    report_ts = (
        pd.to_datetime(focus.get("reporting_date"), errors="coerce")
        if "reporting_date" in focus.index
        else pd.NaT
    )

    print()
    if label_source == "default":
        base_h = scale_pd(base_rate_12m, from_months=12, to_months=horizon)
        lift = score / base_h if base_h > 0 else float("nan")
        print(f"  {C.BOLD}Ausfallwahrscheinlichkeit · Horizont {horizon_label(horizon)}{C.RESET}")
        print(f"  {C.DIM}Basis: 10-K GJ {year_s}  ·  Filing {filing}  ·  Bilanzstichtag {report}{C.RESET}")
        print(
            f"    {fmt_score(score, label_source=label_source, horizon_months=horizon, base_rate_12m=base_rate_12m)}"
        )
        if pd.notna(report_ts):
            end = report_ts + pd.DateOffset(months=horizon)
            print(
                f"  {C.DIM}Fenster: ({report}, {_fmt_date(end)}] — "
                f"P(Insolvenz-Meldung in diesem Zeitraum).{C.RESET}"
            )
        print(
            f"  {C.DIM}Vergleich: Sample-Basisrate ≈ {100 * base_h:.2f} % "
            f"über {horizon_label(horizon)}  ·  Lift {lift:.2f}×{C.RESET}"
        )
        if scaled:
            print(
                f"  {C.YELLOW}Abgeleitet aus {model_horizon}-M-Modell-PD "
                f"({100 * score_model:.2f} %) unter konstanter Hazard-Rate "
                f"— nicht separat kalibriert.{C.RESET}"
            )
        else:
            print(f"  {C.DIM}Direktes Modell-Output (trainiert auf {model_horizon} M).{C.RESET}")

        grid = sorted({12, 24, 36, 60, 120, model_horizon, horizon})
        print()
        print(f"  {C.BOLD}Vorausschau (Termstruktur){C.RESET}")
        print(f"  {'Horizont':<14} {'von':<12} {'bis':<12} {'PD':>7}  {'vs Basis':<16}")
        print(f"  {C.DIM}{'-' * 66}{C.RESET}")
        for m in grid:
            pd_m = scale_pd(score_model, from_months=model_horizon, to_months=m)
            band, color = risk_band(
                pd_m,
                label_source="default",
                horizon_months=m,
                base_rate_12m=base_rate_12m,
            )
            if pd.notna(report_ts):
                von = report_ts.strftime("%Y-%m-%d")
                bis = (report_ts + pd.DateOffset(months=m)).strftime("%Y-%m-%d")
            else:
                von, bis = "—", "—"
            mark = " ←" if m == horizon else (" ◇" if m == model_horizon else "")
            print(
                f"  {horizon_label(m):<14} {von:<12} {bis:<12} "
                f"{color}{100 * pd_m:6.2f}%{C.RESET}  {color}{band:<16}{C.RESET}"
                f"{C.DIM}{mark}{C.RESET}"
            )
        print(
            f"  {C.DIM}← gewählt · ◇ Modellhorizont · "
            f"Fenster = (Bilanzstichtag, Stichtag + Horizont]{C.RESET}"
        )

        print()
        print(f"  {C.DIM}Proxy: 8-K Item 1.03 / alt. Item 3 (Chapter 11) — "
              f"keine regulatorische PD.{C.RESET}")
        print(f"  {C.DIM}Für echte multi-Jahr-Labels: Einstellungen → Training "
              f"mit --default-horizon {horizon}.{C.RESET}")
    else:
        print(f"  {C.BOLD}Misconduct-/Fraud-Score{C.RESET} · GJ {year_s}")
        print(f"    {fmt_score(score_model, label_source=label_source)}")
        print()
        print(f"  {C.DIM}Interpretation: AAER-Risiko, keine regulatorische PD.{C.RESET}")

    print()
    label_hdr = "Default" if label_source == "default" else "AAER"
    score_hdr = f"PD%{horizon}M" if label_source == "default" else "Score"
    window_hdr = "Fenster" if label_source == "default" else ""
    print(
        f"  {'Jahr':<6} {score_hdr:>8}  {'Risiko':<16}  {label_hdr:<7}  "
        f"{'Event':<12} {window_hdr:<23} Verlauf"
    )
    print(f"  {C.DIM}{'-' * 88}{C.RESET}")
    for _, row in result.iterrows():
        s_model = float(row["pd_score"])
        s = (
            scale_pd(s_model, from_months=model_horizon, to_months=horizon)
            if label_source == "default"
            else s_model
        )
        band, color = risk_band(
            s,
            label_source=label_source,
            horizon_months=horizon,
            base_rate_12m=base_rate_12m,
        )
        if label_source == "default":
            if "label_default" in row.index and pd.notna(row["label_default"]):
                label = int(row["label_default"])
            else:
                label = -1
        else:
            label = int(row["label"]) if "label" in row.index and pd.notna(row["label"]) else -1
        if label == 1:
            flag = f"{C.RED}JA   {C.RESET}"
        elif label == 0:
            flag = "nein "
        else:
            flag = "–    "
        event_cell = "—"
        window_cell = ""
        if label_source == "default":
            event_cell = _fmt_date(row.get("bankruptcy_date")) if "bankruptcy_date" in row.index else "—"
            if label != 1:
                event_cell = "—"
            rep_row = (
                pd.to_datetime(row.get("reporting_date"), errors="coerce")
                if "reporting_date" in row.index
                else pd.NaT
            )
            if pd.notna(rep_row):
                end_row = rep_row + pd.DateOffset(months=horizon)
                window_cell = f"({_fmt_date(rep_row)}, {_fmt_date(end_row)}]"
            else:
                window_cell = "—"
        base_h = scale_pd(base_rate_12m, from_months=12, to_months=horizon)
        scale = max(3.0 * base_h, 0.02) if label_source == "default" else 1.0
        spark = "▂▃▄▅▆"[min(4, int(min(s / scale, 1.0) * 5))]
        y = int(row["fyear"]) if "fyear" in row.index and pd.notna(row["fyear"]) else "?"
        marker = " ←" if focus_doc_id and str(row.get("doc_id")) == str(focus_doc_id) else ""
        score_cell = f"{100 * s:7.2f}%" if label_source == "default" else f"{s:8.3f}"
        print(
            f"  {y:<6} {color}{score_cell}{C.RESET}  {color}{band:<16}{C.RESET}  "
            f"{flag}  {event_cell:<12} {window_cell:<23} "
            f"{color}{spark}{C.RESET}{C.DIM}{marker}{C.RESET}"
        )

    print()
    print(
        f"  Mittel {result['pd_score'].mean():.4f}  ·  "
        f"Min {result['pd_score'].min():.4f}  ·  "
        f"Max {result['pd_score'].max():.4f}  {C.DIM}(Modell-{model_horizon}M){C.RESET}"
    )
    if label_source == "default" and "label_default" in result.columns:
        pos = result[result["label_default"] == 1]
        if len(pos):
            print(
                f"  Default-Jahre (12M-Label): "
                f"{', '.join(str(int(y)) for y in pos['fyear'])} "
                f"(Ø-Score {pos['pd_score'].mean():.3f})"
            )
    elif label_source != "default" and "label" in result.columns and result["label"].isin([0, 1]).any():
        pos = result[result["label"] == 1]
        if len(pos):
            print(
                f"  AAER-Jahre im Datensatz: {', '.join(str(int(y)) for y in pos['fyear'])} "
                f"(Ø-Score {pos['pd_score'].mean():.3f})"
            )


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
        if row.get("legacy"):
            title += "  [LEGACY]"
        print(f"\n  {C.BOLD}{C.BLUE}{title}{C.RESET}  {C.DIM}{path.name}{C.RESET}")
        print(f"    trained   {row.get('trained_at') or md.get('created_at_utc') or '—'}")
        print(f"    data      {row.get('data') or '—'}")
        print(
            f"    split     {md.get('split_strategy', '?')}  ·  "
            f"kalibriert={md.get('calibrated', '?')}  ·  mode={md.get('mode', '?')}"
        )
        if not metrics:
            print(f"  {C.DIM}Keine Metriken im Bundle.{C.RESET}")
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
        if md.get("min_fyear") is not None:
            print(
                f"    policy    min_fyear={md.get('min_fyear')}  ·  "
                f"legacy={md.get('trust_legacy_regime', '?')}  ·  "
                f"require_fin={md.get('require_financials', '?')}"
            )

    freeze = ROOT / "benchmarks" / "default_h12_clean" / "REPORT.md"
    if freeze.exists():
        print()
        print(f"  {C.BOLD}Frozen Benchmark{C.RESET}  {C.DIM}{freeze.relative_to(ROOT)}{C.RESET}")
        for line in freeze.read_text(encoding="utf-8").splitlines():
            if line.startswith("| ") or line.startswith("- "):
                print(f"  {line}")

    rolling = ROOT / "benchmarks" / "rolling_default_h12" / "REPORT.md"
    if rolling.exists():
        print()
        print(f"  {C.BOLD}Rolling-Origin{C.RESET}  {C.DIM}{rolling.relative_to(ROOT)}{C.RESET}")
        for line in rolling.read_text(encoding="utf-8").splitlines()[:12]:
            if line.startswith("- ") or line.startswith("| cut") or (
                line.startswith("| ") and line[2].isdigit()
            ):
                print(f"  {line}")

    print()
    print(f"  {C.DIM}Faustregel: ROC-AUC ~0.5 = Zufall, >0.6 brauchbar, >0.7 stark.{C.RESET}")
    print(f"  {C.DIM}Nur Bundles mit gleichem trained_at / data vergleichen.{C.RESET}")
    pause()


def _normalize_firm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal-Normalisierung für Scoring-Eingaben."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "cik" in out.columns:
        out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    if "fyear" not in out.columns and "reporting_date" in out.columns:
        rd = pd.to_datetime(out["reporting_date"], errors="coerce")
        out["fyear"] = rd.dt.year
    if "fyear" in out.columns:
        out["fyear"] = pd.to_numeric(out["fyear"], errors="coerce").astype("Int64")
    for col in ("filing_date", "reporting_date"):
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=False)
            # Fallback DD-MM-YYYY (Zenodo)
            if out[col].isna().all():
                out[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    if "doc_id" not in out.columns:
        if "cik" in out.columns and "reporting_date" in out.columns:
            out["doc_id"] = (
                out["cik"].astype(str)
                + "_"
                + pd.to_datetime(out["reporting_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            )
        elif "cik" in out.columns and "fyear" in out.columns:
            out["doc_id"] = out["cik"].astype(str) + "_" + out["fyear"].astype(str)
        else:
            out["doc_id"] = [f"row_{i}" for i in range(len(out))]
    if "name" not in out.columns:
        out["name"] = out.get("cik", pd.Series(["unbekannt"] * len(out))).astype(str)
    if "mda" not in out.columns:
        for alt in ("text", "mda_text", "md_a", "content"):
            if alt in out.columns:
                out = out.rename(columns={alt: "mda"})
                break
    if "label" not in out.columns:
        out["label"] = 0
    return out.reset_index(drop=True)


def _ask_float(prompt: str) -> float | None:
    raw = ask(prompt + " (leer = unbekannt)", "")
    if not raw:
        return None
    try:
        return float(raw.replace(",", "").replace(" ", ""))
    except ValueError:
        print(f"  {C.YELLOW}Ungültige Zahl — übersprungen.{C.RESET}")
        return None


def _read_mda_input() -> str:
    print(f"  {C.DIM}MD&A: Dateipfad, 'paste' für Mehrzeilen-Eingabe, oder leer.{C.RESET}")
    raw = ask("MD&A-Quelle", "")
    if not raw:
        return ""
    if raw.lower() == "paste":
        print(f"  {C.DIM}Text einfügen, danach eine Zeile mit nur END:{C.RESET}")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines).strip()
    path = Path(raw).expanduser()
    if not path.is_file():
        print(f"  {C.YELLOW}Datei nicht gefunden — ohne MD&A weiter.{C.RESET}")
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _choose_model_path(*, has_text: bool) -> Path:
    meta = active_model_meta()
    label = str(meta.get("label_source") or "default")
    horizon = meta.get("default_horizon_months")
    try:
        horizon_i = int(horizon) if horizon is not None else 12
    except (TypeError, ValueError):
        horizon_i = 12
    path = select_model_path(
        prefer="combined" if has_text else "financial",
        label_source=label,
        horizon_months=horizon_i if label == "default" else None,
        has_text=has_text,
    )
    if path is None:
        raise FileNotFoundError("Kein trainiertes Modell gefunden.")
    if not has_text and "combined" in path.name:
        alt = select_model_path(
            prefer="financial",
            label_source=label,
            horizon_months=horizon_i if label == "default" else None,
            has_text=False,
        )
        if alt is not None:
            print(f"  {C.YELLOW}Kein MD&A — nutze {alt.name}.{C.RESET}")
            return alt
    return path


def maybe_save_csv(
    result: pd.DataFrame,
    default_name: str,
    *,
    model_horizon: int = 12,
    forecast_horizon: int | None = None,
) -> None:
    if not ask("Ergebnis als CSV speichern? (j/n)", "n").lower().startswith("j"):
        return
    out = result.copy()
    if forecast_horizon and "pd_score" in out.columns:
        out["pd_score_model_h"] = model_horizon
        out["pd_horizon_months"] = forecast_horizon
        out["pd_score_horizon"] = [
            scale_pd(float(p), from_months=model_horizon, to_months=forecast_horizon)
            for p in out["pd_score"]
        ]
        for m in (12, 24, 60, 120):
            out[f"pd_{m}m"] = [
                scale_pd(float(p), from_months=model_horizon, to_months=m)
                for p in out["pd_score"]
            ]
    default_path = SCORES_DIR / default_name
    raw = ask("Pfad", str(default_path.relative_to(ROOT)))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"  {C.GREEN}Gespeichert: {shown}{C.RESET}")


def run_scoring(
    firm: pd.DataFrame,
    company_name: str,
    *,
    events_df: pd.DataFrame | None = None,
    default_csv: str | None = None,
) -> None:
    firm = _normalize_firm_frame(firm)
    if firm.empty:
        print(f"  {C.RED}Keine Daten zum Scoren.{C.RESET}")
        pause()
        return

    focus_row = None
    if len(firm) > 1 and "fyear" in firm.columns:
        focus_row = pick_tenk(firm)
        if focus_row is None:
            pause()
            return
        # Alle Jahre scoren (Verlauf), Fokus auf Auswahl
        focus_doc = str(focus_row.get("doc_id", ""))
    else:
        focus_doc = str(firm.iloc[0].get("doc_id", "")) if len(firm) else None

    has_text = "mda" in firm.columns and firm["mda"].fillna("").astype(str).str.len().max() >= 50
    try:
        model_path = _choose_model_path(has_text=bool(has_text))
    except FileNotFoundError as exc:
        print(f"  {C.RED}{exc}{C.RESET}")
        pause()
        return

    meta = dict(load_any(model_path).get("metadata") or {})
    model_horizon = int(meta.get("default_horizon_months") or 12)
    forecast_h = model_horizon
    if meta.get("label_source") == "default":
        forecast_h = ask_forecast_horizon(model_horizon)

    print(f"  Modell: {C.CYAN}{_describe_model(model_path)}{C.RESET}")
    print(
        f"  {C.DIM}Vorausschau={forecast_h}M "
        f"(Modellhorizont={model_horizon}M, label={meta.get('label_source', '?')}){C.RESET}"
    )
    warn_model_coherence()
    try:
        result = score_frame(firm, model_path, events_df=events_df)
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.RED}Scoring fehlgeschlagen: {exc}{C.RESET}")
        pause()
        return

    print_score_table(
        result,
        company_name,
        focus_doc_id=focus_doc,
        meta=meta,
        forecast_horizon_months=forecast_h,
    )
    print()
    maybe_save_csv(
        result,
        default_csv or "scores_last.csv",
        model_horizon=model_horizon,
        forecast_horizon=forecast_h if meta.get("label_source") == "default" else None,
    )
    pause()


def input_from_dataset(index: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame | None] | None:
    print(f"  {C.BOLD}Aus Label-Set{C.RESET}")
    print(f"  {C.DIM}Name (z. B. Coca, IBM) oder CIK.{C.RESET}")
    print()
    query = ask("Firma / CIK")
    if not query:
        return None

    hits = search_companies(query, index) if not index.empty else pd.DataFrame()
    chosen_cik: int | None = None
    chosen_name = query

    if not hits.empty:
        print()
        print(f"  {C.BOLD}Treffer{C.RESET}")
        for i, row in enumerate(hits.itertuples(index=False), start=1):
            print(f"    {C.CYAN}{i:2d}{C.RESET}  CIK {int(row.cik):<10}  {row.name}")
        if len(hits) == 1:
            row = hits.iloc[0]
            chosen_cik = int(row["cik"])
            chosen_name = str(row["name"])
            print(f"\n  {C.DIM}Ein Treffer — wird automatisch gewählt.{C.RESET}")
        else:
            print(f"    {C.CYAN} 0{C.RESET}  Abbrechen")
            choice = ask("Nummer wählen", "1")
            if choice == "0":
                return None
            try:
                row = hits.iloc[int(choice) - 1]
                chosen_cik = int(row["cik"])
                chosen_name = str(row["name"])
            except (ValueError, IndexError):
                print(f"  {C.RED}Ungültige Auswahl.{C.RESET}")
                return None
    elif query.isdigit():
        chosen_cik = int(query)
        chosen_name = f"CIK {chosen_cik}"
    else:
        print(f"  {C.YELLOW}Keine Treffer.{C.RESET}")
        return None

    assert chosen_cik is not None
    print(f"\n  {C.BOLD}Lade{C.RESET} {chosen_name} (CIK {chosen_cik}) …")
    firm = load_firm_years_for_cik(chosen_cik)
    if firm.empty:
        print(f"  {C.RED}Keine Firm-Years gefunden.{C.RESET}")
        return None
    print(f"  {C.GREEN}{len(firm)} Firm-Years{C.RESET}  ({firm.attrs.get('source', '?')})")
    firm = attach_financials(firm)
    return firm, chosen_name, None


def input_from_file() -> tuple[pd.DataFrame, str, pd.DataFrame | None] | None:
    print(f"  {C.BOLD}Datei laden{C.RESET}")
    print(f"  {C.DIM}CSV / CSV.GZ / JSON (Firm-Year-Zeilen) oder .txt/.md (nur MD&A).{C.RESET}")
    print()
    raw = ask("Dateipfad")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        print(f"  {C.RED}Datei nicht gefunden: {path}{C.RESET}")
        return None

    suffix = path.suffix.lower()
    name = path.stem
    if suffix in {".txt", ".md", ".html"}:
        mda = path.read_text(encoding="utf-8", errors="ignore")
        print(f"  {C.GREEN}MD&A geladen ({len(mda):,} Zeichen).{C.RESET}")
        print(f"  {C.DIM}Finanzkennzahlen ergänzen (Fragebogen):{C.RESET}")
        row = _questionnaire_core(mda_default=mda)
        if row is None:
            return None
        firm = pd.DataFrame([row])
        return firm, str(row.get("name") or name), None

    try:
        if path.name.endswith(".csv.gz") or suffix == ".gz":
            firm = pd.read_csv(path)
        elif suffix == ".csv":
            firm = pd.read_csv(path)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            firm = pd.DataFrame(payload if isinstance(payload, list) else [payload])
        else:
            print(f"  {C.RED}Unbekanntes Format ({suffix}).{C.RESET}")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.RED}Lesen fehlgeschlagen: {exc}{C.RESET}")
        return None

    firm = _normalize_firm_frame(firm)
    if "total_assets" not in firm.columns and PANEL.exists() and "cik" in firm.columns:
        firm = attach_financials(firm, fetch_if_missing=False)
    title = str(firm["name"].iloc[0]) if "name" in firm.columns and len(firm) else name
    print(f"  {C.GREEN}{len(firm)} Zeile(n) geladen.{C.RESET}")
    return firm, title, None


def input_from_edgar() -> tuple[pd.DataFrame, str, pd.DataFrame | None] | None:
    print(f"  {C.BOLD}EDGAR live{C.RESET}")
    print(f"  {C.DIM}Lädt companyfacts (Finanzen) + Submissions (8-K Events) für einen CIK.{C.RESET}")
    print()
    cik_raw = ask("CIK")
    if not cik_raw:
        return None
    digits = "".join(ch for ch in cik_raw if ch.isdigit())
    if not digits:
        print(f"  {C.RED}Ungültige CIK.{C.RESET}")
        return None
    cik = int(digits)
    ua = _ensure_sec_ua()
    if ua is None:
        return None

    name = ask("Anzeigename (optional)", f"CIK {cik}")
    print(f"  {C.DIM}Lade Financials …{C.RESET}")
    try:
        panel = build_financials_panel([cik], user_agent=ua)
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.RED}Financials fehlgeschlagen: {exc}{C.RESET}")
        return None
    if panel.empty:
        print(f"  {C.RED}Keine companyfacts für CIK {cik}.{C.RESET}")
        return None
    print(f"  {C.GREEN}{len(panel)} Firm-Years aus EDGAR.{C.RESET}")

    events_df = None
    if ask("8-K-Events ebenfalls laden? (j/n)", "j").lower().startswith("j"):
        print(f"  {C.DIM}Lade Submissions (Cache: {SUBMISSIONS_CACHE.name}) …{C.RESET}")
        try:
            events_df = build_events_table(
                [cik],
                user_agent=ua,
                cache_dir=SUBMISSIONS_CACHE,
            )
            print(f"  {C.GREEN}{len(events_df)} 8-Ks.{C.RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {C.YELLOW}Events fehlgeschlagen: {exc}{C.RESET}")

    # MD&A aus lokalem Datensatz versuchen
    firm = panel.copy()
    firm["name"] = name
    local = load_firm_years_for_cik(cik) if LABELED.exists() or FIRM_YEARS.exists() else pd.DataFrame()
    if not local.empty and "mda" in local.columns:
        keep = ["cik", "fyear", "mda", "filing_date", "reporting_date", "doc_id", "label"]
        keep = [c for c in keep if c in local.columns]
        firm = firm.merge(local[keep], on=["cik", "fyear"], how="left", suffixes=("", "_loc"))
        for col in ("filing_date", "reporting_date", "doc_id", "mda", "label"):
            loc = f"{col}_loc"
            if loc in firm.columns:
                firm[col] = firm[col].fillna(firm[loc]) if col in firm.columns else firm[loc]
                firm = firm.drop(columns=[loc])
        n_txt = int(firm["mda"].fillna("").astype(str).str.len().gt(50).sum()) if "mda" in firm.columns else 0
        print(f"  {C.DIM}MD&A aus lokalem Datensatz für {n_txt}/{len(firm)} Jahre.{C.RESET}")

    if "mda" not in firm.columns or firm["mda"].fillna("").astype(str).str.len().max() < 50:
        print(f"  {C.YELLOW}Kein MD&A online in companyfacts — optional nachreichen.{C.RESET}")
        if ask("MD&A jetzt angeben? (j/n)", "n").lower().startswith("j"):
            mda = _read_mda_input()
            if mda:
                # auf gewähltes / neuestes Jahr legen
                firm = firm.sort_values("fyear").reset_index(drop=True)
                pick = pick_tenk(firm) if len(firm) > 1 else firm.iloc[0]
                if pick is None:
                    return None
                # nur das eine Jahr mit Text scoren
                y = int(pick["fyear"]) if pd.notna(pick.get("fyear")) else None
                if y is not None:
                    firm = firm.loc[firm["fyear"] == y].copy()
                else:
                    firm = pd.DataFrame([pick])
                firm["mda"] = mda

    if "reporting_date" not in firm.columns:
        firm["reporting_date"] = pd.NaT
    firm["reporting_date"] = pd.to_datetime(firm["reporting_date"], errors="coerce")
    miss_rep = firm["reporting_date"].isna() & firm["fyear"].notna()
    if miss_rep.any():
        firm.loc[miss_rep, "reporting_date"] = pd.to_datetime(
            firm.loc[miss_rep, "fyear"].astype(int).astype(str) + "-12-31",
            errors="coerce",
        )
    if "filing_date" not in firm.columns:
        firm["filing_date"] = pd.NaT
    firm["filing_date"] = pd.to_datetime(firm["filing_date"], errors="coerce")
    miss_fil = firm["filing_date"].isna() & firm["reporting_date"].notna()
    if miss_fil.any():
        firm.loc[miss_fil, "filing_date"] = firm.loc[miss_fil, "reporting_date"] + pd.Timedelta(days=90)

    # Unvollständige EDGAR-Jahre ohne Assets ausblenden (z. B. laufendes GJ)
    if "total_assets" in firm.columns:
        before = len(firm)
        firm = firm.loc[firm["total_assets"].notna()].copy()
        dropped = before - len(firm)
        if dropped:
            print(f"  {C.DIM}{dropped} Firm-Year(s) ohne Assets übersprungen.{C.RESET}")

    return firm, name, events_df


def _questionnaire_core(*, mda_default: str = "") -> dict | None:
    print(f"  {C.BOLD}Stammdaten{C.RESET}")
    cik_raw = ask("CIK (optional)", "")
    cik = int("".join(ch for ch in cik_raw if ch.isdigit()) or "0") or None
    name = ask("Unternehmen", "Manuelle Eingabe")
    fyear_raw = ask("Geschäftsjahr", str(pd.Timestamp.today().year - 1))
    try:
        fyear = int(fyear_raw)
    except ValueError:
        print(f"  {C.RED}Ungültiges Jahr.{C.RESET}")
        return None
    report = ask("Bilanzstichtag YYYY-MM-DD", f"{fyear}-12-31")
    filing = ask("10-K Filing-Datum YYYY-MM-DD", f"{fyear + 1}-03-01")

    print()
    print(f"  {C.BOLD}Finanzkennzahlen{C.RESET}  {C.DIM}(USD, leer = unbekannt → Imputation){C.RESET}")
    fields = [
        ("total_assets", "Total Assets"),
        ("total_liabilities", "Total Liabilities"),
        ("current_assets", "Current Assets"),
        ("current_liabilities", "Current Liabilities"),
        ("cash", "Cash & Equivalents"),
        ("long_term_debt", "Long-term Debt"),
        ("equity", "Stockholders' Equity"),
        ("net_income", "Net Income"),
        ("revenue", "Revenue"),
        ("ebit", "EBIT / Operating Income"),
        ("interest_expense", "Interest Expense"),
        ("retained_earnings", "Retained Earnings"),
        ("inventory", "Inventory"),
        ("receivables", "Receivables"),
    ]
    row: dict = {
        "cik": cik if cik is not None else 0,
        "name": name,
        "fyear": fyear,
        "reporting_date": report,
        "filing_date": filing,
        "label": 0,
        "doc_id": f"{cik or 0}_{report}",
    }
    for key, label in fields:
        val = _ask_float(label)
        if val is not None:
            row[key] = val

    print()
    if mda_default:
        row["mda"] = mda_default
        print(f"  {C.DIM}MD&A bereits gesetzt ({len(mda_default):,} Zeichen).{C.RESET}")
    else:
        row["mda"] = _read_mda_input()
    return row


def input_from_questionnaire() -> tuple[pd.DataFrame, str, pd.DataFrame | None] | None:
    print(f"  {C.BOLD}Fragebogen{C.RESET}")
    print(f"  {C.DIM}Ein 10-K manuell erfassen → {active_model_meta().get('default_horizon_months', 12)}"
          f"-Monats-PD.{C.RESET}")
    print()
    row = _questionnaire_core()
    if row is None:
        return None

    events_df = None
    cik = int(row.get("cik") or 0)
    if cik and ask("8-K-Events für diese CIK von EDGAR laden? (j/n)", "n").lower().startswith("j"):
        ua = _ensure_sec_ua()
        if ua:
            try:
                events_df = build_events_table([cik], user_agent=ua, cache_dir=SUBMISSIONS_CACHE)
                print(f"  {C.GREEN}{len(events_df)} 8-Ks geladen.{C.RESET}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {C.YELLOW}Events fehlgeschlagen: {exc}{C.RESET}")

    firm = pd.DataFrame([row])
    return firm, str(row.get("name") or "Manuell"), events_df


def flow_score_company(index: pd.DataFrame) -> None:
    clear()
    banner()
    meta = active_model_meta()
    horizon = FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
    if meta.get("label_source") == "default":
        print(f"  {C.BOLD}10-K scoren → Ausfall-PD{C.RESET}  "
              f"{C.DIM}(Vorausschau {horizon_label(horizon)}){C.RESET}")
    else:
        print(f"  {C.BOLD}10-K / Firm-Year scoren{C.RESET}")
    hr()
    print(f"  {C.CYAN}1{C.RESET}  Aus Label-Set (Firma suchen)")
    print(f"  {C.CYAN}2{C.RESET}  Datei laden (CSV / JSON / MD&A-Text)")
    print(f"  {C.CYAN}3{C.RESET}  EDGAR live (CIK → Finanzen + optional 8-K)")
    print(f"  {C.CYAN}4{C.RESET}  Fragebogen (alles manuell)")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    print()
    choice = ask("Eingabe", "1")

    packed = None
    if choice == "1":
        packed = input_from_dataset(index)
    elif choice == "2":
        packed = input_from_file()
    elif choice == "3":
        packed = input_from_edgar()
    elif choice == "4":
        packed = input_from_questionnaire()
    elif choice in {"0", "q"}:
        return
    else:
        print(f"  {C.YELLOW}Bitte 0–4 wählen.{C.RESET}")
        pause()
        return

    if packed is None:
        pause()
        return
    firm, name, events_df = packed
    cik = int(firm["cik"].dropna().iloc[0]) if "cik" in firm.columns and firm["cik"].notna().any() else 0
    run_scoring(
        firm,
        name,
        events_df=events_df,
        default_csv=f"scores_{cik or 'manual'}.csv",
    )

def show_settings_status() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Daten- & Modell-Status{C.RESET}")
    hr()
    rows = [
        ("AAER", AAER),
        ("Firm-Years Labels", FIRM_YEARS_LABELS),
        ("Firm-Years (groß)", FIRM_YEARS),
        ("Labeled Dataset", LABELED),
        ("Finanz-Panel", PANEL),
        ("8-K Events", EVENTS),
        ("Submissions-Cache", SUBMISSIONS_CACHE),
        ("Models-Dir", MODEL_DIR),
    ]
    for label, path in rows:
        print(f"  {label:<20} {_file_status(path)}")
        print(f"  {C.DIM}{'':20} {path.relative_to(ROOT)}{C.RESET}")

    print()
    print(f"  {C.BOLD}Modell-Bundles{C.RESET}")
    catalog = list_model_catalog()
    if not catalog:
        print(f"  {C.YELLOW}keine .joblib unter models/{C.RESET}")
    else:
        warn_model_coherence(catalog)
        for row in catalog:
            tag = "LEGACY" if row.get("legacy") else "ok"
            print(
                f"  · {row['name']:<32} {C.DIM}{tag}  "
                f"{row.get('component')}/{row.get('label_source')}"
                f"{' h'+str(row['horizon_months']) if row.get('horizon_months') else ''}  "
                f"trained={row.get('trained_at') or '—'}{C.RESET}"
            )
    print()
    mode = os.environ.get("SECPD_LLM_MODE", "mock")
    endpoint = os.environ.get("SECPD_LLM_ENDPOINT", "") or "—"
    model = os.environ.get("SECPD_LLM_MODEL", "internal-default")
    ua = os.environ.get("SECPD_SEC_UA", "") or "—"
    key_set = "gesetzt" if os.environ.get("SECPD_LLM_API_KEY") else "nicht gesetzt"
    print(f"  {C.BOLD}Umgebung{C.RESET}")
    print(f"    LLM-Modus     {C.CYAN}{mode}{C.RESET}")
    print(f"    LLM-Endpoint  {endpoint}")
    print(f"    LLM-Modell    {model}")
    print(f"    LLM-API-Key   {key_set}")
    print(f"    SEC-UA        {ua}")
    pause()


def settings_llm() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}LLM anbinden{C.RESET}")
    hr()
    key_set = (
        "gesetzt"
        if (os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        else "nicht gesetzt"
    )
    print(f"  Aktuell: Modus={C.CYAN}{os.environ.get('SECPD_LLM_MODE', 'mock')}{C.RESET}")
    print(f"           Endpoint={os.environ.get('SECPD_LLM_ENDPOINT', '') or '—'}")
    print(f"           Modell={os.environ.get('SECPD_LLM_MODEL', 'internal-default')}")
    print(f"           API-Key={key_set}")
    print(f"  {C.DIM}Secrets: {SECRETS_FILE.name} (lokal, nicht committen){C.RESET}")
    print()
    print(f"  {C.CYAN}1{C.RESET}  Mock (offline-Heuristik)")
    print(f"  {C.CYAN}2{C.RESET}  OpenAI / ChatGPT — schnell (Default: {DEFAULT_OPENAI_MODEL})")
    print(f"  {C.CYAN}3{C.RESET}  LM Studio (lokal) — 172.16.3.164:1234")
    print(f"  {C.CYAN}4{C.RESET}  Bank-Gateway (OpenAI-kompatibel)")
    print(f"  {C.CYAN}5{C.RESET}  Ping / Test-Call")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    print()
    choice = ask("Auswahl", "0")
    if choice == "1":
        os.environ["SECPD_LLM_MODE"] = "mock"
        save_secrets_env({"SECPD_LLM_MODE": "mock"})
        print(f"  {C.GREEN}LLM-Modus = mock{C.RESET}")
        pause()
        return
    if choice == "5":
        mode = os.environ.get("SECPD_LLM_MODE", "mock")
        if mode in {"openai", "chatgpt"}:
            ep = os.environ.get("SECPD_LLM_ENDPOINT") or DEFAULT_OPENAI_ENDPOINT
            model = os.environ.get("SECPD_LLM_MODEL") or DEFAULT_OPENAI_MODEL
        else:
            ep = os.environ.get("SECPD_LLM_ENDPOINT") or "http://172.16.3.164:1234"
            model = os.environ.get("SECPD_LLM_MODEL") or "auto"
        argv = [
            PY, str(ROOT / "scripts" / "ping_llm.py"),
            "--endpoint", ep, "--model", model, "--analyze",
            "--timeout", os.environ.get("SECPD_LLM_TIMEOUT", "120"),
        ]
        code = _run_script(argv, title="ping_llm.py")
        print(f"  {C.GREEN if code == 0 else C.YELLOW}"
              f"{'OK' if code == 0 else 'Fehlgeschlagen'}{C.RESET}")
        pause()
        return
    if choice == "2":
        os.environ["SECPD_LLM_MODE"] = "openai"
        os.environ["SECPD_LLM_ENDPOINT"] = DEFAULT_OPENAI_ENDPOINT
        cur_model = os.environ.get("SECPD_LLM_MODEL", "")
        default_model = cur_model if cur_model.startswith("gpt-") else DEFAULT_OPENAI_MODEL
        model = ask("OpenAI-Modell", default_model)
        os.environ["SECPD_LLM_MODEL"] = model or DEFAULT_OPENAI_MODEL
        os.environ["SECPD_LLM_JSON_MODE"] = "1"
        os.environ["SECPD_LLM_TIMEOUT"] = os.environ.get("SECPD_LLM_TIMEOUT") or "120"
        key = ask("OpenAI API-Key (sk-…, leer = behalten)", "")
        if key:
            os.environ["SECPD_LLM_API_KEY"] = key
        if not (os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print(f"  {C.RED}Kein API-Key — bitte eingeben.{C.RESET}")
            pause()
            return
        save_secrets_env(
            {
                "SECPD_LLM_MODE": "openai",
                "SECPD_LLM_ENDPOINT": DEFAULT_OPENAI_ENDPOINT,
                "SECPD_LLM_MODEL": os.environ["SECPD_LLM_MODEL"],
                "SECPD_LLM_API_KEY": os.environ.get("SECPD_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY", ""),
                "SECPD_LLM_JSON_MODE": "1",
                "SECPD_LLM_TIMEOUT": os.environ["SECPD_LLM_TIMEOUT"],
            }
        )
        print(f"  {C.GREEN}OpenAI gespeichert ({os.environ['SECPD_LLM_MODEL']}).{C.RESET}")
        pause()
        return
    if choice == "3":
        os.environ["SECPD_LLM_MODE"] = "lmstudio"
        endpoint = ask(
            "LM-Studio-Host/URL",
            "http://172.16.3.164:1234",
        )
        os.environ["SECPD_LLM_ENDPOINT"] = endpoint
        model = ask("Modell (auto = erstes Chat-Modell)", "auto")
        os.environ["SECPD_LLM_MODEL"] = model or "auto"
        os.environ["SECPD_LLM_API_KEY"] = os.environ.get("SECPD_LLM_API_KEY") or "lm-studio"
        os.environ["SECPD_LLM_JSON_MODE"] = "0"
        save_secrets_env(
            {
                "SECPD_LLM_MODE": "lmstudio",
                "SECPD_LLM_ENDPOINT": os.environ["SECPD_LLM_ENDPOINT"],
                "SECPD_LLM_MODEL": os.environ["SECPD_LLM_MODEL"],
                "SECPD_LLM_API_KEY": os.environ["SECPD_LLM_API_KEY"],
                "SECPD_LLM_JSON_MODE": "0",
            }
        )
        print(f"  {C.GREEN}LM Studio gesetzt.{C.RESET}")
        pause()
        return
    if choice != "4":
        return

    os.environ["SECPD_LLM_MODE"] = "bank"
    endpoint = ask("Endpoint-URL", os.environ.get("SECPD_LLM_ENDPOINT") or "")
    if endpoint:
        os.environ["SECPD_LLM_ENDPOINT"] = endpoint
    model = ask("Modellname", os.environ.get("SECPD_LLM_MODEL") or "internal-default")
    if model:
        os.environ["SECPD_LLM_MODEL"] = model
    key = ask("API-Key (leer = behalten)", "")
    if key:
        os.environ["SECPD_LLM_API_KEY"] = key
    payload = {
        "SECPD_LLM_MODE": "bank",
        "SECPD_LLM_ENDPOINT": os.environ.get("SECPD_LLM_ENDPOINT", ""),
        "SECPD_LLM_MODEL": os.environ.get("SECPD_LLM_MODEL", ""),
    }
    if os.environ.get("SECPD_LLM_API_KEY"):
        payload["SECPD_LLM_API_KEY"] = os.environ["SECPD_LLM_API_KEY"]
    save_secrets_env(payload)
    print(f"  {C.GREEN}Bank-Gateway gesetzt.{C.RESET}")
    pause()


def settings_sec_ua() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}SEC User-Agent{C.RESET}")
    hr()
    current = os.environ.get("SECPD_SEC_UA", "") or "—"
    print(f"  Aktuell: {current}")
    print(f"  {C.DIM}Format: „Firma name@firma.de“ (SEC-Pflicht für EDGAR).{C.RESET}")
    print()
    ua = ask("Neuer User-Agent (leer = abbrechen)", "")
    if not ua:
        return
    os.environ["SECPD_SEC_UA"] = ua
    print(f"  {C.GREEN}SECPD_SEC_UA gesetzt.{C.RESET}")
    pause()


def settings_convert_zenodo() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Convert Zenodo{C.RESET}")
    hr()
    fy = FIRM_YEARS_LABELS if FIRM_YEARS_LABELS.exists() else FIRM_YEARS
    print(f"  Firm-Years: {_file_status(fy)}")
    print(f"  AAER:       {_file_status(AAER)}")
    print(f"  Ziel:       {LABELED.relative_to(ROOT)}")
    print()
    if not fy.exists() or not AAER.exists():
        print(f"  {C.RED}Rohdaten fehlen — zuerst Zenodo laden (Menü 4 → Fetch Zenodo).{C.RESET}")
        pause()
        return
    if LABELED.exists() and not ask("Vorhandenes labeled überschreiben? (j/n)", "j").lower().startswith("j"):
        return
    _run_script(
        [
            PY, str(ROOT / "scripts" / "convert_zenodo.py"),
            "--firm-years", str(fy),
            "--aaer", str(AAER),
            "--out", str(LABELED),
        ],
        title="convert_zenodo.py",
    )
    pause()


def settings_fetch_zenodo() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch Zenodo{C.RESET}")
    hr()
    print(f"  Lädt aaer_mark5.csv + firm_years_labels.json (~683 MB).")
    print()
    if not ask("Jetzt herunterladen? (j/n)", "j").lower().startswith("j"):
        return
    _run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_zenodo.py"),
            "--files", "aaer_mark5.csv", "firm_years_labels.json",
            "--dest", str(ROOT / "data" / "raw"),
        ],
        title="fetch_zenodo.py",
    )
    pause()


def settings_fetch_edgar_financials() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch EDGAR Financials{C.RESET}")
    hr()
    print(f"  Dataset: {_file_status(LABELED)}")
    print(f"  Ziel:    {PANEL.relative_to(ROOT)}")
    print(f"  {C.DIM}~5 Min für ~500 CIKs, Internet nötig.{C.RESET}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt — zuerst Convert Zenodo.{C.RESET}")
        pause()
        return
    if _ensure_sec_ua() is None:
        pause()
        return
    if not ask("Abruf starten? (j/n)", "j").lower().startswith("j"):
        return
    _run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_edgar_financials.py"),
            "--dataset", str(LABELED),
            "--out", str(PANEL),
        ],
        title="fetch_edgar_financials.py",
    )
    pause()


def settings_fetch_edgar_events() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch EDGAR 8-K Events{C.RESET}")
    hr()
    print(f"  Dataset: {_file_status(LABELED)}")
    print(f"  Cache:   {_file_status(SUBMISSIONS_CACHE)}")
    print(f"  Ziel:    {EVENTS.relative_to(ROOT)}")
    print(f"  {C.DIM}~10–60 Min, resumefähig über den Cache.{C.RESET}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt — zuerst Convert Zenodo.{C.RESET}")
        pause()
        return
    if _ensure_sec_ua() is None:
        pause()
        return
    if not ask("Abruf starten? (j/n)", "j").lower().startswith("j"):
        return
    _run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_edgar_events.py"),
            "--dataset", str(LABELED),
            "--out", str(EVENTS),
            "--cache-dir", str(SUBMISSIONS_CACHE),
        ],
        title="fetch_edgar_events.py",
    )
    pause()


def settings_train() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Modell trainieren{C.RESET}")
    hr()
    print(f"  Data:       {_file_status(LABELED)}")
    print(f"  Financials: {_file_status(PANEL)}")
    print(f"  Events:     {_file_status(EVENTS)}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt.{C.RESET}")
        pause()
        return
    if not PANEL.exists():
        print(f"  {C.YELLOW}Finanz-Panel fehlt — Training ohne --financials ist schwach.{C.RESET}")
        if not ask("Trotzdem fortfahren? (j/n)", "n").lower().startswith("j"):
            return

    print(f"  {C.BOLD}Label{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}  fraud   — AAER/Misconduct")
    print(f"  {C.CYAN}2{C.RESET}  default — Ausfall-PD aus 8-K Insolvenz (Horizont wählbar)")
    label_choice = ask("Label-Quelle", "2")
    label_source = "default" if label_choice == "2" else "fraud"
    if label_source == "default" and not EVENTS.exists():
        print(f"  {C.RED}default braucht Events — zuerst Fetch EDGAR 8-K.{C.RESET}")
        pause()
        return

    print()
    print(f"  {C.BOLD}Modus{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}  financial — nur Kennzahlen (+ Events)")
    print(f"  {C.CYAN}2{C.RESET}  combined  — Finanzen + Text + Events")
    print(f"  {C.CYAN}3{C.RESET}  ensemble  — Logit-Ensemble")
    mode_map = {"1": "financial", "2": "combined", "3": "ensemble"}
    mode = mode_map.get(ask("Modus", "2"), "combined")

    llm = os.environ.get("SECPD_LLM_MODE", "mock")
    llm_refresh = False
    if mode in {"combined", "ensemble"}:
        print()
        print(f"  {C.BOLD}LLM für Textfeatures{C.RESET}")
        print(f"  {C.CYAN}1{C.RESET}  mock")
        print(f"  {C.CYAN}2{C.RESET}  openai / ChatGPT ({DEFAULT_OPENAI_MODEL})")
        print(f"  {C.CYAN}3{C.RESET}  lmstudio")
        print(f"  {C.CYAN}4{C.RESET}  bank")
        pick = ask("Auswahl", "2" if llm in {"openai", "chatgpt"} else "1")
        llm = {"1": "mock", "2": "openai", "3": "lmstudio", "4": "bank"}.get(pick, "openai")
        if llm == "openai" and not (
            os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        ):
            print(f"  {C.YELLOW}Kein OpenAI-Key — bitte unter Einstellungen → LLM setzen.{C.RESET}")
            key = ask("OpenAI API-Key jetzt eingeben (oder leer = abbrechen)", "")
            if not key:
                pause()
                return
            os.environ["SECPD_LLM_MODE"] = "openai"
            os.environ["SECPD_LLM_API_KEY"] = key
            os.environ["SECPD_LLM_ENDPOINT"] = DEFAULT_OPENAI_ENDPOINT
            os.environ["SECPD_LLM_MODEL"] = DEFAULT_OPENAI_MODEL
            save_secrets_env(
                {
                    "SECPD_LLM_MODE": "openai",
                    "SECPD_LLM_API_KEY": key,
                    "SECPD_LLM_ENDPOINT": DEFAULT_OPENAI_ENDPOINT,
                    "SECPD_LLM_MODEL": DEFAULT_OPENAI_MODEL,
                }
            )
        print()
        print(f"  {C.BOLD}Textfeatures{C.RESET}")
        print(f"  {C.CYAN}1{C.RESET}  Cache nutzen (schnell, nur Misses → LLM)")
        print(f"  {C.CYAN}2{C.RESET}  Neu bewerten (--llm-refresh, überschreibt Cache)")
        cache_pick = ask("Auswahl", "1")
        llm_refresh = cache_pick == "2"

    calibrate = ask("Kalibrieren? (j/n)", "j").lower().startswith("j")

    argv = [
        PY, str(ROOT / "train.py"),
        "--data", str(LABELED),
        "--mode", mode,
        "--label-source", label_source,
        "--out", str(ROOT / "models"),
    ]
    if PANEL.exists():
        argv += ["--financials", str(PANEL)]
    if EVENTS.exists() and ask("Event-Features (--events) nutzen? (j/n)", "j").lower().startswith("j"):
        argv += ["--events", str(EVENTS)]
    if label_source == "default":
        if "--events" not in argv:
            argv += ["--events", str(EVENTS)]
        print()
        print(f"  {C.BOLD}Trainings-Horizont (Label){C.RESET}")
        print(f"  {C.DIM}12=1J, 60=5J, 120=10J — echte Labels, nicht nur Termstruktur.{C.RESET}")
        horizon = ask("Monate", "12")
        try:
            h = max(1, int(horizon))
        except ValueError:
            h = 12
        argv += ["--default-horizon", str(h)]
        argv += ["--min-fyear", str(MIN_FYEAR_WITH_FINANCIALS)]
        if ask(
            f"Nur Zeilen mit Finanzdaten (total_assets)? (j/n)",
            "j",
        ).lower().startswith("j"):
            argv.append("--require-financials")
        print(
            f"  {C.DIM}Policy: Legacy-Regime vor 2004-08-23 ignoriert; "
            f"min-fyear={MIN_FYEAR_WITH_FINANCIALS}.{C.RESET}"
        )
    if mode in {"combined", "ensemble"}:
        argv += ["--llm", llm]
        if llm_refresh:
            argv.append("--llm-refresh")
    if calibrate:
        argv.append("--calibrate")

    print()
    if not ask("Training starten? (j/n)", "j").lower().startswith("j"):
        return
    _run_script(argv, title="train.py")
    global _ACTIVE_MODEL_PATH, _MODEL_META_CACHE
    _ACTIVE_MODEL_PATH = None
    _MODEL_META_CACHE = None
    active_model_meta(refresh=True)
    pause()


def settings_forecast_horizon() -> None:
    global FORECAST_HORIZON_MONTHS
    clear()
    banner()
    print(f"  {C.BOLD}Standard-Vorausschauhorizont{C.RESET}")
    hr()
    meta = active_model_meta()
    model_h = int(meta.get("default_horizon_months") or 12)
    cur = FORECAST_HORIZON_MONTHS or model_h
    print(f"  Modellhorizont:     {model_h} M")
    print(f"  Session-Vorausschau: {cur} M")
    print()
    FORECAST_HORIZON_MONTHS = ask_forecast_horizon(model_h)
    print(f"  {C.GREEN}Gesetzt: {FORECAST_HORIZON_MONTHS} Monate "
          f"({horizon_label(FORECAST_HORIZON_MONTHS)}).{C.RESET}")
    pause()


def settings_risk_bands() -> None:
    """Schwellen relativ zur Sample-Basisrate im jeweiligen Horizont."""
    global RISK_BAND_MID_MULT, RISK_BAND_HIGH_MULT, DEFAULT_BASE_RATE_12M
    clear()
    banner()
    print(f"  {C.BOLD}Risiko-Schwellen (Default-PD){C.RESET}")
    hr()
    meta = active_model_meta()
    metrics = meta.get("metrics") or {}
    model_br = metrics.get("base_rate")
    print(f"  {C.DIM}Bänder = Multiplikatoren × Basisrate im Horizont.{C.RESET}")
    print(f"  {C.DIM}unter Basisrate  <  mid × Basis{C.RESET}")
    print(f"  {C.DIM}um Basisrate     mid … high × Basis{C.RESET}")
    print(f"  {C.DIM}über Basisrate   ≥ high × Basis{C.RESET}")
    print()
    print(f"  Aktuell mid={RISK_BAND_MID_MULT:.2f}  high={RISK_BAND_HIGH_MULT:.2f}")
    print(f"  Fallback-Basisrate 12M: {100 * DEFAULT_BASE_RATE_12M:.2f} %")
    if model_br is not None:
        print(f"  Modell-Testset-Basisrate: {100 * float(model_br):.2f} % "
              f"(wird bevorzugt, falls vorhanden)")
    print()
    print(f"  {C.CYAN}1{C.RESET}  Schwellen anpassen")
    print(f"  {C.CYAN}2{C.RESET}  Defaults wiederherstellen (0.85 / 2.50)")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    choice = ask("Auswahl", "0")
    if choice == "2":
        RISK_BAND_MID_MULT, RISK_BAND_HIGH_MULT = 0.85, 2.5
        DEFAULT_BASE_RATE_12M = 0.012
        print(f"  {C.GREEN}Defaults gesetzt.{C.RESET}")
        pause()
        return
    if choice != "1":
        return

    mid_raw = ask("mid-Multiplikator (um Basisrate ab)", f"{RISK_BAND_MID_MULT:.2f}")
    high_raw = ask("high-Multiplikator (über Basisrate ab)", f"{RISK_BAND_HIGH_MULT:.2f}")
    br_raw = ask("Fallback-Basisrate 12M (z. B. 0.012)", f"{DEFAULT_BASE_RATE_12M:.4f}")
    try:
        mid = float(mid_raw.replace(",", "."))
        high = float(high_raw.replace(",", "."))
        br = float(br_raw.replace(",", "."))
        if not (0 < mid < high) or not (0 < br < 1):
            raise ValueError
    except ValueError:
        print(f"  {C.RED}Ungültig — nichts geändert "
              f"(braucht 0 < mid < high und 0 < Basisrate < 1).{C.RESET}")
        pause()
        return
    RISK_BAND_MID_MULT, RISK_BAND_HIGH_MULT = mid, high
    DEFAULT_BASE_RATE_12M = br
    print()
    print(f"  {C.GREEN}Gesetzt:{C.RESET} mid={mid:.2f}  high={high:.2f}  "
          f"Fallback-BR={100 * br:.2f} %")
    # Beispiel für aktuellen Horizont
    h = FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
    base_h = scale_pd(br, from_months=12, to_months=h)
    print(f"  {C.DIM}Beispiel {horizon_label(h)}: Basis≈{100 * base_h:.2f}% → "
          f"„um“ ab {100 * mid * base_h:.2f}%, "
          f"„über“ ab {100 * high * base_h:.2f}%{C.RESET}")
    pause()


def settings_menu() -> None:
    while True:
        clear()
        banner()
        print(f"  {C.BOLD}Einstellungen{C.RESET}")
        hr()
        fh = FORECAST_HORIZON_MONTHS
        fh_s = f"{fh} M" if fh else "Modelldefault"
        print(f"  {C.CYAN}1{C.RESET}  Status / Voraussetzungen")
        print(f"  {C.CYAN}2{C.RESET}  LLM anbinden")
        print(f"  {C.CYAN}3{C.RESET}  SEC User-Agent")
        print(f"  {C.CYAN}4{C.RESET}  Vorausschauhorizont  {C.DIM}({fh_s}){C.RESET}")
        print(
            f"  {C.CYAN}5{C.RESET}  Risiko-Schwellen  "
            f"{C.DIM}(mid={RISK_BAND_MID_MULT:.2f}, high={RISK_BAND_HIGH_MULT:.2f}){C.RESET}"
        )
        print(f"  {C.CYAN}6{C.RESET}  Fetch Zenodo")
        print(f"  {C.CYAN}7{C.RESET}  Convert Zenodo")
        print(f"  {C.CYAN}8{C.RESET}  Fetch EDGAR Financials")
        print(f"  {C.CYAN}9{C.RESET}  Fetch EDGAR 8-K Events")
        print(f"  {C.CYAN}10{C.RESET} Modell trainieren")
        print(f"  {C.CYAN}0{C.RESET}  Zurück")
        print()
        choice = ask("Auswahl", "0")

        if choice == "1":
            show_settings_status()
        elif choice == "2":
            settings_llm()
        elif choice == "3":
            settings_sec_ua()
        elif choice == "4":
            settings_forecast_horizon()
        elif choice == "5":
            settings_risk_bands()
        elif choice == "6":
            settings_fetch_zenodo()
        elif choice == "7":
            settings_convert_zenodo()
        elif choice == "8":
            settings_fetch_edgar_financials()
        elif choice == "9":
            settings_fetch_edgar_events()
        elif choice == "10":
            settings_train()
        elif choice in {"0", "q", "b", "back"}:
            return
        else:
            print(f"  {C.YELLOW}Bitte 0–10 wählen.{C.RESET}")
            pause()


def show_help() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Hilfe{C.RESET}")
    hr()
    print(
        """
  1) 10-K scoren
     Strukturierte Eingabe:
       1  Label-Set (Firma/CIK suchen)
       2  Datei laden (CSV/JSON oder MD&A-Text)
       3  EDGAR live (Finanzen + optional 8-Ks)
       4  Fragebogen (Kennzahlen + MD&A manuell)
     → 10-K wählen → Prognosehorizont in Monaten
       (12 / 24 / 60 / 120 oder frei, z. B. 18)
     → PD + Termstruktur-Vorausschau
     CSV-Export optional (inkl. pd_12m … pd_120m)

  2) Modellgüte
     · ROC-AUC / PR-AUC / Brier vom letzten Training

  4) Einstellungen
     · Vorausschauhorizont, LLM, SEC-UA, Zenodo, EDGAR, Training
     · Training mit eigenem --default-horizon (echte Labels)

  Hinweise
     · Modell-PD = trainierter Horizont (meist 12 M)
     · Andere Horizonte: konstante Hazard-Termstruktur
     · Echte 5J/10J-PD: neu trainieren mit Horizont 60/120
"""
    )
    pause()


def main_menu() -> None:
    while True:
        missing = _require_files()
        index = load_company_index() if LABELED.exists() else pd.DataFrame(columns=["cik", "name"])
        meta = active_model_meta()

        clear()
        banner()
        if missing:
            print(f"  {C.YELLOW}Fehlende Dateien:{C.RESET}")
            for m in missing:
                print(f"    · {m}")
            print(f"  {C.DIM}Über Einstellungen (4) nachladen / trainieren.{C.RESET}")
            print()

        n_co = index["cik"].nunique() if not index.empty else 0
        label = meta.get("label_source") or "?"
        active = active_model_path()
        active_s = active.name if active else "fehlt"
        print(
            f"  {C.DIM}Label-Set: {n_co} Unternehmen · aktiv: {active_s} · "
            f"target={label} · LLM={os.environ.get('SECPD_LLM_MODE', 'mock')}{C.RESET}"
        )
        warn_model_coherence()
        print()
        if label == "default":
            h = FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
            print(f"  {C.CYAN}1{C.RESET}  10-K scoren (PD, Horizont wählbar)")
            print(f"     {C.DIM}Session-Vorausschau: {horizon_label(h)}{C.RESET}")
        else:
            print(f"  {C.CYAN}1{C.RESET}  10-K scoren")
        print(f"  {C.CYAN}2{C.RESET}  Modellgüte anzeigen")
        print(f"  {C.CYAN}3{C.RESET}  Hilfe")
        print(f"  {C.CYAN}4{C.RESET}  Einstellungen")
        print(f"  {C.CYAN}0{C.RESET}  Beenden")
        print()
        choice = ask("Auswahl", "1")

        if choice == "1":
            if not list_model_catalog():
                print(f"  {C.RED}Kein trainiertes Modell — Einstellungen → Trainieren.{C.RESET}")
                pause()
                continue
            flow_score_company(index)
        elif choice == "2":
            show_model_quality()
        elif choice == "3":
            show_help()
        elif choice == "4":
            settings_menu()
            active_model_meta(refresh=True)
        elif choice in {"0", "q", "quit", "exit"}:
            print(f"\n  {C.DIM}Tschüss.{C.RESET}\n")
            return
        else:
            print(f"  {C.YELLOW}Bitte 0–4 wählen.{C.RESET}")
            pause()


if __name__ == "__main__":
    load_secrets_env()
    main_menu()
