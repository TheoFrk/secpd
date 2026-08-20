"""Modell-Katalog: Bundles unter ``models/`` entdecken, wählen, Kohärenz prüfen.

Fraud-Bundles gelten grundsätzlich als experimentell (zu wenige Positive,
negativer Brier-Skill) und werden nur gewählt, wenn explizit
``label_source='fraud'`` angefragt wird.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from secpd.cli.paths import MODEL_DIR, NATIVE_DEFAULT_HORIZONS
from secpd.cli.ui import C
from secpd.models.persistence import (
    bundle_metadata,
    discover_model_paths,
    load_any,
    parse_bundle_name,
    same_training_run,
)

_MODEL_META_CACHE: dict | None = None
_ACTIVE_MODEL_PATH: Path | None = None


def is_experimental(row: dict[str, Any]) -> bool:
    """Fraud ist immer experimentell; sonst Meta-Feld ``status``."""
    md = row.get("metadata") or {}
    if str(row.get("label_source") or md.get("label_source") or "") == "fraud":
        return True
    return str(md.get("status") or "") == "experimental"


def invalidate_model_cache() -> None:
    """Nach einem Trainingslauf: aktives Modell neu auflösen."""
    global _MODEL_META_CACHE, _ACTIVE_MODEL_PATH
    _MODEL_META_CACHE = None
    _ACTIVE_MODEL_PATH = None


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
        mtime = path.stat().st_mtime if path.exists() else None
        row = {
            "path": path,
            "name": path.name,
            "kind": kind,
            "metadata": md,
            "component": component,
            "label_source": label,
            "rating_target": md.get("rating_target") or parsed.get("rating_target"),
            "horizon_months": int(horizon) if horizon is not None else None,
            "legacy": bool(parsed.get("legacy")),
            "trained_at": md.get("trained_at_utc") or md.get("created_at_utc"),
            "train_run_id": md.get("train_run_id"),
            "mtime": mtime,
            "load_error": md.get("_load_error"),
            "data": md.get("data"),
            "status": md.get("status"),
        }
        row["experimental"] = is_experimental(row)
        out.append(row)
    return out


def warn_model_coherence(catalog: list[dict] | None = None) -> None:
    """Warnt, wenn Financial/Combined nicht aus demselben Trainingslauf stammen."""
    cat = catalog if catalog is not None else list_model_catalog()
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

    def _stamp(row: dict) -> str:
        if row.get("load_error"):
            return "Lade-Fehler"
        rid = row.get("train_run_id")
        ts = row.get("trained_at")
        if rid and ts:
            return f"run={rid} @ {ts}"
        if rid:
            return f"run={rid}"
        if ts:
            return str(ts)
        mtime = row.get("mtime")
        if mtime:
            return datetime.fromtimestamp(float(mtime), tz=timezone.utc).isoformat(timespec="seconds")
        return "ohne Stempel"

    for c in combs:
        peers = [
            f
            for f in fins
            if f.get("label_source") == c.get("label_source")
            and f.get("horizon_months") == c.get("horizon_months")
            and f.get("rating_target") == c.get("rating_target")
        ]
        if not peers:
            print(
                f"  {C.YELLOW}Hinweis: {c['name']} ohne passendes financial-Pendant "
                f"(label={c.get('label_source')}, h={c.get('horizon_months')}) — "
                f"Gütevergleiche über Bundles sind unsicher.{C.RESET}"
            )
            continue
        for f in peers:
            if c.get("load_error") or f.get("load_error"):
                print(
                    f"  {C.YELLOW}Warnung: {c['name']} / {f['name']} nicht lesbar "
                    f"({c.get('load_error') or f.get('load_error')}).{C.RESET}"
                )
                continue
            if not same_training_run(c, f):
                print(
                    f"  {C.YELLOW}Warnung: {c['name']} und {f['name']} stammen "
                    f"nicht aus demselben Lauf "
                    f"({_stamp(c)} vs {_stamp(f)}).{C.RESET}"
                )


def select_model_path(
    *,
    prefer: str = "combined",
    label_source: str | None = None,
    horizon_months: int | None = None,
    rating_target: str | None = None,
    has_text: bool = True,
    require: bool = False,
) -> Path | None:
    """Wählt das beste passende Bundle; Legacy nur als Fallback.

    Fraud-Bundles werden übersprungen, solange nicht explizit
    ``label_source='fraud'`` gesetzt ist.
    """
    cat = list_model_catalog()
    if not cat:
        return None
    want_label = label_source
    if want_label != "fraud":
        cat = [r for r in cat if not r.get("experimental")]
    if require:
        if label_source:
            cat = [r for r in cat if r.get("label_source") == label_source]
        if rating_target:
            cat = [r for r in cat if r.get("rating_target") == rating_target]
        if horizon_months is not None and label_source == "default":
            cat = [r for r in cat if r.get("horizon_months") == horizon_months]
        if not cat:
            return None
    if not cat:
        return None

    want_h = horizon_months
    want_rt = rating_target
    prefer_comp = "combined" if has_text and prefer == "combined" else "financial"
    if not has_text:
        prefer_comp = "financial"

    def score(row: dict) -> tuple:
        comp = str(row.get("component") or "")
        lab = row.get("label_source")
        hor = row.get("horizon_months")
        rtg = row.get("rating_target")
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
        elif lab == "rating" and rtg == "ordinal":
            s += 20
        elif lab == "default":
            s += 10
        if want_rt and rtg == want_rt:
            s += 40
        if want_h is not None and hor == want_h:
            s += 30
        elif want_h is None and hor == 12:
            s += 5
        s -= legacy * 80
        ts = str(row.get("trained_at") or "")
        return (s, ts)

    ranked = sorted(cat, key=score, reverse=True)
    best = ranked[0]
    canon = [r for r in cat if not r.get("legacy")]
    if best.get("legacy") and canon:
        best = sorted(canon, key=score, reverse=True)[0]
    return Path(best["path"])


def select_default_pd_path(
    *,
    has_text: bool,
    want_horizon: int,
) -> tuple[Path | None, dict[str, Any]]:
    """Wählt ein Default-PD-Bundle; bevorzugt nativen Horizont 12/24/36."""
    prefer = "combined" if has_text else "financial"
    ordered: list[int] = []
    if want_horizon in NATIVE_DEFAULT_HORIZONS:
        ordered.append(want_horizon)
    for h in sorted(NATIVE_DEFAULT_HORIZONS, key=lambda x: (abs(x - want_horizon), x)):
        if h not in ordered:
            ordered.append(h)
    for h in ordered:
        path = select_model_path(
            prefer=prefer,
            label_source="default",
            horizon_months=h,
            has_text=has_text,
            require=True,
        )
        if path is None:
            continue
        try:
            md = dict(load_any(path).get("metadata") or {})
        except Exception:  # noqa: BLE001
            md = {}
        md.setdefault("default_horizon_months", h)
        return path, md
    return None, {}


def active_model_path(*, refresh: bool = False) -> Path | None:
    global _ACTIVE_MODEL_PATH
    if _ACTIVE_MODEL_PATH is not None and not refresh and _ACTIVE_MODEL_PATH.exists():
        return _ACTIVE_MODEL_PATH
    path = select_model_path(
        prefer="combined",
        label_source="rating",
        rating_target="ordinal",
        require=True,
    )
    if path is None:
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


def describe_model(path: Path) -> str:
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
    exp = "  [experimentell]" if lab == "fraud" or md.get("status") == "experimental" else ""
    return f"{path.name}  [{mode}/{lab}{h_s}]{exp}  trained={ts}  data={data}"
