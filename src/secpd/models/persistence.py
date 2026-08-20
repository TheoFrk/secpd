"""Modell-Persistierung via joblib — mit Metadaten und Versions-Checks.

Wichtig für den Home→Bank-Transfer: joblib-Artefakte sind **nicht** über
sklearn-Versionen hinweg garantiert kompatibel. Deshalb werden Versionen im
Bundle gestempelt und beim Laden gegen die installierte Umgebung geprüft
(Warnung bei Abweichung). Die gepinnten ``requirements.txt`` halten beide
Umgebungen ohnehin identisch — der Check ist das Sicherheitsnetz.
"""
from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

BUNDLE_KIND_SINGLE = "single"
BUNDLE_KIND_ENSEMBLE = "ensemble"

_LEGACY_NAMES = {
    "financial_model.joblib",
    "combined_model.joblib",
    "ensemble_model.joblib",
}


def bundle_filename(
    component: str,
    *,
    label_source: str = "fraud",
    horizon_months: int | None = None,
    rating_target: str | None = None,
) -> str:
    """Kanonischer Artefakt-Name inkl. Label (und Horizont bei default).

    Beispiele::

        financial_fraud.joblib
        combined_default_h12.joblib
        ensemble_default_h60.joblib
        combined_rating_speculative.joblib
        financial_rating_downgrade_h12.joblib
        combined_rating_ordinal.joblib
    """
    comp = str(component).strip().lower().replace("-", "_")
    if comp in {"fin", "financial_baseline", "financial_model"}:
        comp = "financial"
    elif comp in {"comb", "combined_model"}:
        comp = "combined"
    elif comp in {"ens", "ensemble_model"}:
        comp = "ensemble"
    src = str(label_source).strip().lower() or "fraud"
    if src == "default":
        h = int(horizon_months or 12)
        return f"{comp}_default_h{h}.joblib"
    if src == "rating":
        tgt = str(rating_target or "ordinal").strip().lower()
        if tgt in {"hy", "ig_hy"}:
            tgt = "speculative"
        if tgt == "downgrade":
            h = int(horizon_months or 12)
            return f"{comp}_rating_downgrade_h{h}.joblib"
        if tgt == "speculative":
            return f"{comp}_rating_speculative.joblib"
        return f"{comp}_rating_ordinal.joblib"
    return f"{comp}_{src}.joblib"


def runtime_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Versionen/Zeitstempel der Trainingsumgebung (Reproduzierbarkeit)."""
    import numpy
    import pandas
    import sklearn

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta: dict[str, Any] = {
        "created_at_utc": now,
        "trained_at_utc": now,  # Alias — Forensik/UX
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
    }
    if extra:
        meta.update(extra)
        if "created_at_utc" in meta and "trained_at_utc" not in (extra or {}):
            meta["trained_at_utc"] = meta["created_at_utc"]
    return meta


@dataclass
class ModelBundle:
    """Ein trainiertes Modell samt Feature-Vertrag und Metadaten."""

    pipeline: Any
    feature_cols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_single(bundle: ModelBundle, path: Path | str) -> Path:
    """Speichert ein Einzelmodell (Option A bzw. Financial-Baseline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": BUNDLE_KIND_SINGLE,
            "pipeline": bundle.pipeline,
            "feature_cols": list(bundle.feature_cols),
            "metadata": dict(bundle.metadata),
        },
        path,
    )
    logger.info("Modell gespeichert: %s", path)
    return path


def save_ensemble(
    financial: ModelBundle,
    text: ModelBundle,
    *,
    weights: dict[str, float],
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Speichert das Zwei-Komponenten-Ensemble (Option B) als ein Artefakt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": BUNDLE_KIND_ENSEMBLE,
            "financial": {
                "pipeline": financial.pipeline,
                "feature_cols": list(financial.feature_cols),
            },
            "text": {
                "pipeline": text.pipeline,
                "feature_cols": list(text.feature_cols),
            },
            "weights": dict(weights),
            "metadata": dict(metadata or {}),
        },
        path,
    )
    logger.info("Ensemble gespeichert: %s", path)
    return path


def load_any(path: Path | str) -> dict[str, Any]:
    """Lädt ein Bundle und warnt bei sklearn-Versionsdrift."""
    import sklearn

    payload: dict[str, Any] = joblib.load(Path(path))
    meta = bundle_metadata(payload)
    trained_with = meta.get("sklearn")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "sklearn-Versionsdrift: Bundle mit %s trainiert, installiert ist %s — "
            "requirements.txt beider Umgebungen abgleichen!",
            trained_with,
            sklearn.__version__,
        )
    return payload


def bundle_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Metadaten aus Single- oder Ensemble-Bundle lesen."""
    md = payload.get("metadata")
    if isinstance(md, dict) and md:
        return md
    fin = payload.get("financial")
    if isinstance(fin, dict):
        inner = fin.get("metadata")
        if isinstance(inner, dict):
            return inner
    return {}


def discover_model_paths(models_dir: Path | str) -> list[Path]:
    """Alle .joblib-Bundles unter models/ (inkl. Legacy-Namen)."""
    root = Path(models_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.joblib") if p.is_file())


_NAME_RE = re.compile(
    r"^(?P<comp>financial|combined|ensemble)"
    r"_(?P<label>fraud|default|rating(?:_speculative|_downgrade|_ordinal)?)"
    r"(?:_h(?P<horizon>\d+))?"
    r"\.joblib$",
    re.IGNORECASE,
)


def same_training_run(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    slack_s: float = 120.0,
) -> bool:
    """True, wenn zwei Catalog-Zeilen zum selben Trainingslauf gehören.

    Reihenfolge: ``train_run_id`` → ISO-``trained_at`` (≤ slack) → Datei-mtime.
    Fehlen alle Stempel, gilt das Paar *nicht* als derselbe Lauf (lieber
    warnen) — außer die mtimes liegen eng beieinander.
    """
    ra = a.get("train_run_id") or (a.get("metadata") or {}).get("train_run_id")
    rb = b.get("train_run_id") or (b.get("metadata") or {}).get("train_run_id")
    if ra and rb:
        return str(ra) == str(rb)

    ta, tb = a.get("trained_at") or "", b.get("trained_at") or ""
    if ta and tb:
        try:
            from datetime import datetime

            def _parse(s: str) -> datetime:
                return datetime.fromisoformat(str(s).replace("Z", "+00:00"))

            return abs((_parse(str(ta)) - _parse(str(tb))).total_seconds()) <= slack_s
        except (TypeError, ValueError):
            if ta == tb:
                return True

    ma, mb = a.get("mtime"), b.get("mtime")
    try:
        if ma is not None and mb is not None:
            return abs(float(ma) - float(mb)) <= slack_s
    except (TypeError, ValueError):
        return False
    return False


def parse_bundle_name(path: Path | str) -> dict[str, Any]:
    """Komponente/Label/Horizont aus Dateiname (falls kanonisch)."""
    name = Path(path).name
    m = _NAME_RE.match(name)
    if not m:
        legacy = name.lower() in _LEGACY_NAMES
        return {
            "component": None,
            "label_source": None,
            "rating_target": None,
            "horizon_months": None,
            "legacy": legacy,
            "filename": name,
        }
    raw_label = m.group("label").lower()
    rating_target = None
    if raw_label.startswith("rating"):
        label_source = "rating"
        if raw_label == "rating_downgrade":
            rating_target = "downgrade"
        elif raw_label == "rating_speculative":
            rating_target = "speculative"
        elif raw_label == "rating_ordinal":
            rating_target = "ordinal"
        else:
            rating_target = "ordinal"
    else:
        label_source = raw_label
    return {
        "component": m.group("comp").lower(),
        "label_source": label_source,
        "rating_target": rating_target,
        "horizon_months": int(m.group("horizon")) if m.group("horizon") else None,
        "legacy": False,
        "filename": name,
    }
