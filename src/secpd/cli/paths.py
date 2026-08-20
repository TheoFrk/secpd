"""Projektpfade und Prozess-Konstanten für das interaktive CLI."""
from __future__ import annotations

import sys
from pathlib import Path

from secpd.config import PROJECT_ROOT

ROOT = PROJECT_ROOT
PY = sys.executable
SECRETS_FILE = ROOT / ".secpd.env"

LABELED = ROOT / "data" / "processed" / "zenodo_labeled.csv.gz"
LABELED_FULL = ROOT / "data" / "processed" / "zenodo_full.csv.gz"
PANEL = ROOT / "data" / "raw" / "financials_panel.csv"
PANEL_FULL = ROOT / "data" / "raw" / "financials_panel_full.csv"
EVENTS = ROOT / "data" / "raw" / "edgar_8k_events.csv"
EVENTS_FULL = ROOT / "data" / "raw" / "edgar_8k_events_full.csv"
RATINGS = ROOT / "data" / "raw" / "ratings_panel.csv"
FIRM_YEARS = ROOT / "data" / "raw" / "firm_years.json"
FIRM_YEARS_LABELS = ROOT / "data" / "raw" / "firm_years_labels.json"
AAER = ROOT / "data" / "raw" / "aaer_mark5.csv"
SUBMISSIONS_CACHE = ROOT / "data" / "raw" / "edgar_submissions"
MODEL_DIR = ROOT / "models"
SCORES_DIR = ROOT / "data" / "processed"

#: Native Default-PD-Horizonte (eigene Bundles, keine Termstruktur-Skalierung).
NATIVE_DEFAULT_HORIZONS: tuple[int, ...] = (12, 24, 36)

#: Anzeige-Pfade für Modellgüte (voller Universums-Stand, nicht der 533er-Freeze).
FREEZE_REPORT = ROOT / "benchmarks" / "default_h12_full" / "REPORT.md"
ROLLING_REPORT = ROOT / "benchmarks" / "rolling_full_h12_bankruptcy" / "REPORT.md"
