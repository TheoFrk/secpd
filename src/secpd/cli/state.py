"""Session-Globals für das interaktive CLI (Vorausschau, Risiko-Bänder)."""
from __future__ import annotations

# Session-Default für Vorausschau-Horizont (Monate); None = Modellhorizont.
FORECAST_HORIZON_MONTHS: int | None = None

# Risiko-Bänder relativ zur Basisrate im Horizont (Multiplikatoren).
# unter < mid_mult × Basis | mid_mult…high_mult × Basis = um Basisrate | ≥ high = über
RISK_BAND_MID_MULT: float = 0.85
RISK_BAND_HIGH_MULT: float = 2.5
# Fallback-Basisrate 12M, falls Modell-Metadaten keine liefern
DEFAULT_BASE_RATE_12M: float = 0.012
