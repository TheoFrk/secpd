"""Firm-Year-Frames mit einem gespeicherten PD-Bundle scoren."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..data.events import add_event_features, load_events
from ..data.zenodo import load_dataset, resolve_columns
from ..features.financial import add_financial_features
from ..features.textual import (
    attach_text_features,
    extract_keyword_features,
    needs_keyword_columns,
    needs_llm_columns,
)
from ..llm import get_llm_client
from ..models.ensemble import EnsembleWeights, combine_probabilities
from ..models.persistence import (
    BUNDLE_KIND_ENSEMBLE,
    BUNDLE_KIND_SINGLE,
    load_any,
)

logger = logging.getLogger(__name__)


def score_firm_years(
    data: str | Path | pd.DataFrame,
    model: str | Path,
    *,
    events: str | Path | None = None,
    llm: str | None = None,
    max_chars: int = 12_000,
) -> pd.DataFrame:
    """Gibt das Eingabe-Frame plus ``pd_score`` zurück (eine Zeile je Firm-Year)."""
    payload = load_any(model)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        path = Path(data)
        if path.suffix.lower() == ".json":
            payload_json = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(
                payload_json if isinstance(payload_json, list) else [payload_json]
            )
        else:
            df = load_dataset(path)
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not any(c in df.columns for c in ("label", "misstate", "fraud", "target", "y")):
        df["label"] = 0
    cols = resolve_columns(df)
    df = df.reset_index(drop=True)
    df, _ = add_financial_features(df)

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        feature_cols = list(payload["feature_cols"])
    elif payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        feature_cols = list(
            dict.fromkeys(
                list(payload["financial"]["feature_cols"])
                + list(payload["text"]["feature_cols"])
            )
        )
    else:
        raise ValueError(f"Unbekannter Bundle-Typ: {payload['kind']!r}")

    needs_evt = any(str(c).startswith("evt_") for c in feature_cols)
    if needs_evt:
        if events is None:
            raise ValueError("Bundle erwartet evt_*-Features — --events setzen.")
        df, _ = add_event_features(df, load_events(events))

    if cols.text_col is not None and (
        needs_llm_columns(feature_cols) or needs_keyword_columns(feature_cols)
    ):
        if needs_llm_columns(feature_cols):
            client = get_llm_client(llm, cache_only=True)
            df, _ = attach_text_features(
                df,
                client=client,
                text_col=cols.text_col,
                id_col=cols.id_col,
                max_chars=max_chars,
            )
        else:
            kw = extract_keyword_features(df, text_col=cols.text_col, id_col=cols.id_col)
            df = df.merge(kw, on=cols.id_col, how="left")

    for c in feature_cols:
        if c not in df.columns:
            df[c] = float("nan")

    if payload["kind"] == BUNDLE_KIND_SINGLE:
        scores = payload["pipeline"].predict_proba(df)[:, 1]
    else:
        p_fin = payload["financial"]["pipeline"].predict_proba(df)[:, 1]
        p_txt = payload["text"]["pipeline"].predict_proba(df)[:, 1]
        w = payload.get("weights", {})
        weights = EnsembleWeights(
            w_financial=float(w.get("w_financial", 0.6)),
            w_text=float(w.get("w_text", 0.4)),
        )
        scores = combine_probabilities(p_fin, p_txt, weights, method="logit")

    df["pd_score"] = scores
    logger.info("PD-Scores: n=%d mean=%.4f", len(df), float(pd.Series(scores).mean()))
    return df
