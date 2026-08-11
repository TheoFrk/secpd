"""Textuelle Features: Vorverarbeitung + LLM-Extraktion als Feature-Frame.

Design-Entscheidung (bewusst KEIN LLM-Call innerhalb eines sklearn-
Transformers): Text-Features werden **vorab** pro Dokument berechnet und
per ``doc_id`` an die Tabelle gejoint („precompute-then-join").

Gründe: (1) ``pipeline.predict()`` bleibt deterministisch, schnell und ohne
Netz-/Gateway-Abhängigkeit lauffähig — wichtig auf der air-gapped Box;
(2) Batch-Verarbeitung, Caching und Fehler-Fallbacks bleiben an einer
Stelle kontrollierbar; (3) da die Extraktion pro Dokument deterministisch
und ohne Fitting ist, entsteht dadurch kein Train/Test-Leakage.
"""
from __future__ import annotations

import html
import logging
import re
from typing import Sequence

import pandas as pd

from ..llm.base import BaseLLMClient
from ..llm.schema import TextRiskProfile

logger = logging.getLogger(__name__)

_TABLE_BLOCK = re.compile(r"##TABLE_START.*?##TABLE_END", flags=re.DOTALL)
_WS = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")


def prepare_text(
    text: str,
    *,
    max_chars: int = 12_000,
    strip_tables: bool = True,
    tail_share: float = 0.2,
) -> str:
    """Bereitet MD&A-Text für die LLM-Analyse auf.

    * HTML-Entities dekodieren, Tag-Reste entfernen.
    * ``##TABLE_START…##TABLE_END``-Blöcke (Zahlensuppe im Zenodo-Datensatz)
      entfernen — sie verwässern die sprachliche Analyse.
    * Whitespace normalisieren, dann Head+Tail-Trunkierung: MD&A-Anfang
      (Lagebeschreibung) und -Ende (Liquidität/Going-Concern) sind meist
      informativer als der Mittelteil.
    """
    text = html.unescape(text or "")
    if strip_tables:
        text = _TABLE_BLOCK.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    if len(text) <= max_chars:
        return text
    tail = int(max_chars * tail_share)
    head = max_chars - tail - 20
    return text[:head] + " […] " + text[-tail:]


def extract_text_features(
    df: pd.DataFrame,
    *,
    client: BaseLLMClient,
    text_col: str,
    id_col: str = "doc_id",
    max_chars: int = 12_000,
    prefix: str = "llm_",
    keep_summary: bool = True,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Berechnet LLM-Features je Dokument (eine Zeile pro ``id_col``).

    Fehler einzelner Dokumente führen zu einem Fallback-Profil
    (``confidence=0``) statt zum Batch-Abbruch.
    """
    ids: Sequence[str] = df[id_col].astype(str).tolist()
    texts: Sequence[str] = df[text_col].astype(str).tolist()

    rows: list[dict[str, object]] = []
    n = len(df)
    logger.info("LLM-Textanalyse: %d Dokumente via %s", n, client.name)
    for i, (doc_id, raw) in enumerate(zip(ids, texts), start=1):
        prepared = prepare_text(raw, max_chars=max_chars)
        try:
            profile = client.analyze(prepared, doc_id=doc_id)
        except Exception as exc:  # noqa: BLE001 — robust weiterlaufen
            logger.warning("Fallback für doc_id=%s: %s", doc_id, exc)
            profile = TextRiskProfile.fallback(reason=str(exc)[:120])
        row: dict[str, object] = {id_col: doc_id, **profile.to_features(prefix=prefix)}
        if keep_summary:
            row[f"{prefix}risk_summary"] = profile.risk_summary
        rows.append(row)
        if i % progress_every == 0 or i == n:
            logger.info("  … %d/%d Dokumente analysiert", i, n)

    return pd.DataFrame(rows)


def text_feature_names(prefix: str = "llm_") -> list[str]:
    """Numerische LLM-Feature-Spalten (für Pipeline-Definitionen)."""
    return TextRiskProfile.feature_names(prefix=prefix)
