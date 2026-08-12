"""Textuelle Features: Vorverarbeitung + LLM-Extraktion + Distress-Keywords.

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

#: PD-nahe Phrasen im vollen MD&A (nicht im 12k-LLM-Ausschnitt).
#: Zähler, nicht Binär — der RF kann Schwellen lernen.
_KEYWORD_GROUPS: dict[str, tuple[re.Pattern[str], ...]] = {
    "txt_going_concern": (
        re.compile(r"going[\s-]+concern", re.I),
        re.compile(r"substantial[\s-]+doubt", re.I),
        re.compile(r"ability to continue as a going concern", re.I),
    ),
    "txt_liquidity_stress": (
        re.compile(r"insufficient liquidity", re.I),
        re.compile(r"liquidity (?:shortfall|crisis|concerns?|risks?)", re.I),
        re.compile(r"working capital (?:deficit|deficiency|shortfall)", re.I),
        re.compile(r"unable to (?:meet|satisfy) .{0,40}(?:debt|obligations|cash)", re.I),
    ),
    "txt_covenant": (
        re.compile(r"covenant (?:violation|breach|default|waiver)", re.I),
        re.compile(r"(?:violat\w+|breach\w+|waiv\w+)\s.{0,40}covenant", re.I),
        re.compile(r"non[- ]compliance with .{0,40}covenant", re.I),
    ),
    "txt_restructuring": (
        re.compile(r"\brestructur(?:e|ed|ing)\b", re.I),
        re.compile(r"recapitali[sz]ation", re.I),
        re.compile(r"debt (?:exchange|forbearance)", re.I),
    ),
    "txt_bankruptcy_lang": (
        re.compile(r"chapter\s*11", re.I),
        re.compile(r"\bbankrupt(?:cy|cies)?\b", re.I),
        re.compile(r"\breceivership\b", re.I),
    ),
}


def prepare_text(
    text: str,
    *,
    max_chars: int | None = 12_000,
    strip_tables: bool = True,
    tail_share: float = 0.2,
) -> str:
    """Bereitet MD&A-Text für die LLM-Analyse auf.

    * HTML-Entities dekodieren, Tag-Reste entfernen.
    * ``##TABLE_START…##TABLE_END``-Blöcke (Zahlensuppe im Zenodo-Datensatz)
      entfernen — sie verwässern die sprachliche Analyse.
    * Whitespace normalisieren, dann optional Head+Tail-Trunkierung: MD&A-Anfang
      (Lagebeschreibung) und -Ende (Liquidität/Going-Concern) sind meist
      informativer als der Mittelteil.
    """
    text = html.unescape(text or "")
    if strip_tables:
        text = _TABLE_BLOCK.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    if max_chars is None or len(text) <= max_chars:
        return text
    tail = int(max_chars * tail_share)
    head = max_chars - tail - 20
    return text[:head] + " […] " + text[-tail:]


def _count_group(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return int(sum(len(p.findall(text)) for p in patterns))


def extract_keyword_features(
    df: pd.DataFrame,
    *,
    text_col: str,
    id_col: str = "doc_id",
) -> pd.DataFrame:
    """Zählt Distress-Phrasen im vollen (bereinigten) MD&A."""
    ids: Sequence[str] = df[id_col].astype(str).tolist()
    texts: Sequence[str] = df[text_col].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for doc_id, raw in zip(ids, texts):
        cleaned = prepare_text(raw, max_chars=None)
        row: dict[str, object] = {id_col: doc_id}
        for name, patterns in _KEYWORD_GROUPS.items():
            row[name] = float(_count_group(cleaned, patterns))
        rows.append(row)
    return pd.DataFrame(rows)


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
    (``confidence=0``) statt zum Batch-Abbruch. Alle Profilfelder werden
    geschrieben (alte Bundles); das Combined-Modell nutzt nur
    :func:`text_feature_names`.
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


def attach_text_features(
    df: pd.DataFrame,
    *,
    client: BaseLLMClient,
    text_col: str,
    id_col: str = "doc_id",
    max_chars: int = 12_000,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, list[str]]:
    """Joint LLM-Profil + Keyword-Zähler; liefert Modell-Featurenamen."""
    llm = extract_text_features(
        df,
        client=client,
        text_col=text_col,
        id_col=id_col,
        max_chars=max_chars,
        progress_every=progress_every,
    )
    kw = extract_keyword_features(df, text_col=text_col, id_col=id_col)
    text_side = llm.merge(kw, on=id_col, how="left")
    out = df.merge(text_side, on=id_col, how="left")
    return out, combined_text_feature_names()


def text_feature_names(prefix: str = "llm_") -> list[str]:
    """LLM-Spalten, die ins Combined-Modell eingehen (schlank)."""
    return TextRiskProfile.feature_names(prefix=prefix, for_model=True)


def keyword_feature_names() -> list[str]:
    return list(_KEYWORD_GROUPS.keys())


def combined_text_feature_names() -> list[str]:
    return text_feature_names() + keyword_feature_names()


def needs_llm_columns(feature_cols: Sequence[str]) -> bool:
    return any(str(c).startswith("llm_") for c in feature_cols)


def needs_keyword_columns(feature_cols: Sequence[str]) -> bool:
    return any(str(c).startswith("txt_") for c in feature_cols)
