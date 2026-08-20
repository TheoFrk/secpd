"""Loader für den Zenodo-Datensatz „10k-fraud-detection" (Record 17121948).

Struktur des Datensatzes (Stand der Sichtung, Amin & Aßenmacher 2025):

* ``firm_years.json`` (~4.9 GB) / ``firm_years_labels.json`` (~716 MB):
  JSON-Array, **ein Record pro Zeile**, Felder u. a.
  ``cik, name, city, state, sic, incorp_state, filing_type, fye,
  filing_date (DD-MM-YYYY), reporting_date (DD-MM-YYYY), url, mda``.
  Es gibt KEIN explizites Label-Feld und KEINE Finanzkennzahlen.
* ``aaer_mark5.csv`` (~180 kB, ``;``-separiert): SEC-AAER-Enforcement-Releases
  mit ``cik``, ``fraud_start``/``fraud_end`` (MM-YYYY), ``revoked``,
  Regelverstoß-Flags und ``fsf`` (Financial-Statement-Fraud-Indikator).

⇒ Das Label wird hier konstruiert: Ein Firm-Year gilt als „fraudulent", wenn
sich sein Geschäftsjahr-Intervall mit einem AAER-Betrugsfenster desselben
CIK überschneidet (Standard: nur ``fsf==1``, ohne ``revoked==1``).

Für die 4.9-GB-Datei wird zeilenweise gestreamt (stdlib-only, kein ijson).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COL = "label"

# --------------------------------------------------------------------------- #
# Firm-Year-JSON (Streaming)
# --------------------------------------------------------------------------- #


def iter_firm_year_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Streamt Records aus dem zeilenweisen JSON-Array (speicherschonend).

    Fallback auf ``json.load`` für kleine/abweichend formatierte Dateien.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
        if first not in ("[", ""):
            # Kein zeilenweises Array ⇒ konservativer Voll-Load.
            fh.seek(0)
            data = json.load(fh)
            yield from data
            return
        for line in fh:
            line = line.strip().rstrip(",")
            if not line or line in ("]",):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Übersprungene, nicht parsebare Zeile (%d Zeichen).", len(line))


def firm_years_to_frame(
    path: Path | str,
    *,
    text_field: str = "mda",
    max_records: int | None = None,
    min_text_chars: int = 200,
    truncate_chars: int | None = 40_000,
) -> pd.DataFrame:
    """Konvertiert das Firm-Year-JSON in einen tidy DataFrame.

    Erzeugt ``doc_id`` (``{cik}_{reporting_date}``), ``fyear`` sowie die
    Textspalte (optional trunkiert, um CSV-Größe und LLM-Kosten zu begrenzen).
    """
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(iter_firm_year_records(path)):
        if max_records is not None and i >= max_records:
            break
        text = str(rec.get(text_field) or "")
        if len(text) < min_text_chars:
            continue
        if truncate_chars is not None:
            text = text[:truncate_chars]
        rows.append(
            {
                "cik": rec.get("cik"),
                "name": rec.get("name"),
                "sic": rec.get("sic"),
                "filing_type": rec.get("filing_type"),
                "filing_date": rec.get("filing_date"),
                "reporting_date": rec.get("reporting_date"),
                text_field: text,
                "text_chars": len(text),
            }
        )
        if (i + 1) % 5_000 == 0:
            logger.info("  … %d Records gelesen", i + 1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("filing_date", "reporting_date"):
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cik", "reporting_date"]).reset_index(drop=True)
    df["fyear"] = df["reporting_date"].dt.year.astype(int)
    df["doc_id"] = (
        df["cik"].astype(int).astype(str) + "_" + df["reporting_date"].dt.strftime("%Y-%m-%d")
    )
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    logger.info("Firm-Years geladen: %d Zeilen, %d CIKs", len(df), df["cik"].nunique())
    return df


# --------------------------------------------------------------------------- #
# AAER-Tabelle & Label-Konstruktion
# --------------------------------------------------------------------------- #


def load_aaer(path: Path | str) -> pd.DataFrame:
    """Liest ``aaer_mark5.csv`` und parst die Betrugsfenster.

    ``fraud_start`` ⇒ Monatsanfang, ``fraud_end`` ⇒ Monatsende.
    """
    df = pd.read_csv(path, sep=";", dtype=str, engine="python", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    df["cik"] = pd.to_numeric(df.get("cik"), errors="coerce").astype("Int64")
    df["fraud_start"] = pd.to_datetime(df.get("fraud_start"), format="%m-%Y", errors="coerce")
    end = pd.to_datetime(df.get("fraud_end"), format="%m-%Y", errors="coerce")
    df["fraud_end"] = end + pd.offsets.MonthEnd(0)

    for flag in ("revoked", "fsf"):
        if flag in df.columns:
            df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
        else:
            df[flag] = 0

    df = df.dropna(subset=["cik", "fraud_start", "fraud_end"]).reset_index(drop=True)
    logger.info("AAER geladen: %d Releases, %d CIKs", len(df), df["cik"].nunique())
    return df


def attach_fraud_labels(
    firm_years: pd.DataFrame,
    aaer: pd.DataFrame,
    *,
    require_fsf: bool = True,
    drop_revoked: bool = True,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Label = 1, wenn Geschäftsjahr-Intervall ∩ AAER-Betrugsfenster ≠ ∅.

    Geschäftsjahr-Intervall: ``(reporting_date − 1 Jahr, reporting_date]``.
    Überschneidung: ``fraud_start <= reporting_date`` und
    ``fraud_end >= fiscal_start``.
    """
    aaer_f = aaer.copy()
    if drop_revoked:
        aaer_f = aaer_f[aaer_f["revoked"] != 1]
    if require_fsf:
        aaer_f = aaer_f[aaer_f["fsf"] == 1]
    aaer_f = aaer_f[["cik", "fraud_start", "fraud_end"]].dropna()

    fy = firm_years[["doc_id", "cik", "reporting_date"]].copy()
    fy["fiscal_start"] = fy["reporting_date"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)

    merged = fy.merge(aaer_f, on="cik", how="left")
    overlap = (merged["fraud_start"] <= merged["reporting_date"]) & (
        merged["fraud_end"] >= merged["fiscal_start"]
    )
    merged["_hit"] = overlap.fillna(False).astype(int)
    hits = merged.groupby("doc_id")["_hit"].max()

    out = firm_years.copy()
    out[label_col] = out["doc_id"].map(hits).fillna(0).astype(int)
    rate = out[label_col].mean() if len(out) else 0.0
    logger.info(
        "Labels gesetzt: %d/%d positiv (Basisrate %.2f%%)",
        int(out[label_col].sum()), len(out), 100 * rate,
    )
    return out


# --------------------------------------------------------------------------- #
# Generischer Loader für den (konvertierten) Modellierungs-Datensatz
# --------------------------------------------------------------------------- #

LABEL_CANDIDATES = ("label", "label_rating", "misstate", "fraud", "is_fraud", "target", "y", "default")
ID_CANDIDATES = ("doc_id", "accession", "accession_number")
TEXT_CANDIDATES = ("mda", "text", "item7", "item_7", "mgmt_discussion", "risk_factors")
YEAR_CANDIDATES = ("fyear", "fiscal_year", "year")


@dataclass(frozen=True)
class ResolvedColumns:
    """Ergebnis der Spalten-Auflösung — wird beim Training geloggt."""

    label_col: str
    id_col: str
    text_col: str | None
    year_col: str | None


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Liest den konvertierten Datensatz (``.csv``/``.csv.gz``/``.parquet``)."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        raise ValueError(
            "Rohes Zenodo-JSON bitte zuerst konvertieren: "
            "python scripts/convert_zenodo.py --help"
        )
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def resolve_columns(
    df: pd.DataFrame,
    *,
    label_col: str | None = None,
    id_col: str | None = None,
    text_col: str | None = None,
    year_col: str | None = None,
) -> ResolvedColumns:
    """Löst Kernspalten auf (explizite Overrides > Auto-Detection)."""

    def pick(explicit: str | None, candidates: tuple[str, ...], what: str, required: bool) -> str | None:
        if explicit:
            if explicit not in df.columns:
                raise KeyError(f"{what}-Spalte {explicit!r} nicht im Datensatz.")
            return explicit
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise KeyError(f"Keine {what}-Spalte gefunden (Kandidaten: {candidates}).")
        return None

    label = pick(label_col, LABEL_CANDIDATES, "Label", required=True)
    doc = pick(id_col, ID_CANDIDATES, "ID", required=False)
    text = pick(text_col, TEXT_CANDIDATES, "Text", required=False)
    year = pick(year_col, YEAR_CANDIDATES, "Jahr", required=False)

    if doc is None:
        if "cik" in df.columns and year is not None:
            df["doc_id"] = df["cik"].astype(str) + "_" + df[year].astype(str)
        else:
            df["doc_id"] = df.index.astype(str)
        doc = "doc_id"

    resolved = ResolvedColumns(label_col=str(label), id_col=str(doc), text_col=text, year_col=year)
    logger.info("Spalten aufgelöst: %s", resolved)
    return resolved
