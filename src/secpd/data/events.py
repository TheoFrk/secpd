"""8-K-Events aus der EDGAR-Submissions-API: Default-Label + Event-Features.

Zwei Aufgaben, streng getrennt:

1. **Default-Label** (Insolvenz = Zielvariable): Ein 8-K mit Item
   ``1.03`` (neues Regime) bzw. ``3`` (altes Regime) meldet
   "Bankruptcy or Receivership". ``label_default = 1``, wenn diese Meldung
   innerhalb des Prognosehorizonts nach dem Bilanzstichtag erfolgt —
   klassische 1-Jahres-PD-Definition. Das Insolvenz-Item fließt
   AUSSCHLIESSLICH ins Label, niemals in Features (Leakage-Regel).
2. **Event-Features** (Frühindikatoren): PIT-saubere Zähler über ein
   rollierendes Fenster bis einschließlich des 10-K-Filing-Datums —
   Auditor-Wechsel, Officer-Abgänge, Covenant-Brüche, Impairments,
   Delisting-Notices.

API-Fakten: ``https://data.sec.gov/submissions/CIK{cik:010d}.json`` liefert
``filings.recent`` als parallele Arrays; ältere Filings liegen in
Zusatzdateien unter ``filings.files[*].name`` (gleiches Array-Format, ohne
Wrapper). Pflicht-``User-Agent`` via ``SECPD_SEC_UA``.

Die 8-K-Item-Nummerierung wechselte am 2004-08-23 — Matching daher mit
Datums-Guard und exaktem Token-Vergleich (``"3"`` darf nie ``"3.01"`` treffen).

**Default-Policy (ab Daten-Sanierung):** Submissions-API-``items`` vor dem
Regimewechsel sind unzuverlässig (leere ``ITEM INFORMATION``). Deshalb gelten
Bankruptcy-Labels und Event-Features standardmäßig nur für Filings
``≥ REGIME_SWITCH``. Legacy-Treffer bleiben über ``trust_legacy_regime=True``
für Audits erreichbar, fließen aber nicht mehr ins Training.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"

#: Stichtag des Wechsels der 8-K-Item-Nummerierung.
REGIME_SWITCH = pd.Timestamp("2004-08-23")

#: Ab diesem Geschäftsjahr sind XBRL-Companyfacts flächig verfügbar.
MIN_FYEAR_WITH_FINANCIALS = 2009

#: Konzept → (Item neues Regime ≥ 2004-08-23, Item altes Regime davor oder None).
ITEM_MAP: dict[str, tuple[str, str | None]] = {
    "bankruptcy": ("1.03", "3"),            # → NUR Label, nie Feature!
    "auditor_change": ("4.01", "4"),
    "officer_departure": ("5.02", "6"),
    "covenant_accel": ("2.04", None),
    "impairment": ("2.06", None),
    "delisting": ("3.01", None),
    "restatement": ("4.02", None),          # nahe am Fraud-Label — Default: aus.
}

#: Konzepte, die als Features zugelassen sind (Reihenfolge = Spaltenreihenfolge).
FEATURE_CONCEPTS: tuple[str, ...] = (
    "auditor_change",
    "officer_departure",
    "covenant_accel",
    "impairment",
    "delisting",
)

DEFAULT_LABEL_COL = "label_default"


def parse_label_concepts(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """``'bankruptcy,delisting'`` → Tuple; Default nur Bankruptcy."""
    if raw is None:
        return ("bankruptcy",)
    if isinstance(raw, str):
        parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    else:
        parts = tuple(str(p).strip() for p in raw if str(p).strip())
    if not parts:
        return ("bankruptcy",)
    unknown = [c for c in parts if c not in ITEM_MAP]
    if unknown:
        raise ValueError(f"Unbekannte Label-Konzepte: {unknown} (bekannt: {sorted(ITEM_MAP)})")
    return parts


def feature_exclusions_for_labels(label_concepts: Sequence[str]) -> tuple[str, ...]:
    """Label-Konzepte, die sonst als evt_*-Feature leaken würden."""
    return tuple(c for c in label_concepts if c in FEATURE_CONCEPTS)


# --------------------------------------------------------------------------- #
# Fetching (Online — Home-Setup)
# --------------------------------------------------------------------------- #


def fetch_submissions(
    cik: int,
    *,
    user_agent: str,
    cache_dir: Path | str = "data/raw/edgar_submissions",
    force: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Lädt das Submissions-JSON eines CIK (mit Datei-Cache, resumefähig).

    Der Cache macht Volläufe unterbrechbar: bereits geladene CIKs werden
    beim nächsten Aufruf übersprungen.
    """
    if not user_agent:
        raise RuntimeError(
            "SEC verlangt einen User-Agent — SECPD_SEC_UA setzen, "
            'z. B. "Commerzbank Praktikum vorname.nachname@example.com".'
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"CIK{int(cik):010d}.json"
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    resp = requests.get(
        SUBMISSIONS_URL.format(cik=int(cik)),
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _iter_filing_arrays(
    payload: dict[str, Any],
    *,
    cik: int,
    user_agent: str,
    cache_dir: Path | str,
    force: bool,
    sleep_s: float,
) -> Iterable[dict[str, list[Any]]]:
    """Liefert alle Array-Blöcke: ``filings.recent`` + sämtliche Zusatzseiten."""
    filings = payload.get("filings", {})
    recent = filings.get("recent")
    if recent:
        yield recent
    cache_dir = Path(cache_dir)
    for page in filings.get("files", []) or []:
        name = page.get("name")
        if not name:
            continue
        page_file = cache_dir / name
        if page_file.exists() and not force:
            yield json.loads(page_file.read_text(encoding="utf-8"))
            continue
        time.sleep(sleep_s)
        resp = requests.get(
            SUBMISSIONS_PAGE_URL.format(name=name),
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        page_file.write_text(json.dumps(data), encoding="utf-8")
        yield data


def extract_8k_rows(arrays: dict[str, list[Any]], cik: int) -> list[dict[str, Any]]:
    """Zieht 8-K-/8-K/A-Zeilen aus einem parallelen Array-Block."""
    forms = arrays.get("form", []) or []
    dates = arrays.get("filingDate", []) or []
    items = arrays.get("items", []) or []
    accs = arrays.get("accessionNumber", []) or []
    rows: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue
        rows.append(
            {
                "cik": int(cik),
                "filing_date_8k": dates[i] if i < len(dates) else None,
                "items": (items[i] if i < len(items) else "") or "",
                "accession": accs[i] if i < len(accs) else "",
            }
        )
    return rows


def build_events_table(
    ciks: Sequence[int],
    *,
    user_agent: str,
    cache_dir: Path | str = "data/raw/edgar_submissions",
    sleep_s: float = 0.15,
    force: bool = False,
) -> pd.DataFrame:
    """Lädt Submissions für alle CIKs und extrahiert die 8-K-Eventliste.

    Fehlertolerant: einzelne CIK-Fehler werden geloggt, der Lauf geht weiter
    (Muster wie ``build_financials_panel``).
    """
    all_rows: list[dict[str, Any]] = []
    unique = sorted({int(c) for c in ciks if pd.notna(c)})
    cache_root = Path(cache_dir)
    for n, cik in enumerate(unique, start=1):
        cached = (cache_root / f"CIK{int(cik):010d}.json").exists() and not force
        try:
            payload = fetch_submissions(
                cik, user_agent=user_agent, cache_dir=cache_dir, force=force
            )
            for arrays in _iter_filing_arrays(
                payload, cik=cik, user_agent=user_agent,
                cache_dir=cache_dir, force=force, sleep_s=sleep_s,
            ):
                all_rows.extend(extract_8k_rows(arrays, cik))
        except Exception as exc:  # noqa: BLE001 — Sammellauf soll weiterlaufen
            logger.warning("CIK %s übersprungen: %s", cik, exc)
        if n % 25 == 0:
            logger.info("  … %d/%d CIKs verarbeitet", n, len(unique))
        if not cached:
            time.sleep(sleep_s)

    df = pd.DataFrame(all_rows, columns=["cik", "filing_date_8k", "items", "accession"])
    if not df.empty:
        df["filing_date_8k"] = pd.to_datetime(df["filing_date_8k"], errors="coerce")
        df = df.dropna(subset=["filing_date_8k"]).reset_index(drop=True)
    logger.info("8-K-Events extrahiert: %d Zeilen aus %d CIKs", len(df), len(unique))
    return df


def log_item_coverage(events: pd.DataFrame) -> None:
    """Item-Coverage je Dekade — alte Filings sind lückenhaft kodiert."""
    if events.empty:
        logger.info("Keine Events — keine Coverage-Statistik.")
        return
    tmp = events.copy()
    tmp["decade"] = (tmp["filing_date_8k"].dt.year // 10) * 10
    tmp["has_items"] = tmp["items"].astype(str).str.strip().ne("")
    cov = tmp.groupby("decade")["has_items"].agg(["count", "mean"])
    for decade, row in cov.iterrows():
        logger.info(
            "  Item-Coverage %ds: %d 8-Ks, %.1f%% mit Items",
            int(decade), int(row["count"]), 100 * row["mean"],
        )


# --------------------------------------------------------------------------- #
# Offline-Verarbeitung: Laden, Item-Matching, Label, Features
# --------------------------------------------------------------------------- #


def load_events(path: Path | str) -> pd.DataFrame:
    """Liest die Eventliste (``cik, filing_date_8k, items, accession``)."""
    df = pd.read_csv(path, dtype={"items": str, "accession": str})
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["filing_date_8k"] = pd.to_datetime(df["filing_date_8k"], errors="coerce")
    df["items"] = df["items"].fillna("")
    df = df.dropna(subset=["cik", "filing_date_8k"]).reset_index(drop=True)
    logger.info("Events geladen: %d 8-Ks, %d CIKs", len(df), df["cik"].nunique())
    return df


def _tokens(items: str) -> set[str]:
    return {t.strip() for t in str(items).split(",") if t.strip()}


def match_concept(items: str, filing_date: pd.Timestamp, concept: str) -> bool:
    """Exaktes Token-Match mit Regime-Guard (kein Substring-Match)."""
    new_code, old_code = ITEM_MAP[concept]
    toks = _tokens(items)
    if pd.isna(filing_date):
        return False
    if filing_date >= REGIME_SWITCH:
        return new_code in toks
    return old_code is not None and old_code in toks


def mark_concepts(
    events: pd.DataFrame,
    *,
    trust_legacy_regime: bool = False,
) -> pd.DataFrame:
    """Ergänzt je Konzept eine bool-Spalte ``is_<konzept>``.

    Ohne ``trust_legacy_regime`` zählen nur Treffer ab ``REGIME_SWITCH``
    (Item-Metadaten der Submissions-API davor sind nicht belastbar).
    """
    out = events.copy()
    tok_sets = out["items"].map(_tokens)
    is_new = out["filing_date_8k"] >= REGIME_SWITCH
    for concept, (new_code, old_code) in ITEM_MAP.items():
        hit_new = tok_sets.map(lambda s, c=new_code: c in s) & is_new
        if trust_legacy_regime and old_code is not None:
            hit_old = tok_sets.map(lambda s, c=old_code: c in s) & ~is_new
        else:
            hit_old = pd.Series(False, index=out.index)
        out[f"is_{concept}"] = (hit_new | hit_old).astype(bool)
    return out


def credit_event_dates(
    events: pd.DataFrame,
    *,
    concepts: Sequence[str] = ("bankruptcy",),
    trust_legacy_regime: bool = False,
) -> pd.Series:
    """Erstes Credit-Event-Datum je CIK (Bankruptcy und/oder Delisting)."""
    wanted = tuple(concepts) or ("bankruptcy",)
    unknown = [c for c in wanted if c not in ITEM_MAP]
    if unknown:
        raise ValueError(f"Unbekannte Label-Konzepte: {unknown}")
    marked = mark_concepts(events, trust_legacy_regime=trust_legacy_regime)
    mask = pd.Series(False, index=marked.index)
    for concept in wanted:
        mask = mask | marked[f"is_{concept}"]
    hit = marked.loc[mask, ["cik", "filing_date_8k"]]
    if hit.empty:
        return pd.Series(dtype="datetime64[ns]", name="credit_event_date")
    out = hit.groupby("cik")["filing_date_8k"].min()
    out.name = "credit_event_date"
    return out


def bankruptcy_dates(
    events: pd.DataFrame,
    *,
    trust_legacy_regime: bool = False,
) -> pd.Series:
    """Erstes Bankruptcy-8-K je CIK (Index: cik, Werte: Timestamp)."""
    out = credit_event_dates(
        events, concepts=("bankruptcy",), trust_legacy_regime=trust_legacy_regime
    )
    out.name = "bankruptcy_date"
    return out


def annotate_default_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_months: int = 12,
    trust_legacy_regime: bool = False,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Hängt ``label_default`` + ``bankruptcy_date`` an — ohne Zeilen zu droppen.

    Für UI/Scoring-Verläufe: jedes Firm-Year behält seine Zeile; Label 0/1
    bzw. fehlendes Event bleibt sichtbar. ``post_petition`` markiert 10-Ks,
    die nach dem Insolvenzantrag eingereicht wurden (kein gültiger
    Prognosepunkt — MD&A kennt Chapter 11).
    """
    out = df.copy()
    rep = pd.to_datetime(out.get("reporting_date"), errors="coerce")
    bk = bankruptcy_dates(events, trust_legacy_regime=trust_legacy_regime)
    out["bankruptcy_date"] = out["cik"].map(bk)
    fil = pd.to_datetime(out.get("filing_date"), errors="coerce")
    post_petition = (
        out["bankruptcy_date"].notna()
        & fil.notna()
        & (fil > out["bankruptcy_date"])
    )
    out["post_petition"] = post_petition.fillna(False)
    horizon_end = rep + pd.DateOffset(months=horizon_months)
    label = (
        rep.notna()
        & out["bankruptcy_date"].notna()
        & (rep < out["bankruptcy_date"])
        & (out["bankruptcy_date"] <= horizon_end)
    )
    out[label_col] = label.fillna(False).astype(int)
    return out


def attach_default_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_months: int = 12,
    drop_censored: bool = True,
    drop_post_bankruptcy: bool = True,
    trust_legacy_regime: bool = False,
    label_col: str = DEFAULT_LABEL_COL,
    label_concepts: Sequence[str] = ("bankruptcy",),
) -> pd.DataFrame:
    """Konstruiert das Default-Label über den Prognosehorizont.

    ``label = 1`` ⟺ ``reporting_date < event_date ≤ reporting_date +
    horizon_months``. Das Credit-Event ist ZIEL, nicht Wissen des Modells —
    Label-Konzepte (Default: nur Bankruptcy 1.03) existieren bewusst nicht
    als Features.

    Post-Event-Zeilen: nicht nur ``reporting_date >= event_date``, sondern
    auch ``filing_date > event_date``. Sonst bleiben 10-Ks im Sample, deren
    MD&A schon Chapter-11-Prosa enthält (Kodak, PG&E: Filing ~6 Wochen nach
    dem Insolvenzantrag, ``reporting_date`` aber noch davor).

    ``label_concepts`` kann z. B. ``("bankruptcy", "delisting")`` sein
    (früherer Distress-Korb). Delisting fließt dann nicht mehr in ``evt_*``.
    """
    out = df.copy()
    rep = pd.to_datetime(out.get("reporting_date"), errors="coerce")
    n_no_date = int(rep.isna().sum())
    if n_no_date:
        logger.warning(
            "%d Zeilen ohne parsebares reporting_date — für das Default-Label "
            "gedroppt.", n_no_date,
        )
    out = out.loc[rep.notna()].copy()
    rep = rep.loc[rep.notna()]

    concepts = tuple(label_concepts) or ("bankruptcy",)
    ev_dates = credit_event_dates(
        events, concepts=concepts, trust_legacy_regime=trust_legacy_regime
    )
    if not trust_legacy_regime:
        logger.info(
            "Default-Label: Legacy-Regime vor %s ignoriert "
            "(trust_legacy_regime=False).",
            REGIME_SWITCH.date(),
        )
    out["_bk_date"] = out["cik"].map(ev_dates)
    horizon_end = rep + pd.DateOffset(months=horizon_months)

    label = ((rep < out["_bk_date"]) & (out["_bk_date"] <= horizon_end)).fillna(False)
    out[label_col] = label.astype(int)

    n_post = 0
    if drop_post_bankruptcy:
        fil = pd.to_datetime(out.get("filing_date"), errors="coerce")
        post_rep = (rep >= out["_bk_date"]).fillna(False)
        # 10-K nach dem Antrag: MD&A kennt Chapter 11, auch wenn der
        # Bilanzstichtag noch vor dem Event liegt (Kodak, PG&E).
        post_fil = (
            fil.notna()
            & out["_bk_date"].notna()
            & (fil > out["_bk_date"])
        )
        post = post_rep | post_fil.fillna(False)
        n_post = int(post.sum())
        out, rep, horizon_end = out.loc[~post], rep.loc[~post], horizon_end.loc[~post]

    n_cens = 0
    if drop_censored and not events.empty:
        global_max = events["filing_date_8k"].max()
        censored = (horizon_end > global_max) & (out[label_col] == 0)
        n_cens = int(censored.sum())
        out = out.loc[~censored]

    out = out.drop(columns=["_bk_date"]).reset_index(drop=True)
    logger.info(
        "Default-Label (Horizont %d M, Konzepte=%s): %d Events bekannt | "
        "%d/%d positiv (Basisrate %.2f%%) | gedroppt: %d post-event "
        "(inkl. post-petition-Filings), %d rechtszensiert",
        horizon_months, ",".join(concepts), len(ev_dates),
        int(out[label_col].sum()), len(out),
        100 * out[label_col].mean() if len(out) else 0.0, n_post, n_cens,
    )
    return out


# --------------------------------------------------------------------------- #
# Event-Features (Frühindikatoren — nie das Default-Label selbst)
# --------------------------------------------------------------------------- #


def event_feature_names(
    prefix: str = "evt_",
    *,
    include_restatement_flag: bool = False,
    exclude_concepts: Sequence[str] = (),
) -> list[str]:
    concepts = [c for c in FEATURE_CONCEPTS if c not in set(exclude_concepts)]
    names = [f"{prefix}n_8k"] + [f"{prefix}n_{c}" for c in concepts]
    if include_restatement_flag:
        names.append(f"{prefix}n_restatement")
    return names


def add_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: int = 365,
    prefix: str = "evt_",
    include_restatement_flag: bool = False,
    trust_legacy_regime: bool = False,
    exclude_concepts: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """PIT-saubere Event-Zähler je Firm-Year.

    Fenster: ``(filing_date_10k − window_days, filing_date_10k]`` — nur 8-Ks
    bis einschließlich des 10-K-Filing-Datums; alles danach wäre Look-ahead.
    Kein Event ⇒ 0 (nicht NaN): die Abwesenheit von 8-Ks ist Information.
    Bankruptcy (1.03/alt-3) ist bewusst KEIN Feature (Label-Leakage).
    Konzepte, die im Label stecken (z. B. Delisting), über ``exclude_concepts``
    ebenfalls raushalten.
    """
    out = df.copy()
    feature_cols = event_feature_names(
        prefix,
        include_restatement_flag=include_restatement_flag,
        exclude_concepts=exclude_concepts,
    )
    for col in feature_cols:
        out[col] = 0

    upper = pd.to_datetime(out.get("filing_date"), errors="coerce")
    fallback = upper.isna()
    if fallback.any():
        logger.warning(
            "%d Zeilen ohne filing_date — reporting_date als PIT-Obergrenze "
            "verwendet (konservativer).", int(fallback.sum()),
        )
        upper = upper.fillna(pd.to_datetime(out.get("reporting_date"), errors="coerce"))

    valid = upper.notna() & out["cik"].notna()
    if events.empty or not valid.any():
        return out, feature_cols

    concepts = [c for c in FEATURE_CONCEPTS if c not in set(exclude_concepts)]
    if include_restatement_flag:
        concepts = concepts + ["restatement"]
    marked = mark_concepts(events, trust_legacy_regime=trust_legacy_regime)[
        ["cik", "filing_date_8k"] + [f"is_{c}" for c in concepts]
    ]

    left = pd.DataFrame(
        {"_row": out.index[valid], "cik": out.loc[valid, "cik"].astype("Int64"),
         "_upper": upper.loc[valid]}
    )
    merged = left.merge(marked, on="cik", how="inner")
    in_window = (
        (merged["filing_date_8k"] > merged["_upper"] - pd.Timedelta(days=window_days))
        & (merged["filing_date_8k"] <= merged["_upper"])
    )
    merged = merged.loc[in_window]
    if merged.empty:
        return out, feature_cols

    agg = merged.groupby("_row").agg(
        **{f"{prefix}n_8k": ("filing_date_8k", "size")},
        **{f"{prefix}n_{c}": (f"is_{c}", "sum") for c in concepts},
    )
    for col in agg.columns:
        out.loc[agg.index, col] = agg[col].astype(int)
    logger.info(
        "Event-Features (%d, Fenster %d Tage) für %d/%d Firm-Years mit ≥1 8-K.",
        len(feature_cols), window_days, int((out[f"{prefix}n_8k"] > 0).sum()), len(out),
    )
    return out, feature_cols
