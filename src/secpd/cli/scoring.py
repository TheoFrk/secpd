"""Scoring-Flow: Firm-Years laden, Features, Score, lesbare Ausgabe."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secpd.cli import state
from secpd.cli.catalog import (
    active_model_meta,
    describe_model,
    select_default_pd_path,
    select_model_path,
    warn_model_coherence,
)
from secpd.cli.debug import llm_client_label, print_llm_cache_report, raise_if_cache_miss_forbidden
from secpd.cli.paths import (
    EVENTS,
    FIRM_YEARS,
    LABELED,
    PANEL,
    RATINGS,
    ROOT,
    SCORES_DIR,
    SUBMISSIONS_CACHE,
)
from secpd.cli.ui import (
    C,
    _base_rate_12m_from_meta as base_rate_12m_from_meta,
    ask,
    ask_forecast_horizon,
    banner,
    clear,
    fmt_score,
    horizon_label,
    hr,
    pause,
    risk_band,
    scale_pd,
)
from secpd.config import load_settings
from secpd.data.edgar import build_financials_panel
from secpd.data.events import (
    MIN_FYEAR_WITH_FINANCIALS,
    add_event_features,
    annotate_default_labels,
    build_events_table,
    load_events,
)
from secpd.data.ratings import attach_rating_labels, format_notch, load_ratings, notch_to_letter
from secpd.data.zenodo import resolve_columns
from secpd.features.financial import add_financial_features
from secpd.features.textual import attach_text_features, extract_keyword_features, needs_keyword_columns, needs_llm_columns
from secpd.llm import get_llm_client
from secpd.models.ensemble import EnsembleWeights, combine_probabilities
from secpd.models.pipeline import predict_output
from secpd.models.persistence import BUNDLE_KIND_ENSEMBLE, BUNDLE_KIND_SINGLE, load_any

# Modell-Anzeige in der CLI: Moody's-Buchstaben (Aaa … C), intern weiter Notch 1–21.
_RATING_SCALE = "moodys"

# ensure_sec_ua is defined in settings; imported lazily in input_from_edgar.

def ensure_sec_ua() -> str | None:
    from secpd.cli.settings import ensure_sec_ua as _impl
    return _impl()


# --------------------------------------------------------------------------- #
# Daten laden (Label-Set, Panel, EDGAR)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


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
    elif payload["kind"] == BUNDLE_KIND_ENSEMBLE:
        feature_cols = list(
            dict.fromkeys(
                list(payload["financial"]["feature_cols"])
                + list(payload["text"]["feature_cols"])
            )
        )
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

    if cols.text_col and (
        needs_llm_columns(feature_cols) or needs_keyword_columns(feature_cols)
    ):
        if cols.text_col not in work.columns:
            raise RuntimeError("Modell braucht Text (MD&A), aber keine Textspalte gefunden.")
        if needs_llm_columns(feature_cols):
            if work[cols.text_col].fillna("").astype(str).str.len().max() < 50:
                raise RuntimeError("MD&A-Text fehlt oder ist zu kurz für das Combined-Modell.")
            settings = load_settings()
            client = get_llm_client(None, cache_only=settings.llm_cache_only)
            print(
                f"  {C.DIM}Textanalyse ({len(work)} Dokumente, "
                f"LLM={llm_client_label(client)}) …{C.RESET}"
            )
            work, _ = attach_text_features(
                work,
                client=client,
                text_col=cols.text_col,
                id_col=cols.id_col,
                progress_every=10_000,
            )
            print_llm_cache_report(client)
            raise_if_cache_miss_forbidden(client)
        else:
            kw = extract_keyword_features(
                work, text_col=cols.text_col, id_col=cols.id_col
            )
            work = work.merge(kw, on=cols.id_col, how="left")

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
        meta = dict(payload.get("metadata") or {})
        task = meta.get("task")
        scores = predict_output(payload["pipeline"], work, task=task)
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
    meta = dict(payload.get("metadata") or {})
    task = str(meta.get("task") or "")
    is_ordinal = task == "regression" or meta.get("rating_target") == "ordinal"
    if is_ordinal:
        out["rating_notch"] = scores
        out["rating_letter"] = [notch_to_letter(x, scale=_RATING_SCALE) for x in scores]
        p_exact, p_pm1 = rating_tree_vote_prob(payload["pipeline"], work, pred=scores)
        out["rating_p_exact"] = p_exact
        out["rating_p_pm1"] = p_pm1
        out["pd_score"] = float("nan")
    else:
        out["pd_score"] = scores
    if "cik" in work.columns:
        out["cik"] = work["cik"].values
    if "fyear" in work.columns:
        out["fyear"] = work["fyear"].values
    if "doc_id" in work.columns and "doc_id" not in out.columns:
        out["doc_id"] = work["doc_id"].values
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
            if "post_petition" in annotated.columns:
                out["post_petition"] = annotated["post_petition"].values
        elif "label_default" in work.columns:
            out["label_default"] = work["label_default"].values
    if "total_assets" in work.columns:
        out["has_financials"] = work["total_assets"].notna().values
    for col in ("filing_date", "reporting_date"):
        if col in work.columns:
            out[col] = work[col].values
    return out.sort_values("fyear" if "fyear" in out.columns else cols.id_col)


def rating_tree_vote_prob(
    pipe,
    work,
    pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """P(genau diese Note) und P(±1 Notch) aus den RF-Baumstimmen.

    Jeder Baum gibt eine Notch-Regression; gerundet = Stimme. Anteil der
    Bäume mit der angezeigten Note (Pipeline-Mittel, gerundet) bzw.
    innerhalb ±1 Notch.
    """
    n = len(work)
    empty = np.full(n, np.nan)
    try:
        last = pipe.steps[-1][1]
        estimators = getattr(last, "estimators_", None)
        if not estimators:
            return empty, empty
        xt = pipe[:-1].transform(work)
        votes = np.column_stack(
            [np.clip(est.predict(xt), 1.0, 21.0) for est in estimators]
        )
        rounded = np.clip(np.round(votes), 1, 21)
        if pred is None:
            target = np.clip(np.round(votes.mean(axis=1)), 1, 21)
        else:
            target = np.clip(np.round(np.asarray(pred, dtype=float)), 1, 21)
        p_exact = (rounded == target[:, None]).mean(axis=1)
        p_pm1 = (np.abs(rounded - target[:, None]) <= 1.0).mean(axis=1)
        return p_exact, p_pm1
    except Exception:  # noqa: BLE001
        return empty, empty


def attach_observed_ratings(result: pd.DataFrame, firm: pd.DataFrame) -> pd.DataFrame:
    """PIT-Agentur-Note aus ratings_panel.csv an die Score-Zeilen."""
    if not RATINGS.exists() or "cik" not in firm.columns:
        return result
    try:
        panel = load_ratings(RATINGS)
        labeled = attach_rating_labels(firm, panel, target="ordinal", drop_unrated=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.DIM}Ratings-Panel nicht joinbar: {exc}{C.RESET}")
        return result
    keys = [c for c in ("cik", "fyear") if c in result.columns and c in labeled.columns]
    if not keys:
        keys = [c for c in ("cik", "reporting_date") if c in result.columns and c in labeled.columns]
    if not keys:
        return result
    extra = [c for c in ("rating_asof", "notch_asof", "rating_asof_date", "agency_asof") if c in labeled.columns]
    if not extra:
        return result
    merged = result.merge(labeled[keys + extra].drop_duplicates(keys), on=keys, how="left")
    return merged


def fmt_date(val: object) -> str:
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
        filing = fmt_date(row.get("filing_date"))
        report = fmt_date(row.get("reporting_date"))
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


def score_block_reason(row: pd.Series) -> str | None:
    """Warum diese Zeile keine Prognose tragen darf (sonst None).

    Zwei Fälle: (1) 10-K nach Insolvenzantrag — MD&A kennt Chapter 11.
    (2) Keine XBRL-/Panel-Finanzdaten — Median-Imputation wäre eine Pseudo-PD
    (GM 2008, Delta 2004).
    """
    post = False
    if "post_petition" in row.index:
        flag = row.get("post_petition")
        if pd.notna(flag) and bool(flag):
            post = True
    elif "bankruptcy_date" in row.index and "filing_date" in row.index:
        bk = pd.to_datetime(row.get("bankruptcy_date"), errors="coerce")
        fil = pd.to_datetime(row.get("filing_date"), errors="coerce")
        if pd.notna(bk) and pd.notna(fil) and fil > bk:
            post = True
    if post:
        return "10-K nach Insolvenzantrag — MD&A enthält Chapter-11-Prosa"

    has_fin: bool | None = None
    if "has_financials" in row.index and pd.notna(row.get("has_financials")):
        has_fin = bool(row["has_financials"])
    elif "total_assets" in row.index:
        has_fin = pd.notna(row.get("total_assets"))
    if has_fin is False:
        extra = ""
        fy = row.get("fyear") if "fyear" in row.index else None
        try:
            if pd.notna(fy) and int(fy) < MIN_FYEAR_WITH_FINANCIALS:
                extra = f", XBRL erst ab GJ {MIN_FYEAR_WITH_FINANCIALS} flächig"
        except (TypeError, ValueError):
            pass
        return f"keine Finanzdaten — Median-Imputation wäre Pseudo-PD{extra}"
    return None


def print_unscoreable(year_s: str, reason: str, *, filing: str, report: str) -> None:
    print(f"  {C.BOLD}Nicht scorebar · GJ {year_s}{C.RESET}")
    print(f"  {C.DIM}Filing {filing}  ·  Bilanzstichtag {report}{C.RESET}")
    print(f"    {C.YELLOW}{reason}{C.RESET}")
    print(
        f"  {C.DIM}Kein PD-/Rating-Output — die Zeile ist kein gültiger "
        f"Prognosepunkt.{C.RESET}"
    )


def print_pd_forecast(
    *,
    score_model: float,
    pd_model_horizon: int,
    horizon: int,
    base_rate_12m: float,
    year_s: str,
    filing: str,
    report: str,
    report_ts,
    title: str,
    show_filing_line: bool = True,
    show_proxy_notes: bool = True,
    rating_notch: float | None = None,
    rating_p_exact: float | None = None,
    rating_p_pm1: float | None = None,
) -> None:
    """PD für den gewählten Horizont inkl. Termstruktur-Tabelle."""
    score = scale_pd(score_model, from_months=pd_model_horizon, to_months=horizon)
    scaled = horizon != pd_model_horizon
    base_h = scale_pd(base_rate_12m, from_months=12, to_months=horizon)
    lift = score / base_h if base_h > 0 else float("nan")
    print(f"  {C.BOLD}{title}{C.RESET}")
    if show_filing_line:
        print(
            f"  {C.DIM}Basis: 10-K GJ {year_s}  ·  Filing {filing}  ·  "
            f"Bilanzstichtag {report}{C.RESET}"
        )
    print(
        f"    {fmt_score(score, label_source='default', horizon_months=horizon, base_rate_12m=base_rate_12m)}"
    )
    if pd.notna(report_ts):
        end = report_ts + pd.DateOffset(months=horizon)
        print(
            f"  {C.DIM}Fenster: ({report}, {fmt_date(end)}] — "
            f"P(Insolvenz-Meldung in diesem Zeitraum).{C.RESET}"
        )
    print(
        f"  {C.DIM}Vergleich: Sample-Basisrate ≈ {100 * base_h:.2f} % "
        f"über {horizon_label(horizon)}  ·  Lift {lift:.2f}×{C.RESET}"
    )
    if scaled:
        print(
            f"  {C.YELLOW}Abgeleitet aus {pd_model_horizon}-M-Modell-PD "
            f"({100 * score_model:.2f} %) unter konstanter Hazard-Rate "
            f"— nicht separat kalibriert.{C.RESET}"
        )
    else:
        print(f"  {C.DIM}Direktes Modell-Output (trainiert auf {pd_model_horizon} M).{C.RESET}")

    grid = sorted({12, 24, 36, 60, 120, pd_model_horizon, horizon})
    has_rating_col = rating_notch is not None and pd.notna(rating_notch)
    rating_letter = (
        notch_to_letter(rating_notch, scale=_RATING_SCALE) if has_rating_col else ""
    )
    p_note = (
        float(rating_p_exact)
        if rating_p_exact is not None and pd.notna(rating_p_exact)
        else float("nan")
    )
    print()
    print(f"  {C.BOLD}Vorausschau (Termstruktur){C.RESET}")
    if has_rating_col:
        print(
            f"  {'Horizont':<14} {'von':<12} {'bis':<12} {'PD':>7}  "
            f"{'vs Basis':<16} {'Rating(Moody-Skala)':<20} {'P(Note)':>8}"
        )
        print(f"  {C.DIM}{'-' * 100}{C.RESET}")
    else:
        print(f"  {'Horizont':<14} {'von':<12} {'bis':<12} {'PD':>7}  {'vs Basis':<16}")
        print(f"  {C.DIM}{'-' * 66}{C.RESET}")
    for m in grid:
        pd_m = scale_pd(score_model, from_months=pd_model_horizon, to_months=m)
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
        mark = " ←" if m == horizon else (" ◇" if m == pd_model_horizon else "")
        line = (
            f"  {horizon_label(m):<14} {von:<12} {bis:<12} "
            f"{color}{100 * pd_m:6.2f}%{C.RESET}  {color}{band:<16}{C.RESET}"
        )
        if has_rating_col:
            p_s = f"{100 * p_note:6.0f}%" if p_note == p_note else "     —"
            line += f" {C.CYAN}{rating_letter:<20}{C.RESET} {p_s}"
        print(line + f"{C.DIM}{mark}{C.RESET}")
    print(
        f"  {C.DIM}← gewählt · ◇ Modellhorizont · "
        f"Fenster = (Bilanzstichtag, Stichtag + Horizont]{C.RESET}"
    )
    if has_rating_col:
        pm1 = (
            float(rating_p_pm1)
            if rating_p_pm1 is not None and pd.notna(rating_p_pm1)
            else float("nan")
        )
        pm1_s = f", ±1 Notch {100 * pm1:.0f} %" if pm1 == pm1 else ""
        print(
            f"  {C.DIM}Rating(Moody-Skala) = PIT-Shadow-Rating (Aaa … C, unabhängig "
            f"vom PD-Horizont). P(Note) = Anteil der RF-Bäume mit genau "
            f"{rating_letter}{pm1_s}.{C.RESET}"
        )
    if show_proxy_notes:
        print()
        print(
            f"  {C.DIM}Proxy: 8-K Item 1.03 / alt. Item 3 (Chapter 11) — "
            f"keine regulatorische PD.{C.RESET}"
        )
        print(
            f"  {C.DIM}Für echte multi-Jahr-Labels: Einstellungen → Training "
            f"mit --default-horizon {horizon}.{C.RESET}"
        )


def _rating_forecast_kwargs(focus: pd.Series) -> dict[str, float]:
    """Rating-Spalten für die Termstruktur, falls das Modell eine Note liefert."""
    if "rating_notch" not in focus.index or not pd.notna(focus.get("rating_notch")):
        return {}
    out: dict[str, float] = {"rating_notch": float(focus["rating_notch"])}
    if "rating_p_exact" in focus.index and pd.notna(focus.get("rating_p_exact")):
        out["rating_p_exact"] = float(focus["rating_p_exact"])
    if "rating_p_pm1" in focus.index and pd.notna(focus.get("rating_p_pm1")):
        out["rating_p_pm1"] = float(focus["rating_p_pm1"])
    return out


def print_score_table(
    result: pd.DataFrame,
    company_name: str,
    *,
    focus_doc_id: str | None = None,
    focus_fyear: int | None = None,
    meta: dict | None = None,
    forecast_horizon_months: int | None = None,
    pd_meta: dict | None = None,
) -> None:
    meta = meta or {}
    pd_meta = pd_meta or {}
    label_source = str(meta.get("label_source") or "fraud")
    pd_model_horizon = int(
        pd_meta.get("default_horizon_months")
        or meta.get("default_horizon_months")
        or 12
    )
    horizon = int(forecast_horizon_months or pd_model_horizon)
    br_src = pd_meta if pd_meta.get("metrics") else meta
    base_rate_12m = base_rate_12m_from_meta(br_src, pd_model_horizon)

    hr()
    print(f"  {C.BOLD}Ergebnis · {company_name}{C.RESET}")
    hr()
    if result.empty:
        print(f"  {C.YELLOW}Keine Scores.{C.RESET}")
        return

    focus = result.iloc[-1]
    if focus_fyear is not None and "fyear" in result.columns:
        fy = pd.to_numeric(result["fyear"], errors="coerce")
        hit = result[fy == int(focus_fyear)]
        if not hit.empty:
            focus = hit.iloc[0]
    elif focus_doc_id and "doc_id" in result.columns:
        hit = result[result["doc_id"].astype(str) == str(focus_doc_id)]
        if not hit.empty:
            focus = hit.iloc[0]

    has_rating = "rating_notch" in focus.index and pd.notna(focus.get("rating_notch"))
    pd_raw = focus.get("pd_score") if "pd_score" in focus.index else float("nan")
    try:
        score_model = float(pd_raw)
    except (TypeError, ValueError):
        score_model = float("nan")
    has_pd = score_model == score_model
    year = focus.get("fyear", "?")
    year_s = str(int(year)) if pd.notna(year) and str(year) != "?" else "?"
    filing = fmt_date(focus.get("filing_date")) if "filing_date" in focus.index else "—"
    report = fmt_date(focus.get("reporting_date")) if "reporting_date" in focus.index else "—"
    report_ts = (
        pd.to_datetime(focus.get("reporting_date"), errors="coerce")
        if "reporting_date" in focus.index
        else pd.NaT
    )

    print()
    block = score_block_reason(focus)
    if block:
        print_unscoreable(year_s, block, filing=filing, report=report)
        if has_rating or label_source == "rating":
            obs_n = focus.get("notch_asof") if "notch_asof" in focus.index else pd.NA
            obs_letter = focus.get("rating_asof") if "rating_asof" in focus.index else pd.NA
            if pd.notna(obs_n):
                obs_s = f"{obs_letter or notch_to_letter(obs_n)} (Notch {int(round(float(obs_n)))}/21)"
                print(f"    Rating (Agentur)   {obs_s}")
    elif label_source == "default":
        print_pd_forecast(
            score_model=score_model,
            pd_model_horizon=pd_model_horizon,
            horizon=horizon,
            base_rate_12m=base_rate_12m,
            year_s=year_s,
            filing=filing,
            report=report,
            report_ts=report_ts,
            title=f"Ausfallwahrscheinlichkeit · Horizont {horizon_label(horizon)}",
            **_rating_forecast_kwargs(focus),
        )
    elif has_rating or label_source == "rating":
        pred_n = focus.get("rating_notch")
        pred_s = format_notch(pred_n, scale=_RATING_SCALE) if pd.notna(pred_n) else "—"
        p_exact = focus.get("rating_p_exact") if "rating_p_exact" in focus.index else float("nan")
        p_pm1 = focus.get("rating_p_pm1") if "rating_p_pm1" in focus.index else float("nan")
        conf_s = ""
        if pd.notna(p_exact):
            conf_s = f"  {C.DIM}P(Note) {100 * float(p_exact):.0f} %"
            if pd.notna(p_pm1):
                conf_s += f"  ·  ±1 Notch {100 * float(p_pm1):.0f} %"
            conf_s += f"{C.RESET}"
        obs_n = focus.get("notch_asof") if "notch_asof" in focus.index else pd.NA
        obs_letter = focus.get("rating_asof") if "rating_asof" in focus.index else pd.NA
        if pd.notna(obs_n):
            obs_s = f"{obs_letter or notch_to_letter(obs_n)} (Notch {int(round(float(obs_n)))}/21)"
        else:
            obs_s = "—"
        ag = focus.get("agency_asof") if "agency_asof" in focus.index else ""
        ag_s = f" · {ag}" if isinstance(ag, str) and ag and ag != "nan" else ""
        asof_d = (
            fmt_date(focus.get("rating_asof_date"))
            if "rating_asof_date" in focus.index
            else "—"
        )
        print(f"  {C.BOLD}Unternehmensrating · GJ {year_s}{C.RESET}")
        print(f"  {C.DIM}Filing {filing}  ·  Bilanzstichtag {report}{C.RESET}")
        print(f"    Rating (Modell)    {C.CYAN}{pred_s}{C.RESET}{conf_s}")
        print(f"    Rating (Agentur)   {obs_s}{C.DIM}{ag_s}  as-of {asof_d}{C.RESET}")
        if has_pd:
            print()
            print_pd_forecast(
                score_model=score_model,
                pd_model_horizon=pd_model_horizon,
                horizon=horizon,
                base_rate_12m=base_rate_12m,
                year_s=year_s,
                filing=filing,
                report=report,
                report_ts=report_ts,
                title=f"PD · Horizont {horizon_label(horizon)} (sekundär)",
                show_filing_line=False,
                show_proxy_notes=False,
                **_rating_forecast_kwargs(focus),
            )
        print()
        print(
            f"  {C.DIM}Shadow-Rating auf Moody's-Skala (Aaa=21 … C=1). "
            f"Keine regulatorische Note.{C.RESET}"
        )
    else:
        print(f"  {C.BOLD}Misconduct-/Fraud-Score{C.RESET} · GJ {year_s}")
        print(f"    {fmt_score(score_model, label_source=label_source)}")
        print()
        print(f"  {C.DIM}Interpretation: AAER-Risiko, keine regulatorische PD.{C.RESET}")

    print()
    if has_rating or label_source == "rating":
        pd_hdr = f"PD%{horizon}M"
        print(
            f"  {'Jahr':<6} {'Modell':<22} {'P(Note)':>7}  {'Agentur':<22} "
            f"{pd_hdr:>8}  Verlauf"
        )
        print(f"  {C.DIM}{'-' * 80}{C.RESET}")
        for _, row in result.iterrows():
            y = int(row["fyear"]) if "fyear" in row.index and pd.notna(row["fyear"]) else "?"
            pred_cell = (
                format_notch(row.get("rating_notch"), scale=_RATING_SCALE)
                if "rating_notch" in row.index
                else "—"
            )
            p_cell = "     —"
            if "rating_p_exact" in row.index and pd.notna(row.get("rating_p_exact")):
                p_cell = f"{100 * float(row['rating_p_exact']):6.0f}%"
            row_block = score_block_reason(row)
            if row_block:
                pred_cell = "nicht scorebar"
                p_cell = "     —"
            if "notch_asof" in row.index and pd.notna(row.get("notch_asof")):
                obs_letter = row.get("rating_asof") if "rating_asof" in row.index else ""
                obs_cell = f"{obs_letter or notch_to_letter(row['notch_asof'])} ({int(round(float(row['notch_asof'])))}/21)"
            else:
                obs_cell = "—"
            pd_cell = "     —"
            spark_src = 0.5
            if row_block:
                pd_cell = " n/score"
            elif "pd_score" in row.index and pd.notna(row.get("pd_score")):
                try:
                    p_h = scale_pd(
                        float(row["pd_score"]),
                        from_months=pd_model_horizon,
                        to_months=horizon,
                    )
                    pd_cell = f"{100 * p_h:7.2f}%"
                    spark_src = min(max(p_h / 0.05, 0.0), 1.0)
                except (TypeError, ValueError):
                    pass
            elif "rating_notch" in row.index and pd.notna(row.get("rating_notch")):
                spark_src = 1.0 - (float(row["rating_notch"]) - 1.0) / 20.0
            spark = "▂▃▄▅▆"[min(4, int(spark_src * 5))]
            row_year = (
                int(row["fyear"])
                if "fyear" in row.index and pd.notna(row["fyear"])
                else None
            )
            if focus_fyear is not None and row_year == int(focus_fyear):
                marker = " ←"
            elif focus_doc_id and str(row.get("doc_id")) == str(focus_doc_id):
                marker = " ←"
            else:
                marker = ""
            print(
                f"  {y:<6} {pred_cell:<22} {p_cell}  {obs_cell:<22} {pd_cell}  "
                f"{spark}{C.DIM}{marker}{C.RESET}"
            )
        if "rating_notch" in result.columns:
            ok = result[[score_block_reason(r) is None for _, r in result.iterrows()]]
            rn = pd.to_numeric(ok["rating_notch"], errors="coerce") if not ok.empty else pd.Series(dtype=float)
            print()
            if rn.notna().any():
                print(
                    f"  Modell-Notch  Mittel {rn.mean():.2f}  ·  "
                    f"Min {rn.min():.1f}  ·  Max {rn.max():.1f}  "
                    f"{C.DIM}(nur scorebare Jahre){C.RESET}"
                )
            print(
                f"  {C.DIM}nicht scorebar = keine XBRL-Finanzdaten oder 10-K nach Antrag.{C.RESET}"
            )
        return

    label_hdr = "Default" if label_source == "default" else "AAER"
    score_hdr = f"PD%{horizon}M" if label_source == "default" else "Score"
    window_hdr = "Fenster" if label_source == "default" else ""
    print(
        f"  {'Jahr':<6} {score_hdr:>8}  {'Risiko':<16}  {label_hdr:<7}  "
        f"{'Event':<12} {window_hdr:<23} Verlauf"
    )
    print(f"  {C.DIM}{'-' * 88}{C.RESET}")
    for _, row in result.iterrows():
        y = int(row["fyear"]) if "fyear" in row.index and pd.notna(row["fyear"]) else "?"
        if focus_fyear is not None and "fyear" in row.index and pd.notna(row["fyear"]) and int(row["fyear"]) == int(focus_fyear):
            marker = " ←"
        elif focus_doc_id and str(row.get("doc_id")) == str(focus_doc_id):
            marker = " ←"
        else:
            marker = ""
        row_block = score_block_reason(row)
        if row_block:
            print(
                f"  {y:<6} {C.YELLOW}{'n/score':>8}{C.RESET}  "
                f"{C.YELLOW}{'nicht scorebar':<16}{C.RESET}  "
                f"{'–    '}  {'—':<12} {'—':<23} "
                f"{C.DIM}{marker}{C.RESET}"
            )
            continue
        s_model = float(row["pd_score"])
        s = (
            scale_pd(s_model, from_months=pd_model_horizon, to_months=horizon)
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
            event_cell = fmt_date(row.get("bankruptcy_date")) if "bankruptcy_date" in row.index else "—"
            if label != 1:
                event_cell = "—"
            rep_row = (
                pd.to_datetime(row.get("reporting_date"), errors="coerce")
                if "reporting_date" in row.index
                else pd.NaT
            )
            if pd.notna(rep_row):
                end_row = rep_row + pd.DateOffset(months=horizon)
                window_cell = f"({fmt_date(rep_row)}, {fmt_date(end_row)}]"
            else:
                window_cell = "—"
        base_h = scale_pd(base_rate_12m, from_months=12, to_months=horizon)
        scale = max(3.0 * base_h, 0.02) if label_source == "default" else 1.0
        spark = "▂▃▄▅▆"[min(4, int(min(s / scale, 1.0) * 5))]
        score_cell = f"{100 * s:7.2f}%" if label_source == "default" else f"{s:8.3f}"
        print(
            f"  {y:<6} {color}{score_cell}{C.RESET}  {color}{band:<16}{C.RESET}  "
            f"{flag}  {event_cell:<12} {window_cell:<23} "
            f"{color}{spark}{C.RESET}{C.DIM}{marker}{C.RESET}"
        )

    print()
    ok_mask = result.apply(lambda r: score_block_reason(r) is None, axis=1)
    usable = result.loc[ok_mask]
    if usable.empty or "pd_score" not in usable.columns:
        print(f"  {C.DIM}n/score = nicht scorebar (keine XBRL-Finanzdaten oder 10-K nach Antrag).{C.RESET}")
    else:
        print(
            f"  Mittel {usable['pd_score'].mean():.4f}  ·  "
            f"Min {usable['pd_score'].min():.4f}  ·  "
            f"Max {usable['pd_score'].max():.4f}  {C.DIM}(Modell-{pd_model_horizon}M, nur scorebare Jahre){C.RESET}"
        )
        print(f"  {C.DIM}n/score = nicht scorebar (keine XBRL-Finanzdaten oder 10-K nach Antrag).{C.RESET}")
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


# --------------------------------------------------------------------------- #
# Eingabe-Quellen (Label-Set / Datei / EDGAR / Fragebogen)
# --------------------------------------------------------------------------- #


def normalize_firm_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def ask_float(prompt: str) -> float | None:
    raw = ask(prompt + " (leer = unbekannt)", "")
    if not raw:
        return None
    try:
        return float(raw.replace(",", "").replace(" ", ""))
    except ValueError:
        print(f"  {C.YELLOW}Ungültige Zahl — übersprungen.{C.RESET}")
        return None


def read_mda_input() -> str:
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


def choose_model_path(*, has_text: bool) -> Path:
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
        rating_target="ordinal" if label == "rating" else None,
        has_text=has_text,
        require=label in {"default", "rating"},
    )
    if path is None:
        path = select_model_path(
            prefer="combined" if has_text else "financial",
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
        for m in (12, 24, 36, 60, 120):
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
    firm = normalize_firm_frame(firm)
    if firm.empty:
        print(f"  {C.RED}Keine Daten zum Scoren.{C.RESET}")
        pause()
        return

    focus_row = None
    focus_doc = None
    focus_fyear = None
    if len(firm) > 1 and "fyear" in firm.columns:
        focus_row = pick_tenk(firm)
        if focus_row is None:
            pause()
            return
        # Alle Jahre scoren (Verlauf), Fokus auf Auswahl
        focus_doc = str(focus_row.get("doc_id", ""))
        if pd.notna(focus_row.get("fyear")):
            focus_fyear = int(focus_row["fyear"])
    else:
        focus_doc = str(firm.iloc[0].get("doc_id", "")) if len(firm) else None
        if len(firm) and pd.notna(firm.iloc[0].get("fyear")):
            focus_fyear = int(firm.iloc[0]["fyear"])

    has_text = "mda" in firm.columns and firm["mda"].fillna("").astype(str).str.len().max() >= 50
    try:
        model_path = choose_model_path(has_text=bool(has_text))
    except FileNotFoundError as exc:
        print(f"  {C.RED}{exc}{C.RESET}")
        pause()
        return

    meta = dict(load_any(model_path).get("metadata") or {})
    label_source = str(meta.get("label_source") or "fraud")
    model_horizon = int(meta.get("default_horizon_months") or 12)
    pd_model_horizon = model_horizon
    pd_meta: dict[str, Any] = {}
    pd_path: Path | None = None

    want_pd = label_source == "default"
    if label_source == "rating" or meta.get("task") == "regression":
        probe, _ = select_default_pd_path(has_text=bool(has_text), want_horizon=12)
        want_pd = probe is not None

    forecast_h = model_horizon if label_source == "default" else 12
    if want_pd:
        forecast_h = ask_forecast_horizon(forecast_h)

    print(f"  Modell: {C.CYAN}{describe_model(model_path)}{C.RESET}")
    print(
        f"  {C.DIM}Vorausschau={forecast_h}M "
        f"(label={meta.get('label_source', '?')}"
        f"{f', Modellhorizont={model_horizon}M' if label_source == 'default' else ''})"
        f"{C.RESET}"
    )
    warn_model_coherence()
    try:
        result = score_frame(firm, model_path, events_df=events_df)
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.RED}Scoring fehlgeschlagen: {exc}{C.RESET}")
        pause()
        return

    if label_source == "rating" or meta.get("task") == "regression":
        pd_path, pd_meta = select_default_pd_path(
            has_text=bool(has_text),
            want_horizon=forecast_h,
        )
        if pd_path is not None and pd_path.resolve() != Path(model_path).resolve():
            try:
                pd_result = score_frame(firm, pd_path, events_df=events_df)
                keys = [c for c in ("doc_id", "fyear") if c in result.columns and c in pd_result.columns]
                if keys:
                    keep = keys + [
                        c
                        for c in (
                            "pd_score",
                            "label_default",
                            "bankruptcy_date",
                            "post_petition",
                            "has_financials",
                        )
                        if c in pd_result.columns
                    ]
                    drop_overlap = [c for c in keep if c not in keys and c in result.columns]
                    result = result.drop(columns=["pd_score", *drop_overlap], errors="ignore").merge(
                        pd_result[keep].drop_duplicates(keys),
                        on=keys,
                        how="left",
                    )
                pd_model_horizon = int(pd_meta.get("default_horizon_months") or 12)
                native = "nativ" if pd_model_horizon == forecast_h else f"skaliert von {pd_model_horizon}M"
                print(f"  {C.DIM}PD-Modell: {pd_path.name} ({native}){C.RESET}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {C.YELLOW}PD-Modell nicht anwendbar: {exc}{C.RESET}")
                pd_path = None
                pd_meta = {}
    result = attach_observed_ratings(result, firm)

    print_score_table(
        result,
        company_name,
        focus_doc_id=focus_doc,
        focus_fyear=focus_fyear,
        meta=meta,
        forecast_horizon_months=forecast_h,
        pd_meta=pd_meta or None,
    )
    print()
    maybe_save_csv(
        result,
        default_csv or "scores_last.csv",
        model_horizon=pd_model_horizon,
        forecast_horizon=forecast_h if want_pd and "pd_score" in result.columns else None,
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
        print(f"  {C.YELLOW}Keine Treffer im Label-Set.{C.RESET}")
        print(f"  {C.DIM}Neue Emittenten über Menü 3 · EDGAR live (CIK).{C.RESET}")
        return None

    assert chosen_cik is not None
    print(f"\n  {C.BOLD}Lade{C.RESET} {chosen_name} (CIK {chosen_cik}) …")
    firm = load_firm_years_for_cik(chosen_cik)
    if firm.empty:
        print(f"  {C.RED}Keine Firm-Years im Label-Set.{C.RESET}")
        print(
            f"  {C.DIM}Suche läuft nur über die historischen 10-Ks (Zenodo). "
            f"Neue Emittenten: zurück und Menü 3 · EDGAR live.{C.RESET}"
        )
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
        row = questionnaire_core(mda_default=mda)
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

    firm = normalize_firm_frame(firm)
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
    ua = ensure_sec_ua()
    if ua is None:
        return None

    name = ask("Anzeigename (optional)", f"CIK {cik}")
    print(f"  {C.DIM}Lade Financials …{C.RESET}")
    try:
        panel = build_financials_panel([cik], user_agent=ua)
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.RED}Financials fehlgeschlagen: {exc}{C.RESET}")
        return None
    if panel.empty or "cik" not in panel.columns:
        print(f"  {C.RED}Keine 10-K/10-Q-Werte in companyfacts für CIK {cik}.{C.RESET}")
        print(
            f"  {C.DIM}Häufig: privater Emittent, oder Neu-Listing vor dem ersten "
            f"10-K (SpaceX 2026: zunächst nur 10-Q).{C.RESET}"
        )
        return None
    if "entity_name" in panel.columns:
        ent = panel["entity_name"].dropna().astype(str)
        if not ent.empty and (not name or name == f"CIK {cik}"):
            name = str(ent.iloc[-1])
    n_q = int((panel.get("source_form") == "10-Q").sum()) if "source_form" in panel.columns else 0
    n_k = int((panel.get("source_form") == "10-K").sum()) if "source_form" in panel.columns else 0
    print(f"  {C.GREEN}{len(panel)} Firm-Years aus EDGAR.{C.RESET}  {name}")
    if n_q and not n_k:
        print(
            f"  {C.YELLOW}Kein 10-K — 10-Q als Zwischenabschluss "
            f"(GuV oft YTD, nicht 12 Monate).{C.RESET}"
        )

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
    if not local.empty:
        keep = ["cik", "fyear", "mda", "filing_date", "reporting_date", "doc_id", "label", "sic"]
        keep = [c for c in keep if c in local.columns]
        firm = firm.merge(local[keep], on=["cik", "fyear"], how="left", suffixes=("", "_loc"))
        for col in ("filing_date", "reporting_date", "doc_id", "mda", "label", "sic"):
            loc = f"{col}_loc"
            if loc in firm.columns:
                firm[col] = firm[col].fillna(firm[loc]) if col in firm.columns else firm[loc]
                firm = firm.drop(columns=[loc])
        if "mda" in firm.columns:
            n_txt = int(firm["mda"].fillna("").astype(str).str.len().gt(50).sum())
            print(f"  {C.DIM}MD&A aus lokalem Datensatz für {n_txt}/{len(firm)} Jahre.{C.RESET}")
        # SIC ist firmenspezifisch — fehlende EDGAR-Jahre vom lokalen Datensatz auffüllen
        if "sic" in local.columns:
            sic_val = pd.to_numeric(local["sic"], errors="coerce").dropna()
            if not sic_val.empty:
                if "sic" not in firm.columns or firm["sic"].isna().all():
                    firm["sic"] = sic_val.iloc[-1]
                else:
                    firm["sic"] = pd.to_numeric(firm["sic"], errors="coerce").fillna(sic_val.iloc[-1])

    if "mda" not in firm.columns or firm["mda"].fillna("").astype(str).str.len().max() < 50:
        print(f"  {C.YELLOW}Kein MD&A online in companyfacts — optional nachreichen.{C.RESET}")
        if ask("MD&A jetzt angeben? (j/n)", "n").lower().startswith("j"):
            mda = read_mda_input()
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


def questionnaire_core(*, mda_default: str = "") -> dict | None:
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
        val = ask_float(label)
        if val is not None:
            row[key] = val

    print()
    if mda_default:
        row["mda"] = mda_default
        print(f"  {C.DIM}MD&A bereits gesetzt ({len(mda_default):,} Zeichen).{C.RESET}")
    else:
        row["mda"] = read_mda_input()
    return row


def input_from_questionnaire() -> tuple[pd.DataFrame, str, pd.DataFrame | None] | None:
    print(f"  {C.BOLD}Fragebogen{C.RESET}")
    print(f"  {C.DIM}Ein 10-K manuell erfassen → Rating + PD (Horizont danach wählbar).{C.RESET}")
    print()
    row = questionnaire_core()
    if row is None:
        return None

    events_df = None
    cik = int(row.get("cik") or 0)
    if cik and ask("8-K-Events für diese CIK von EDGAR laden? (j/n)", "n").lower().startswith("j"):
        ua = ensure_sec_ua()
        if ua:
            try:
                events_df = build_events_table([cik], user_agent=ua, cache_dir=SUBMISSIONS_CACHE)
                print(f"  {C.GREEN}{len(events_df)} 8-Ks geladen.{C.RESET}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {C.YELLOW}Events fehlgeschlagen: {exc}{C.RESET}")

    firm = pd.DataFrame([row])
    return firm, str(row.get("name") or "Manuell"), events_df


# --------------------------------------------------------------------------- #
# Scoring-Menü
# --------------------------------------------------------------------------- #


def flow_score_company(index: pd.DataFrame) -> None:
    clear()
    banner()
    meta = active_model_meta()
    horizon = state.FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
    if meta.get("label_source") == "default":
        print(f"  {C.BOLD}10-K scoren → Ausfall-PD{C.RESET}  "
              f"{C.DIM}(Vorausschau {horizon_label(horizon)}){C.RESET}")
    elif meta.get("label_source") == "rating":
        print(f"  {C.BOLD}10-K scoren → Rating + PD{C.RESET}  "
              f"{C.DIM}(PD-Vorausschau {horizon_label(horizon)}){C.RESET}")
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
