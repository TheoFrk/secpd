"""Tests für 8-K-Events: Default-Label, Item-Regime, PIT-Features."""
from __future__ import annotations

import pandas as pd
import pytest

from secpd.data.events import (
    REGIME_SWITCH,
    add_event_features,
    annotate_default_labels,
    attach_default_labels,
    bankruptcy_dates,
    match_concept,
)
from secpd.data.synthetic import make_synthetic_dataset
from secpd.models.pipeline import build_pipeline
from secpd.splitting import smart_split


def _events(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["cik", "filing_date_8k", "items"])
    df["filing_date_8k"] = pd.to_datetime(df["filing_date_8k"])
    df["accession"] = ""
    return df


def _firm_years(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["cik", "reporting_date", "filing_date"])
    df["doc_id"] = df["cik"].astype(str) + "_" + df.index.astype(str)
    return df


# --------------------------------------------------------------------------- #
# 1) Label-Horizont: Grenze inklusiv, ein Tag darüber ⇒ 0
# --------------------------------------------------------------------------- #
def test_default_label_horizon_boundary():
    fy = _firm_years([(1, "2010-12-31", "2011-03-31"), (2, "2010-12-31", "2011-03-31")])
    ev = _events([
        (1, "2011-12-31", "1.03"),   # exakt reporting_date + 12M ⇒ Label 1
        (2, "2012-01-01", "1.03"),   # ein Tag darüber ⇒ Label 0
        (9, "2030-01-01", "9.01"),   # spätes Dummy-Event ⇒ keine Zensierung der Testfälle
    ])
    out = attach_default_labels(fy, ev, horizon_months=12)
    got = out.set_index("cik")["label_default"].to_dict()
    assert got == {1: 1, 2: 0}


# --------------------------------------------------------------------------- #
# 2) Rechtszensierung: Label-0 nahe global_max fliegt, Label-1 bleibt
# --------------------------------------------------------------------------- #
def test_censoring_drops_uncovered_zeros_keeps_ones():
    fy = _firm_years([
        (1, "2019-12-31", "2020-03-31"),   # Horizont bis 2020-12-31 > global_max ⇒ drop
        (2, "2019-12-31", "2020-03-31"),   # Bankruptcy im Horizont ⇒ bleibt (Label 1)
        (3, "2018-12-31", "2019-03-31"),   # Horizont bis 2019-12-31 ≤ global_max ⇒ bleibt (0)
    ])
    ev = _events([(2, "2020-05-01", "1.03"), (9, "2020-06-30", "9.01")])
    out = attach_default_labels(fy, ev, horizon_months=12, drop_censored=True)
    got = out.set_index("cik")["label_default"].to_dict()
    assert got == {2: 1, 3: 0}


# --------------------------------------------------------------------------- #
# 3) Post-Bankruptcy-Firm-Years werden gedroppt
# --------------------------------------------------------------------------- #
def test_post_bankruptcy_rows_dropped():
    fy = _firm_years([
        (1, "2009-12-31", "2010-03-31"),   # vor Bankruptcy ⇒ Label 1
        (1, "2010-12-31", "2011-03-31"),   # reporting_date ≥ bankruptcy ⇒ drop
    ])
    ev = _events([(1, "2010-06-01", "1.03"), (9, "2030-01-01", "9.01")])
    out = attach_default_labels(fy, ev, horizon_months=12)
    assert len(out) == 1
    assert out.iloc[0]["label_default"] == 1


# --------------------------------------------------------------------------- #
# 4) Item-Regime: match_concept kennt beide Regime; Labels default nur neu
# --------------------------------------------------------------------------- #
def test_item_regime_matching():
    before = REGIME_SWITCH - pd.Timedelta(days=1)
    after = REGIME_SWITCH
    assert match_concept("3,7", before, "bankruptcy") is True
    assert match_concept("3", after, "bankruptcy") is False          # neues Regime: "3" ≠ 1.03
    assert match_concept("3.01", before, "bankruptcy") is False      # kein Substring-Match
    assert match_concept("3.01,9.01", after, "bankruptcy") is False
    assert match_concept("1.03,9.01", after, "bankruptcy") is True
    assert match_concept("1.03", before, "bankruptcy") is False      # altes Regime: kein 1.03
    # Default-Policy: Legacy-Item-3 zählt nicht als Bankruptcy-Datum
    ev = _events([(1, "2001-12-02", "3"), (1, "2010-01-01", "1.03")])
    bk = bankruptcy_dates(ev)
    assert bk.loc[1] == pd.Timestamp("2010-01-01")
    # Audit/Override: Legacy weiterhin einschaltbar
    bk_legacy = bankruptcy_dates(ev, trust_legacy_regime=True)
    assert bk_legacy.loc[1] == pd.Timestamp("2001-12-02")


def test_legacy_bankruptcy_ignored_in_labels():
    fy = _firm_years([(1, "1999-12-31", "2000-03-31")])
    ev = _events([
        (1, "2000-06-01", "3"),          # Alt-Regime — Standard: ignorieren
        (9, "2030-01-01", "9.01"),
    ])
    out = attach_default_labels(fy, ev, horizon_months=12, drop_censored=False)
    assert out.iloc[0]["label_default"] == 0
    out_legacy = attach_default_labels(
        fy, ev, horizon_months=12, drop_censored=False, trust_legacy_regime=True
    )
    assert out_legacy.iloc[0]["label_default"] == 1


def test_legacy_event_features_masked():
    fy = _firm_years([(1, "2000-12-31", "2001-03-31")])
    ev = _events([(1, "2001-01-15", "6")])  # Alt-Regime Officer
    out, _ = add_event_features(fy, ev, window_days=365)
    assert out.iloc[0]["evt_n_officer_departure"] == 0
    out_legacy, _ = add_event_features(fy, ev, window_days=365, trust_legacy_regime=True)
    assert out_legacy.iloc[0]["evt_n_officer_departure"] == 1


def test_annotate_default_labels_keeps_rows():
    fy = _firm_years([
        (1, "2009-12-31", "2010-03-31"),
        (1, "2010-12-31", "2011-03-31"),
    ])
    ev = _events([(1, "2010-06-01", "1.03"), (9, "2030-01-01", "9.01")])
    ann = annotate_default_labels(fy, ev, horizon_months=12)
    assert len(ann) == 2
    assert list(ann["label_default"]) == [1, 0]
    assert pd.Timestamp(ann.iloc[0]["bankruptcy_date"]) == pd.Timestamp("2010-06-01")


# --------------------------------------------------------------------------- #
# 5) PIT: 8-K nach dem 10-K-Filing zählt in keinem Feature
# --------------------------------------------------------------------------- #
def test_pit_window_excludes_future_events():
    fy = _firm_years([(1, "2010-12-31", "2011-03-31")])
    ev = _events([
        (1, "2011-03-31", "4.01"),   # exakt am Filing-Tag ⇒ zählt
        (1, "2011-04-01", "4.01"),   # danach ⇒ Look-ahead, zählt nicht
        (1, "2010-05-01", "5.02"),   # innerhalb 365 Tage ⇒ zählt
        (1, "2010-03-01", "2.04"),   # älter als Fenster ⇒ zählt nicht
        (1, "2010-06-01", "1.03"),   # Bankruptcy ⇒ zählt NUR in evt_n_8k, kein eigenes Feature
    ])
    out, cols = add_event_features(fy, ev, window_days=365)
    row = out.iloc[0]
    assert row["evt_n_8k"] == 3                      # 4.01(am Tag) + 5.02 + 1.03
    assert row["evt_n_auditor_change"] == 1
    assert row["evt_n_officer_departure"] == 1
    assert row["evt_n_covenant_accel"] == 0
    assert not any("bankruptcy" in c for c in cols)  # Leakage-Regel


def test_delisting_label_is_not_a_feature():
    fy = _firm_years([(1, "2010-12-31", "2011-03-31")])
    ev = _events([(1, "2011-06-01", "3.01"), (9, "2030-01-01", "9.01")])
    out = attach_default_labels(
        fy, ev, horizon_months=12, label_concepts=("bankruptcy", "delisting")
    )
    assert int(out.iloc[0]["label_default"]) == 1
    _, cols = add_event_features(out, ev, exclude_concepts=("delisting",))
    assert "evt_n_delisting" not in cols
    assert "evt_n_auditor_change" in cols


# --------------------------------------------------------------------------- #
# 6) Mini-End-to-End: Default-Label + Event-Features + Training
# --------------------------------------------------------------------------- #
def test_end_to_end_default_label_training():
    df = make_synthetic_dataset(n=300, seed=7)
    rng_ciks = df["cik"].drop_duplicates().head(30).tolist()
    ev_rows = []
    for i, cik in enumerate(rng_ciks):
        # Einige CIKs bekommen eine Insolvenz kurz nach ihrem letzten Firm-Year:
        last_rep = df.loc[df["cik"] == cik, "reporting_date"].max()
        if i % 3 == 0:
            ev_rows.append((cik, (last_rep + pd.Timedelta(days=180)).strftime("%Y-%m-%d"), "1.03"))
        ev_rows.append((cik, (last_rep - pd.Timedelta(days=100)).strftime("%Y-%m-%d"), "2.06"))
    ev_rows.append((999999, "2035-01-01", "9.01"))   # global_max weit hinten ⇒ kaum Zensierung
    ev = _events(ev_rows)

    labeled = attach_default_labels(df, ev, horizon_months=12)
    assert "label_default" in labeled.columns
    assert labeled["label_default"].sum() > 0

    feat, evt_cols = add_event_features(labeled, ev)
    from secpd.features.financial import add_financial_features
    feat, fin_cols = add_financial_features(feat)
    numeric = fin_cols + evt_cols

    tr, te, _ = smart_split(feat, label_col="label_default", group_col="cik",
                            year_col="fyear", test_size=0.25, strategy="group",
                            random_state=0)
    pipe = build_pipeline(numeric, n_estimators=50, random_state=0)
    pipe.fit(feat.iloc[tr], feat.iloc[tr]["label_default"].astype(int))
    probs = pipe.predict_proba(feat.iloc[te])[:, 1]
    assert probs.shape[0] == len(te)
    assert 0.0 <= probs.min() and probs.max() <= 1.0
