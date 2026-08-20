"""Tests für Rating-Notches, PIT-Join und Label-Konstruktion."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from secpd.data.ratings import (
    attach_rating_labels,
    fmp_symbol,
    is_speculative,
    normalize_ratings_panel,
    notch_from_rating,
    parse_fmp_payload,
    ticker_map_from_submissions,
)
from secpd.data.synthetic import make_synthetic_dataset, make_synthetic_ratings
from secpd.models.persistence import bundle_filename, parse_bundle_name


def test_notch_scales_and_hy_cut():
    assert notch_from_rating("BBB-") == 12
    assert notch_from_rating("BB+") == 11
    assert notch_from_rating("Baa3", agency="moodys") == 12
    assert notch_from_rating("A+", agency="fmp") == 17
    assert notch_from_rating(None, agency="fmp", rating_score=5) == 16
    assert notch_from_rating("(P)Baa2", agency="moodys") == 13
    assert is_speculative(11)
    assert not is_speculative(12)
    assert fmp_symbol("BRK.B") == "BRK-B"


def test_notch_to_letter_roundtrip():
    from secpd.data.ratings import format_notch, notch_to_letter

    assert notch_to_letter(13) == "BBB"
    assert notch_to_letter(21) == "AAA"
    assert notch_to_letter(1) == "D"
    assert notch_to_letter(13, scale="fmp") == "B+"
    assert notch_to_letter(15, scale="moodys") == "A3"
    assert notch_to_letter(13, scale="moodys") == "Baa2"
    assert notch_to_letter(21, scale="moodys") == "Aaa"
    assert notch_to_letter(1, scale="moodys") == "C"
    assert "Baa2" in format_notch(13, scale="moodys")
    assert "13/21" in format_notch(13)
    # Roundtrip über S&P-Skala
    for n in range(1, 22):
        letter = notch_to_letter(n)
        back = notch_from_rating(letter)
        assert back == n or (n == 1 and back in {1})
    # Roundtrip über Moody's
    for n in range(1, 22):
        letter = notch_to_letter(n, scale="moodys")
        back = notch_from_rating(letter, agency="moodys")
        assert back == n


def test_parse_fmp_payload_to_panel():
    payload = [
        {"symbol": "AAPL", "date": "2010-12-31", "rating": "A+", "ratingScore": 5},
        {"symbol": "AAPL", "date": "2011-12-31", "rating": "B-", "ratingScore": 3},
    ]
    out = parse_fmp_payload(payload, ticker="AAPL")
    assert list(out["ticker"].unique()) == ["AAPL"]
    assert out["notch"].notna().all()
    assert int(out.loc[out["rating_date"] == "2011-12-31", "notch"].iloc[0]) == 11


def test_speculative_label_pit_asof():
    fy = pd.DataFrame(
        {
            "cik": [1, 1],
            "reporting_date": ["2011-12-31", "2012-12-31"],
            "fyear": [2011, 2012],
        }
    )
    rtg = normalize_ratings_panel(
        pd.DataFrame(
            {
                "cik": [1, 1, 1],
                "ticker": ["X", "X", "X"],
                "rating_date": ["2011-06-01", "2012-06-01", "2013-06-01"],
                "rating": ["A", "B-", "B-"],
                "agency": ["fmp", "fmp", "fmp"],
            }
        )
    )
    out = attach_rating_labels(fy, rtg, target="speculative")
    got = out.set_index("fyear")["label_rating"].to_dict()
    # 2011: as-of Juni 2011 = A → IG. 2012: as-of Juni 2012 = B- → HY.
    # Juni 2013 darf 2012 nicht vorwegnehmen.
    assert got == {2011: 0, 2012: 1}


def test_future_rating_does_not_leak_into_asof():
    fy = pd.DataFrame({"cik": [1], "reporting_date": ["2010-12-31"]})
    rtg = normalize_ratings_panel(
        pd.DataFrame(
            {
                "cik": [1],
                "ticker": ["X"],
                "rating_date": ["2011-01-15"],
                "rating": ["C"],
                "agency": ["fmp"],
            }
        )
    )
    out = attach_rating_labels(fy, rtg, target="speculative", drop_unrated=True)
    assert out.empty  # kein Rating ≤ reporting_date


def test_downgrade_horizon_boundary():
    fy = pd.DataFrame(
        {
            "cik": [1, 2],
            "reporting_date": ["2010-12-31", "2010-12-31"],
        }
    )
    rtg = normalize_ratings_panel(
        pd.DataFrame(
            {
                "cik": [1, 1, 2, 2, 1, 2],
                "ticker": ["A", "A", "B", "B", "A", "B"],
                "rating_date": [
                    "2010-06-01",
                    "2011-12-31",  # exakt +12M → zählt
                    "2010-06-01",
                    "2012-01-01",  # ein Tag über Horizont → 0
                    "2030-01-01",
                    "2030-01-01",
                ],
                "rating": ["A", "B-", "A", "B-", "A", "A"],
                "agency": ["fmp"] * 6,
            }
        )
    )
    out = attach_rating_labels(fy, rtg, target="downgrade", horizon_months=12, drop_censored=False)
    got = out.set_index("cik")["label_rating"].to_dict()
    assert got == {1: 1, 2: 0}


def test_unrated_dropped_and_synthetic_roundtrip():
    df = make_synthetic_dataset(n=80, seed=3)
    ratings = make_synthetic_ratings(df, seed=3)
    out = attach_rating_labels(df, ratings, target="speculative")
    assert out["label_rating"].isin([0, 1]).all()
    assert out["label_rating"].nunique() == 2
    assert out["notch_asof"].notna().all()


def test_ordinal_label_pit_and_no_lookahead():
    fy = pd.DataFrame(
        {
            "cik": [1, 1],
            "reporting_date": ["2011-12-31", "2012-12-31"],
            "fyear": [2011, 2012],
        }
    )
    rtg = normalize_ratings_panel(
        pd.DataFrame(
            {
                "cik": [1, 1, 1],
                "ticker": ["X", "X", "X"],
                "rating_date": ["2011-06-01", "2012-06-01", "2013-06-01"],
                "rating": ["A", "BBB", "B"],
                "agency": ["fitch", "fitch", "fitch"],
            }
        )
    )
    out = attach_rating_labels(fy, rtg, target="ordinal")
    got = out.set_index("fyear")["label_rating"].to_dict()
    assert got[2011] == 16  # A
    assert got[2012] == 13  # BBB, nicht das B von 2013


def test_parse_nrsro_filters_short_term(tmp_path: Path):
    from secpd.data.ratings import parse_nrsro_csv

    csv = tmp_path / "nrsro.csv"
    csv.write_text(
        "rating_agency_name,issuer_name,legal_entity_identifier,object_type_rated,"
        "rating,rating_action_date,rating_type,rating_type_term,central_index_key\n"
        "Moody's Investors Service,Acme Corp,HWUPKR0MPOU8FGXBT394,Instrument,"
        "Baa2,2015-06-01,Organization,LT Corporate Family Ratings,320193\n"
        "Moody's Investors Service,Acme Corp,HWUPKR0MPOU8FGXBT394,Instrument,"
        "P-2,2015-06-01,Program,Commercial Paper,320193\n"
        "Fitch Ratings,Acme Corp,,Instrument,BBB,2016-01-15,Instrument,Long Term Rating,320193\n"
        "Fitch Ratings,Acme Corp,,Instrument,A-,2017-03-01,Long Term Rating,,320193\n",
        encoding="utf-8",
    )
    out = parse_nrsro_csv(csv)
    assert not out.empty
    assert "P-2" not in set(out["rating"].astype(str))
    assert int(out["cik"].iloc[0]) == 320193
    assert out["notch"].notna().all()


def test_assign_cik_via_lei_and_name():
    from secpd.data.ratings import assign_cik_to_ratings, normalize_issuer_name

    panel = pd.DataFrame(
        {
            "cik": [pd.NA, pd.NA],
            "lei": ["HWUPKR0MPOU8FGXBT394", ""],
            "name_key": ["", normalize_issuer_name("Beta Inc")],
            "rating_date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
            "rating": ["A", "BB"],
            "notch": [16, 10],
            "agency": ["moodys", "fitch"],
        }
    )
    mp = pd.DataFrame(
        {
            "cik": [320193, 99],
            "ticker": ["AAPL", "BETA"],
            "name": ["Apple Inc.", "Beta Inc"],
            "lei": ["HWUPKR0MPOU8FGXBT394", ""],
        }
    )
    out = assign_cik_to_ratings(panel, mp)
    assert set(out["cik"]) == {320193, 99}


def test_ticker_map_from_submissions(tmp_path: Path):
    payload = {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "lei": "HWUPKR0MPOU8FGXBT394",
    }
    (tmp_path / "CIK0000320193.json").write_text(json.dumps(payload), encoding="utf-8")
    mp = ticker_map_from_submissions(tmp_path)
    assert mp.iloc[0]["cik"] == 320193
    assert mp.iloc[0]["ticker"] == "AAPL"


def test_bundle_filename_rating_variants():
    assert bundle_filename("combined", label_source="rating") == "combined_rating_ordinal.joblib"
    assert (
        bundle_filename("combined", label_source="rating", rating_target="ordinal")
        == "combined_rating_ordinal.joblib"
    )
    assert (
        bundle_filename("combined", label_source="rating", rating_target="speculative")
        == "combined_rating_speculative.joblib"
    )
    assert (
        bundle_filename(
            "financial",
            label_source="rating",
            rating_target="downgrade",
            horizon_months=12,
        )
        == "financial_rating_downgrade_h12.joblib"
    )
    parsed = parse_bundle_name("models/combined_rating_downgrade_h12.joblib")
    assert parsed["label_source"] == "rating"
    assert parsed["rating_target"] == "downgrade"
    assert parsed["horizon_months"] == 12
    parsed2 = parse_bundle_name("combined_rating_speculative.joblib")
    assert parsed2["label_source"] == "rating"
    assert parsed2["rating_target"] == "speculative"
    parsed3 = parse_bundle_name("combined_rating_ordinal.joblib")
    assert parsed3["label_source"] == "rating"
    assert parsed3["rating_target"] == "ordinal"


def test_same_training_run_id_and_mtime_fallback():
    from secpd.models.persistence import same_training_run

    assert same_training_run(
        {"train_run_id": "abc", "trained_at": None},
        {"train_run_id": "abc", "trained_at": None},
    )
    assert not same_training_run(
        {"train_run_id": "abc"},
        {"train_run_id": "xyz"},
    )
    # None vs None, aber mtimes innerhalb 2 min → gleicher Lauf
    assert same_training_run(
        {"trained_at": None, "mtime": 1_000.0},
        {"trained_at": None, "mtime": 1_050.0},
    )
    assert not same_training_run(
        {"trained_at": None, "mtime": 1_000.0},
        {"trained_at": None, "mtime": 5_000.0},
    )


def test_rating_tree_vote_prob_matches_displayed_letter():
    import numpy as np
    import pandas as pd

    from secpd.cli.scoring import rating_tree_vote_prob
    from secpd.models.pipeline import build_regression_pipeline, predict_output

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    y = np.clip(np.round(12 + X["a"] - 0.5 * X["b"]), 1, 21)
    pipe = build_regression_pipeline(["a", "b"], n_estimators=40, random_state=1)
    pipe.fit(X, y)
    pred = predict_output(pipe, X, task="regression")
    p_exact, p_pm1 = rating_tree_vote_prob(pipe, X, pred=pred)
    assert p_exact.shape == (len(X),)
    assert p_pm1.shape == (len(X),)
    assert np.all(p_exact >= 0) and np.all(p_exact <= 1)
    assert np.all(p_pm1 + 1e-12 >= p_exact)
    assert np.all(p_pm1 <= 1)
    assert float(np.nanmean(p_exact)) > 0.05
