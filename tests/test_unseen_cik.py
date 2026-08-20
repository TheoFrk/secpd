"""Modellgüte auf fremden CIKs (nicht im Trainingsuniversum)."""
from __future__ import annotations

import numpy as np

from secpd.data.synthetic import make_synthetic_dataset
from secpd.evaluation import evaluate_by_firm_novelty, evaluate_probs, firm_overlap_stats
from secpd.features.financial import add_financial_features
from secpd.models.pipeline import build_pipeline
from secpd.splitting import smart_split


def test_synthetic_panel_repeats_ciks_across_years():
    df = make_synthetic_dataset(n=200, seed=1)
    years_per_firm = df.groupby("cik")["fyear"].nunique()
    assert df["cik"].nunique() < len(df)
    assert int(years_per_firm.min()) >= 2


def test_temporal_split_keeps_firms_in_universe():
    df = make_synthetic_dataset(n=400, seed=2)
    tr, te, strat = smart_split(
        df, label_col="label", group_col="cik", year_col="fyear",
        test_size=0.25, strategy="temporal", random_state=0,
    )
    assert strat == "temporal"
    ov = firm_overlap_stats(df.iloc[tr]["cik"], df.iloc[te]["cik"])
    assert ov["overlap_rate"] > 0.5
    assert ov["n_overlap_firms"] > 0


def test_group_split_holds_out_entire_ciks():
    df = make_synthetic_dataset(n=400, seed=3)
    tr, te, strat = smart_split(
        df, label_col="label", group_col="cik", year_col="fyear",
        test_size=0.25, strategy="group", random_state=0,
    )
    assert strat == "group"
    ov = firm_overlap_stats(df.iloc[tr]["cik"], df.iloc[te]["cik"])
    assert ov["overlap_rate"] == 0.0
    assert ov["n_new_test_firms"] == ov["n_test_firms"]
    assert set(df.iloc[tr]["cik"]).isdisjoint(set(df.iloc[te]["cik"]))


def test_evaluate_by_firm_novelty_splits_seen_and_unseen():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    train = [1, 1, 2]
    test = [1, 1, 9, 9, 9, 9]
    out = evaluate_by_firm_novelty(y, p, train_groups=train, test_groups=test)
    assert out["overlap"]["n_overlap_firms"] == 1
    assert out["overlap"]["n_new_test_firms"] == 1
    assert out["in_universe"]["n"] == 2
    assert out["unseen_cik"]["n"] == 4
    assert out["unseen_cik"]["roc_auc"] > 0.9


def test_model_generalizes_to_unseen_ciks():
    """Signal muss auf gehaltenen Firmen greifen, nicht nur auf bekannten CIKs."""
    df = make_synthetic_dataset(n=500, seed=7)
    df, fin_cols = add_financial_features(df)
    tr, te, strat = smart_split(
        df, label_col="label", group_col="cik", year_col="fyear",
        test_size=0.3, strategy="group", random_state=0,
    )
    assert strat == "group"
    ov = firm_overlap_stats(df.iloc[tr]["cik"], df.iloc[te]["cik"])
    assert ov["overlap_rate"] == 0.0

    y_tr = df.iloc[tr]["label"].to_numpy()
    y_te = df.iloc[te]["label"].to_numpy()
    pipe = build_pipeline(fin_cols, n_estimators=80, random_state=0)
    pipe.fit(df.iloc[tr], y_tr)
    p = pipe.predict_proba(df.iloc[te])[:, 1]
    metrics = evaluate_probs(y_te, p)
    novelty = evaluate_by_firm_novelty(
        y_te, p,
        train_groups=df.iloc[tr]["cik"],
        test_groups=df.iloc[te]["cik"],
    )
    assert novelty["in_universe"]["n"] == 0
    assert novelty["unseen_cik"]["n"] == metrics["n"]
    assert metrics["roc_auc"] > 0.60
    assert novelty["unseen_cik"]["roc_auc"] > 0.60
