"""Tests für erweiterte Evaluation (Skill, Top-k, Cluster-Bootstrap)."""
from __future__ import annotations

import numpy as np

from secpd.evaluation import (
    brier_skill_score,
    cluster_bootstrap_ci,
    cluster_bootstrap_delta,
    evaluate_probs,
    firm_overlap_stats,
    top_k_capture,
    unseen_cik_mask,
)


def test_brier_skill_perfect_and_null():
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    base = float(y.mean())
    p_null = np.full(len(y), base)
    assert abs(brier_skill_score(y, p_null)) < 1e-9
    p_perfect = y.astype(float)
    assert brier_skill_score(y, p_perfect) > 0.9


def test_evaluate_ordinal_perfect():
    from secpd.evaluation import evaluate_ordinal

    y = np.array([12.0, 13.0, 14.0, 10.0])
    m = evaluate_ordinal(y, y)
    assert m["mae"] == 0.0
    assert m["hit_pm1"] == 1.0
    assert m["spearman"] == 1.0


def test_regression_pipeline_smoke():
    import pandas as pd
    from secpd.models.pipeline import build_regression_pipeline, predict_output

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = np.clip(np.round(12 + X["a"] - 0.5 * X["b"]), 1, 21)
    pipe = build_regression_pipeline(["a", "b"], n_estimators=40, random_state=0)
    pipe.fit(X, y)
    pred = predict_output(pipe, X, task="regression")
    assert pred.min() >= 1.0 and pred.max() <= 21.0
    assert float(np.mean(np.abs(pred - y))) < 3.0


def test_evaluate_probs_includes_skill():
    y = np.array([0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.2, 0.05, 0.7, 0.15])
    m = evaluate_probs(y, p)
    assert "brier_skill" in m
    assert m["brier_skill"] > 0


def test_top_k_capture():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    m = top_k_capture(y, p, fractions=(0.1, 0.5))
    assert m["top_10pct_events"] == 1.0
    assert m["top_10pct_capture"] == 0.5
    assert m["top_50pct_events"] == 2.0
    assert m["top_50pct_capture"] == 1.0


def test_cluster_bootstrap_ci_runs():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5)
    y = rng.integers(0, 2, size=len(groups))
    # mind. 2 Klassen erzwingen
    y[0] = 0
    y[1] = 1
    p = rng.random(len(groups))
    ci = cluster_bootstrap_ci(y, p, groups, n_boot=50, seed=0)
    assert "roc_auc" in ci
    assert ci["roc_auc"]["ci_low"] <= ci["roc_auc"]["ci_high"]


def test_cluster_bootstrap_delta_and_overlap():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    p_a = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.2, 0.85, 0.15])
    p_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    d = cluster_bootstrap_delta(y, p_a, p_b, groups, n_boot=40, seed=1, metric="roc_auc")
    assert d["point"] > 0
    ov = firm_overlap_stats([1, 2, 3], [2, 3, 9])
    assert ov["n_overlap_firms"] == 2
    assert abs(ov["overlap_rate"] - 2 / 3) < 1e-9
    mask = unseen_cik_mask([1, 2, 3], [2, 9, 3, 9])
    assert list(mask) == [False, True, False, True]
