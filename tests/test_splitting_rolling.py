"""Tests für Rolling-Origin-Splits und Group-Kalibrierungs-CV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from secpd.models.pipeline import build_pipeline, fit_pipeline, make_calibration_cv
from secpd.splitting import rolling_origin_splits


def test_rolling_origin_splits_basic():
    rows = []
    for cik in range(1, 6):
        for yr in range(2010, 2021):
            rows.append(
                {
                    "cik": cik,
                    "fyear": yr,
                    "label_default": 1 if (cik == 1 and yr in (2014, 2018)) or (cik == 2 and yr == 2016) else 0,
                }
            )
    # Extra positives so min_train_positives passes for later cutoffs
    for yr in (2011, 2012, 2013, 2015, 2017, 2019):
        rows.append({"cik": 99, "fyear": yr, "label_default": 1})
    df = pd.DataFrame(rows)
    folds = rolling_origin_splits(
        df,
        label_col="label_default",
        year_col="fyear",
        test_window_years=2,
        step_years=2,
        min_train_years=4,
        min_train_positives=2,
        min_test_positives=1,
    )
    assert len(folds) >= 1
    for tr, te, meta in folds:
        assert meta["cutoff"] < min(meta["test_years"])
        assert len(tr) > 0 and len(te) > 0
        assert set(tr).isdisjoint(set(te))


def test_make_calibration_cv_groupkfold():
    groups = np.repeat(np.arange(12), 4)
    y = np.zeros(len(groups), dtype=int)
    # Positives in mehreren Gruppen
    y[0] = 1
    y[8] = 1
    y[16] = 1
    y[24] = 1
    y[32] = 1
    y[40] = 1
    cv = make_calibration_cv(groups, y, n_splits=3)
    assert isinstance(cv, list)
    assert len(cv) == 3
    assert all(len(pair) == 2 for pair in cv)


def test_fit_pipeline_with_groups():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"fin_a": rng.normal(size=n), "fin_b": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    y[:10] = 0
    y[10:20] = 1
    groups = np.repeat(np.arange(16), 5)
    cv = make_calibration_cv(groups, y, n_splits=3)
    pipe = build_pipeline(
        ["fin_a", "fin_b"],
        n_estimators=20,
        calibrate=True,
        calibration_method="sigmoid",
        cv=cv,
        random_state=0,
    )
    fit_pipeline(pipe, X, y, groups=groups)
    assert isinstance(pipe.named_steps["clf"], CalibratedClassifierCV)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (n,)
