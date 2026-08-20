"""sklearn-Pipelines (Option A: Combined-ML; plus Text-Only-Modell für Option B).

* RandomForest mit ``class_weight="balanced_subsample"`` — Fraud-/Default-
  Labels sind stark unbalanciert.
* Optionale Kalibrierung: Default ``sigmoid`` (stabil bei wenigen Positiven);
  Cross-Validation bevorzugt Group-Splits über CIK (als Index-Liste an
  ``CalibratedClassifierCV``, ohne Metadata-Routing).
* Nur Median-Imputation als Preprocessing — Bäume brauchen kein Scaling,
  und weniger Fitting-Schritte bedeuten weniger Leakage-Fläche.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_calibration_cv(
    groups: np.ndarray | pd.Series | None,
    y: np.ndarray,
    *,
    n_splits: int = 3,
) -> Any:
    """Gruppen-bewusste CV als Index-Splits (kein Metadata-Routing nötig).

    Liefert ``[(train_idx, test_idx), ...]`` für ``CalibratedClassifierCV(cv=...)``,
    falls GroupKFold beide Klassen in jedem Fold behält — sonst ``int``.
    """
    if groups is None:
        return max(2, min(n_splits, 3))
    g = np.asarray(groups)
    y = np.asarray(y).astype(int)
    n_groups = int(pd.Series(g).nunique())
    splits = min(n_splits, n_groups)
    if splits < 2:
        return max(2, min(n_splits, 3))
    gkf = GroupKFold(n_splits=splits)
    dummy = np.zeros(len(y))
    fold_ids = list(gkf.split(dummy, y, groups=g))
    for tr, te in fold_ids:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return max(2, min(n_splits, 3))
    return fold_ids


def build_pipeline(
    numeric_features: list[str],
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    cv: Any = 3,
    random_state: int = 42,
) -> Pipeline:
    """Tabulare Klassifikations-Pipeline (Finanz- und/oder LLM-Features).

    Kalibrierung default ``sigmoid`` — bei wenigen Positiven stabiler als
    ``isotonic`` (siehe Daten-Sanierungsplan).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), list(numeric_features)),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    method = calibration_method if calibration_method in {"sigmoid", "isotonic"} else "sigmoid"
    estimator = (
        CalibratedClassifierCV(estimator=clf, method=method, cv=cv) if calibrate else clf
    )
    return Pipeline([("pre", preprocessor), ("clf", estimator)])


def fit_pipeline(
    pipe: Pipeline,
    X: Any,
    y: Any,
    *,
    groups: np.ndarray | pd.Series | None = None,
) -> Pipeline:
    """``pipe.fit`` — Gruppen-Splits werden in ``make_calibration_cv`` vorab gebaut."""
    del groups
    return pipe.fit(X, y)


def build_text_pipeline(
    llm_features: list[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Kleines Text-Only-Modell (Option B): logistische Regression auf LLM-Scores.

    Interpretierbar und mit wenigen Features stabil — liefert ``p_text`` für
    die Ensemble-Aggregation.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(llm_features),
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    return Pipeline([("pre", preprocessor), ("clf", clf)])


def build_regression_pipeline(
    numeric_features: list[str],
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Tabulare Regressions-Pipeline (ordinales Rating, Notch 1–21)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), list(numeric_features)),
        ],
        remainder="drop",
    )
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("pre", preprocessor), ("reg", reg)])


def is_regression_pipeline(pipe: Pipeline) -> bool:
    """True, wenn die letzte Stufe kein ``predict_proba`` hat."""
    last = pipe.steps[-1][1] if pipe.steps else None
    return hasattr(last, "predict") and not hasattr(last, "predict_proba")


def predict_output(
    pipe: Pipeline,
    X: Any,
    *,
    task: str | None = None,
    clip_lo: float = 1.0,
    clip_hi: float = 21.0,
) -> np.ndarray:
    """Klassifikation → P(positiv); Regression → geclippte Notch-Vorhersage."""
    if task is None:
        task = "regression" if is_regression_pipeline(pipe) else "classification"
    if task == "regression":
        pred = np.asarray(pipe.predict(X), dtype=float)
        return np.clip(pred, clip_lo, clip_hi)
    return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
