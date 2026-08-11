"""sklearn-Pipelines (Option A: Combined-ML; plus Text-Only-Modell für Option B).

* RandomForest mit ``class_weight="balanced_subsample"`` — Fraud-/Default-
  Labels sind stark unbalanciert.
* Optionale Kalibrierung (isotonic, CV): Rohe RF-Scores sind als PDs schlecht
  kalibriert; für Kreditrisiko-Zwecke zählt neben dem Ranking auch das
  Wahrscheinlichkeitsniveau (Brier-Score in der Evaluation).
* Nur Median-Imputation als Preprocessing — Bäume brauchen kein Scaling,
  und weniger Fitting-Schritte bedeuten weniger Leakage-Fläche.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(
    numeric_features: list[str],
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    calibrate: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Tabulare Klassifikations-Pipeline (Finanz- und/oder LLM-Features)."""
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
    estimator = (
        CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3) if calibrate else clf
    )
    return Pipeline([("pre", preprocessor), ("clf", estimator)])


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
