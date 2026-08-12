"""Keyword-Zähler und schlanke LLM-Modellfeatures."""
from __future__ import annotations

import pandas as pd

from secpd.features.textual import (
    combined_text_feature_names,
    extract_keyword_features,
    keyword_feature_names,
    text_feature_names,
)
from secpd.llm.schema import TextRiskProfile


def test_keyword_going_concern_and_covenant():
    df = pd.DataFrame(
        {
            "doc_id": ["a", "b"],
            "mda": [
                "There is substantial doubt about our ability to continue as a going concern. "
                "A covenant violation occurred last quarter.",
                "Revenue grew and liquidity remained ample. We believe results will improve.",
            ],
        }
    )
    out = extract_keyword_features(df, text_col="mda")
    assert out.loc[0, "txt_going_concern"] >= 2
    assert out.loc[0, "txt_covenant"] >= 1
    assert out.loc[1, "txt_going_concern"] == 0
    assert out.loc[1, "txt_covenant"] == 0


def test_model_feature_names_are_slim():
    names = text_feature_names()
    assert names == ["llm_risk_sentiment", "llm_complexity_score"]
    assert "llm_confidence" not in names
    assert set(keyword_feature_names()) <= set(combined_text_feature_names())
    # Cache/Debug behält alle Profilfelder
    assert "llm_vagueness_score" in TextRiskProfile.feature_names()
