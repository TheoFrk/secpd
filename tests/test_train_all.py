"""Sweep-Plan für scripts/train_all.py (ohne echtes Training)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_train_all():
    spec = importlib.util.spec_from_file_location(
        "train_all", ROOT / "scripts" / "train_all.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_plan_jobs_default_and_rating():
    ta = _load_train_all()
    jobs = ta.plan_jobs(horizons=(12, 36), include_fraud=True, include_rating=True)
    names = [j["name"] for j in jobs]
    assert names == [
        "default_h12",
        "default_h36",
        "fraud",
        "rating_ordinal",
    ]


def test_argv_combined_writes_same_flags():
    ta = _load_train_all()
    job = {"name": "default_h12", "label_source": "default", "horizon": 12}
    argv = ta.argv_for(
        job,
        data=ROOT / "data/processed/zenodo_full.csv.gz",
        financials=ROOT / "data/raw/financials_panel_full.csv",
        events=ROOT / "data/raw/edgar_8k_events_full.csv",
        ratings=None,
        out=ROOT / "models",
        mode="combined",
        llm="openai",
        llm_cache_only=True,
        calibrate=True,
    )
    assert "--mode" in argv and "combined" in argv
    assert "--label-source" in argv and "default" in argv
    assert "--default-horizon" in argv and "12" in argv
    assert "--llm-cache-only" in argv
    assert "--calibrate" in argv
    assert "--calibrate-method" in argv and "auto" in argv
    assert "--require-financials" in argv
    rating = {"name": "rating_ordinal", "label_source": "rating", "rating_target": "ordinal"}
    rargv = ta.argv_for(
        rating,
        data=ROOT / "data/processed/zenodo_full.csv.gz",
        financials=ROOT / "data/raw/financials_panel_full.csv",
        events=ROOT / "data/raw/edgar_8k_events_full.csv",
        ratings=ROOT / "data/raw/ratings_panel.csv",
        out=ROOT / "models",
        mode="combined",
        calibrate=True,
    )
    assert "--rating-target" in rargv and "ordinal" in rargv
    assert "--calibrate" not in rargv
