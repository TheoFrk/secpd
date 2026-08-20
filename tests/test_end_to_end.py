"""End-to-End: Synthetik → Features → Split → Training → Persistenz → Ensemble."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpd.data.synthetic import make_synthetic_dataset
from secpd.evaluation import decile_table, evaluate_probs, firm_overlap_stats
from secpd.features.financial import add_financial_features
from secpd.features.textual import extract_text_features, text_feature_names
from secpd.llm.mock import MockLLMClient
from secpd.models.ensemble import EnsembleWeights, combine_probabilities
from secpd.models.persistence import ModelBundle, load_any, runtime_metadata, save_single
from secpd.models.pipeline import build_pipeline, build_text_pipeline
from secpd.splitting import smart_split


def test_full_pipeline(tmp_path: Path) -> None:
    df = make_synthetic_dataset(n=500, seed=7)
    df, fin_cols = add_financial_features(df)

    text_feats = extract_text_features(
        df, client=MockLLMClient(), text_col="mda", id_col="doc_id", progress_every=10_000
    )
    df = df.merge(text_feats, on="doc_id", how="left")
    llm_cols = text_feature_names()

    tr, te, strategy = smart_split(
        df, label_col="label", group_col="cik", year_col="fyear", test_size=0.25
    )
    assert strategy in {"temporal", "group", "random"}
    assert set(tr).isdisjoint(set(te))

    y_tr = df.iloc[tr]["label"].to_numpy()
    y_te = df.iloc[te]["label"].to_numpy()

    pipe = build_pipeline(fin_cols + llm_cols, n_estimators=120)
    pipe.fit(df.iloc[tr], y_tr)
    p = pipe.predict_proba(df.iloc[te])[:, 1]

    metrics = evaluate_probs(y_te, p)
    assert 0.0 <= p.min() and p.max() <= 1.0
    assert metrics["roc_auc"] > 0.60  # Signal fließt end-to-end

    # Fremde CIKs: Group-Holdout, keine Firma aus dem Training.
    tr_g, te_g, strat_g = smart_split(
        df,
        label_col="label",
        group_col="cik",
        year_col="fyear",
        test_size=0.3,
        strategy="group",
        random_state=0,
    )
    assert strat_g == "group"
    ov = firm_overlap_stats(df.iloc[tr_g]["cik"], df.iloc[te_g]["cik"])
    assert ov["overlap_rate"] == 0.0
    pipe_g = build_pipeline(fin_cols + llm_cols, n_estimators=120)
    pipe_g.fit(df.iloc[tr_g], df.iloc[tr_g]["label"].to_numpy())
    p_g = pipe_g.predict_proba(df.iloc[te_g])[:, 1]
    unseen = evaluate_probs(df.iloc[te_g]["label"].to_numpy(), p_g)
    assert unseen["roc_auc"] > 0.60

    # Persistenz-Roundtrip
    path = tmp_path / "model.joblib"
    save_single(ModelBundle(pipe, fin_cols + llm_cols, runtime_metadata()), path)
    payload = load_any(path)
    p2 = payload["pipeline"].predict_proba(df.iloc[te])[:, 1]
    assert np.allclose(p, p2)

    # Dezil-Tabelle: Struktur + Trennschärfe (Top-30 % fangen > 35 % der Events;
    # robuster als ein einzelnes Top-Dezil mit ~15–20 Beobachtungen)
    table = decile_table(y_te, p)
    assert len(table) == 10
    assert table["cum_capture"].is_monotonic_increasing
    assert abs(table["cum_capture"].iloc[-1] - 1.0) < 1e-9
    assert table["cum_capture"].iloc[2] > 0.35

    # Ensemble-Pfad
    text_pipe = build_text_pipeline(llm_cols)
    text_pipe.fit(df.iloc[tr], y_tr)
    p_text = text_pipe.predict_proba(df.iloc[te])[:, 1]
    p_ens = combine_probabilities(p, p_text, EnsembleWeights(0.6, 0.4))
    assert p_ens.shape == p.shape
    assert (p_ens >= 0).all() and (p_ens <= 1).all()
