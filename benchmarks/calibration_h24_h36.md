# Kalibrierung: sigmoid vs. isotonic (h24 / h36)

Holdout wie `train.py` (temporal, `zenodo_full`, min_fyear=2009, require-financials,
bankruptcy-only). Combined-Modell, LLM cache-only.

Leitmetrik: **Brier-Skill** (sekundär ROC). Positive im Test: h24=96, h36=165.

## Gleicher Feature-Stand (nach SIC-Division / Größen-Buckets)

| Horizont | Methode | ROC | PR | Brier | Skill |
|----------|---------|-----|----|-------|-------|
| h24 | sigmoid | 0.805 | 0.061 | 0.01046 | −0.010 |
| h24 | isotonic | 0.806 | 0.063 | 0.01049 | −0.013 |
| h36 | sigmoid | 0.769 | 0.060 | 0.01860 | −0.024 |
| h36 | isotonic | 0.763 | 0.059 | 0.01826 | −0.005 |

- h24: Skill knapp bei sigmoid, Ranking praktisch gleich.
- h36: isotonic klar besser kalibriert (Skill −0.005 vs. −0.024); sigmoid hat etwas höheres ROC.

## Ausgelieferte Bundles (ohne neue Branchen-/Größen-Dummies)

| Bundle | Methode | ROC | Skill |
|--------|---------|-----|-------|
| combined_default_h24 | isotonic | 0.816 | **+0.010** |
| combined_default_h36 | isotonic | 0.755 | −0.001 |

Die Produktions-Bundles bleiben. Die Neu-Trainings mit SIC-/Größen-Dummies
verschlechtern den Skill gegenüber dem eingefrorenen Stand — die Dummies
wandern deshalb nur ins Rating-Retraining, nicht in die PD-Bundles.

`train.py --calibrate-method auto` bleibt: sigmoid bei <100 Trainings-Positiven,
sonst isotonic (passt zu h36). Flags `--calibrate-method {auto,sigmoid,isotonic}`
sind in `train.py`, `scripts/train_all.py` und `scripts/rolling_eval.py` durchgereicht.
