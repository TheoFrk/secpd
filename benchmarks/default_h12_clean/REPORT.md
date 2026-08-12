# SEC-PD Frozen Benchmark

- freeze_hash: `c293c92e9297e7ecebe4c2d1cc9989a631b50e95`
- split: temporal · test n=722 · positives=6 · base=0.83%
- firm overlap: 98.5% (195/198)
- policy: legacy=False · min_fyear=2009 · require_financials=True

| Modell | ROC-AUC | PR-AUC | Brier | Skill | Top10% Capture | Top10% Lift |
|--------|---------|--------|-------|-------|----------------|-------------|
| Null (Test-Basisrate) | 0.500 | 0.008 | 0.0082 | +0.000 | 10.0% | 1.00× |
| Financial | 0.904 | 0.105 | 0.0082 | +0.006 | 66.7% | 6.69× |
| Combined | 0.896 | 0.149 | 0.0080 | +0.028 | 66.7% | 6.69× |

## CIK-Bootstrap 95%-CI

- Financial ROC: 0.783–0.985
- Combined ROC: 0.746–0.989
- Δ Combined−Financial ROC: -0.007 [-0.036, +0.012]
- Δ Combined−Financial PR: +0.045 [-0.026, +0.223]

**Hinweis:** Bei wenigen Positives sind AUC-CIs breit — Top-k und Skill mitlesen.
