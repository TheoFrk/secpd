# SEC-PD Frozen Benchmark

- freeze_hash: `c293c92e9297e7ecebe4c2d1cc9989a631b50e95`
- split: temporal · test n=722 · positives=6 · base=0.83%
- firm overlap: 98.5% (195/198)
- policy: legacy=False · min_fyear=2009 · require_financials=True

| Modell | ROC-AUC | PR-AUC | Brier | Skill | Top10% Capture | Top10% Lift |
|--------|---------|--------|-------|-------|----------------|-------------|
| Null (Test-Basisrate) | 0.500 | 0.008 | 0.0082 | +0.000 | 10.0% | 1.00× |
| Financial | 0.894 | 0.085 | 0.0082 | +0.008 | 66.7% | 6.69× |
| Combined | 0.946 | 0.157 | 0.0081 | +0.021 | 83.3% | 8.36× |

## CIK-Bootstrap 95%-CI

- Financial ROC: 0.790–0.982
- Combined ROC: 0.867–0.991
- Δ Combined−Financial ROC: +0.052 [+0.002, +0.130]
- Δ Combined−Financial PR: +0.072 [+0.001, +0.250]

**Hinweis:** Bei wenigen Positives sind AUC-CIs breit — Top-k und Skill mitlesen.
