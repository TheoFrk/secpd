# SEC-PD Frozen Benchmark

- freeze_hash: `5603bc3b684fc6b5f4a2ceca1ca0353b570a7a97`
- split: temporal · test n=13036 · positives=32 · base=0.25%
- firm overlap: 86.4% (4027/4662)
- policy: legacy=False · min_fyear=2009 · require_financials=True

| Modell | ROC-AUC | PR-AUC | Brier | Skill | Top10% Capture | Top10% Lift |
|--------|---------|--------|-------|-------|----------------|-------------|
| Null (Test-Basisrate) | 0.500 | 0.002 | 0.0024 | +0.000 | 10.0% | 1.00× |
| Financial | 0.793 | 0.012 | 0.0025 | -0.013 | 53.1% | 5.31× |
| Combined | 0.868 | 0.043 | 0.0024 | +0.012 | 56.2% | 5.62× |

## CIK-Bootstrap 95%-CI

- Financial ROC: 0.700–0.870
- Combined ROC: 0.817–0.912
- Δ Combined−Financial ROC: +0.075 [+0.033, +0.127]
- Δ Combined−Financial PR: +0.030 [+0.002, +0.115]

**Hinweis:** Bei wenigen Positives sind AUC-CIs breit — Top-k und Skill mitlesen.
