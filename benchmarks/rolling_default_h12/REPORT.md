# SEC-PD Rolling-Origin (Financial)

- folds: 5 · pooled n=2065 · positives=22
- mean fold ROC=0.759 · PR=0.098 · Skill=+0.005
- mean firm-overlap: 97.3%
- pooled ROC=0.750 [0.627, 0.859]
- pooled PR=0.050 · Skill=+0.008 · Top10% capture=40.9%

| cutoff | test years | n | pos | ROC | PR | Skill | overlap | new-firm ROC |
|--------|------------|---|-----|-----|----|-------|---------|--------------|
| 2014 | 2015,2016 | 482 | 3 | 0.736 | 0.031 | -0.019 | 96% | 0.643 |
| 2016 | 2017,2018 | 455 | 4 | 0.518 | 0.017 | -0.002 | 95% | n/a |
| 2018 | 2019,2020 | 414 | 9 | 0.734 | 0.166 | +0.009 | 97% | n/a |
| 2020 | 2021,2022 | 384 | 4 | 0.832 | 0.101 | +0.014 | 98% | n/a |
| 2022 | 2023,2024 | 330 | 2 | 0.979 | 0.174 | +0.023 | 100% | n/a |

Financial-only Rolling-Eval. Combined absichtlich ausgelassen (Freeze: kein signifikanter Mehrwert).
