# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=40454 · positives=82
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.806 · combined=0.845
- mean fold PR  financial=0.043 · combined=0.077
- pooled ROC financial=0.732 [0.676, 0.793]
- pooled ROC combined=0.790 [0.733, 0.850]
- pooled PR financial=0.014 · combined=0.055 · Top10% fin=41.5% comb=59.8%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 5061 | 6 | 0.853 | 0.888 | 0.007 | 0.007 | 0.83 | 0.67 |
| 2014 | 5650 | 12 | 0.831 | 0.900 | 0.014 | 0.038 | 0.50 | 0.58 |
| 2016 | 6079 | 8 | 0.791 | 0.756 | 0.174 | 0.186 | 0.50 | 0.50 |
| 2018 | 6789 | 23 | 0.840 | 0.895 | 0.025 | 0.108 | 0.52 | 0.74 |
| 2020 | 8227 | 8 | 0.735 | 0.839 | 0.028 | 0.096 | 0.50 | 0.75 |
| 2022 | 8648 | 25 | 0.783 | 0.793 | 0.010 | 0.026 | 0.44 | 0.48 |

## Group-Split (keine Firm-Wiederholung)

- n=9294 pos=20
- Financial ROC=0.830 PR=0.082 Top10%=0.50
- Combined  ROC=0.894 PR=0.148 Top10%=0.65
