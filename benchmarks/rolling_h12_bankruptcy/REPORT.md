# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 5 · pooled n=2065 · positives=22
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.759 · combined=0.863
- mean fold PR  financial=0.098 · combined=0.091
- pooled ROC financial=0.750 [0.619, 0.860]
- pooled ROC combined=0.841 [0.737, 0.913]
- pooled PR financial=0.050 · combined=0.060 · Top10% fin=40.9% comb=50.0%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2014 | 482 | 3 | 0.736 | 0.823 | 0.031 | 0.029 | 0.33 | 0.33 |
| 2016 | 455 | 4 | 0.518 | 0.775 | 0.017 | 0.044 | 0.25 | 0.25 |
| 2018 | 414 | 9 | 0.734 | 0.810 | 0.166 | 0.135 | 0.33 | 0.44 |
| 2020 | 384 | 4 | 0.832 | 0.941 | 0.101 | 0.132 | 0.50 | 0.75 |
| 2022 | 330 | 2 | 0.979 | 0.966 | 0.174 | 0.117 | 1.00 | 1.00 |

## Group-Split (keine Firm-Wiederholung)

- n=715 pos=4
- Financial ROC=0.683 PR=0.055 Top10%=0.50
- Combined  ROC=0.667 PR=0.056 Top10%=0.50
