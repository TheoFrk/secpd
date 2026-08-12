# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 5 · pooled n=2065 · positives=22
- text features: llm_risk_sentiment, llm_complexity_score, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.759 · combined=0.857
- mean fold PR  financial=0.098 · combined=0.117
- pooled ROC financial=0.750 [0.627, 0.859]
- pooled ROC combined=0.834 [0.744, 0.899]
- pooled PR financial=0.050 · combined=0.072 · Top10% fin=40.9% comb=59.1%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2014 | 482 | 3 | 0.736 | 0.802 | 0.031 | 0.028 | 0.33 | 0.33 |
| 2016 | 455 | 4 | 0.518 | 0.849 | 0.017 | 0.067 | 0.25 | 0.50 |
| 2018 | 414 | 9 | 0.734 | 0.753 | 0.166 | 0.125 | 0.33 | 0.44 |
| 2020 | 384 | 4 | 0.832 | 0.915 | 0.101 | 0.248 | 0.50 | 0.75 |
| 2022 | 330 | 2 | 0.979 | 0.966 | 0.174 | 0.118 | 1.00 | 1.00 |

## Group-Split (keine Firm-Wiederholung)

- n=715 pos=4
- Financial ROC=0.683 PR=0.055 Top10%=0.50
- Combined  ROC=0.670 PR=0.050 Top10%=0.50
