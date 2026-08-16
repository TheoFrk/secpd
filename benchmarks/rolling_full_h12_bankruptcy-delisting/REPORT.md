# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=27553 · positives=1393
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.725 · combined=0.735
- mean fold PR  financial=0.144 · combined=0.147
- pooled ROC financial=0.757 [0.744, 0.770]
- pooled ROC combined=0.762 [0.750, 0.775]
- pooled PR financial=0.179 · combined=0.185 · Top10% fin=39.3% comb=40.3%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 3713 | 98 | 0.647 | 0.663 | 0.058 | 0.052 | 0.19 | 0.23 |
| 2014 | 4054 | 127 | 0.700 | 0.715 | 0.090 | 0.083 | 0.31 | 0.35 |
| 2016 | 4267 | 168 | 0.724 | 0.741 | 0.126 | 0.132 | 0.33 | 0.34 |
| 2018 | 4672 | 189 | 0.723 | 0.729 | 0.113 | 0.112 | 0.32 | 0.35 |
| 2020 | 5578 | 440 | 0.784 | 0.785 | 0.243 | 0.253 | 0.37 | 0.38 |
| 2022 | 5269 | 371 | 0.769 | 0.776 | 0.233 | 0.247 | 0.37 | 0.41 |

## Group-Split (keine Firm-Wiederholung)

- n=6554 pos=307
- Financial ROC=0.754 PR=0.198 Top10%=0.39
- Combined  ROC=0.759 PR=0.215 Top10%=0.43
