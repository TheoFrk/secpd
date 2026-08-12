# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=1446 · positives=175
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.669 · combined=0.671
- mean fold PR  financial=0.279 · combined=0.286
- pooled ROC financial=0.686 [0.630, 0.746]
- pooled ROC combined=0.695 [0.641, 0.755]
- pooled PR financial=0.277 · combined=0.277 · Top10% fin=29.7% comb=29.1%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 321 | 35 | 0.672 | 0.664 | 0.177 | 0.173 | 0.14 | 0.14 |
| 2014 | 294 | 38 | 0.727 | 0.740 | 0.358 | 0.389 | 0.34 | 0.32 |
| 2016 | 272 | 43 | 0.783 | 0.791 | 0.400 | 0.446 | 0.30 | 0.30 |
| 2018 | 233 | 28 | 0.666 | 0.693 | 0.277 | 0.263 | 0.25 | 0.29 |
| 2020 | 213 | 17 | 0.767 | 0.745 | 0.290 | 0.267 | 0.35 | 0.35 |
| 2022 | 113 | 14 | 0.400 | 0.393 | 0.174 | 0.179 | 0.07 | 0.14 |

## Group-Split (keine Firm-Wiederholung)

- n=393 pos=37
- Financial ROC=0.597 PR=0.193 Top10%=0.27
- Combined  ROC=0.618 PR=0.205 Top10%=0.19
