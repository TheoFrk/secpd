# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=1519 · positives=87
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.661 · combined=0.654
- mean fold PR  financial=0.147 · combined=0.146
- pooled ROC financial=0.655 [0.591, 0.716]
- pooled ROC combined=0.656 [0.597, 0.710]
- pooled PR financial=0.109 · combined=0.101 · Top10% fin=18.4% comb=16.1%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 321 | 15 | 0.624 | 0.644 | 0.064 | 0.067 | 0.07 | 0.07 |
| 2014 | 294 | 17 | 0.667 | 0.712 | 0.164 | 0.170 | 0.24 | 0.18 |
| 2016 | 272 | 24 | 0.723 | 0.727 | 0.219 | 0.248 | 0.33 | 0.33 |
| 2018 | 233 | 16 | 0.621 | 0.627 | 0.105 | 0.099 | 0.06 | 0.06 |
| 2020 | 213 | 8 | 0.813 | 0.726 | 0.149 | 0.115 | 0.38 | 0.25 |
| 2022 | 186 | 7 | 0.516 | 0.488 | 0.183 | 0.176 | 0.14 | 0.14 |

## Group-Split (keine Firm-Wiederholung)

- n=409 pos=18
- Financial ROC=0.613 PR=0.184 Top10%=0.22
- Combined  ROC=0.614 PR=0.133 Top10%=0.28
