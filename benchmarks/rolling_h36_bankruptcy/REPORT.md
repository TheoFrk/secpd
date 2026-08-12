# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 5 · pooled n=2245 · positives=84
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.846 · combined=0.855
- mean fold PR  financial=0.295 · combined=0.301
- pooled ROC financial=0.834 [0.771, 0.886]
- pooled ROC combined=0.843 [0.785, 0.894]
- pooled PR financial=0.256 · combined=0.253 · Top10% fin=53.6% comb=59.5%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 510 | 11 | 0.815 | 0.809 | 0.475 | 0.450 | 0.73 | 0.73 |
| 2014 | 482 | 13 | 0.851 | 0.869 | 0.106 | 0.122 | 0.38 | 0.46 |
| 2016 | 455 | 22 | 0.785 | 0.807 | 0.300 | 0.298 | 0.41 | 0.55 |
| 2018 | 414 | 21 | 0.853 | 0.849 | 0.255 | 0.238 | 0.43 | 0.38 |
| 2020 | 384 | 17 | 0.927 | 0.940 | 0.340 | 0.395 | 0.76 | 0.82 |

## Group-Split (keine Firm-Wiederholung)

- n=647 pos=21
- Financial ROC=0.850 PR=0.330 Top10%=0.57
- Combined  ROC=0.849 PR=0.285 Top10%=0.62
