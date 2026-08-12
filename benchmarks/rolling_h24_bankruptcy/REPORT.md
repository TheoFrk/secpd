# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=2445 · positives=58
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.811 · combined=0.893
- mean fold PR  financial=0.219 · combined=0.232
- pooled ROC financial=0.833 [0.771, 0.891]
- pooled ROC combined=0.879 [0.824, 0.929]
- pooled PR financial=0.197 · combined=0.196 · Top10% fin=58.6% comb=65.5%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 510 | 8 | 0.425 | 0.901 | 0.035 | 0.183 | 0.25 | 0.62 |
| 2014 | 482 | 8 | 0.893 | 0.897 | 0.102 | 0.115 | 0.62 | 0.75 |
| 2016 | 455 | 11 | 0.860 | 0.825 | 0.245 | 0.231 | 0.55 | 0.55 |
| 2018 | 414 | 15 | 0.848 | 0.855 | 0.293 | 0.238 | 0.53 | 0.60 |
| 2020 | 384 | 11 | 0.874 | 0.914 | 0.192 | 0.224 | 0.73 | 0.73 |
| 2022 | 200 | 5 | 0.968 | 0.962 | 0.448 | 0.399 | 0.80 | 0.80 |

## Group-Split (keine Firm-Wiederholung)

- n=684 pos=13
- Financial ROC=0.860 PR=0.167 Top10%=0.46
- Combined  ROC=0.823 PR=0.115 Top10%=0.54
