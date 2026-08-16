# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=25628 · positives=2681
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.748 · combined=0.750
- mean fold PR  financial=0.291 · combined=0.294
- pooled ROC financial=0.771 [0.760, 0.783]
- pooled ROC combined=0.774 [0.763, 0.787]
- pooled PR financial=0.358 · combined=0.368 · Top10% fin=37.9% comb=38.6%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 3713 | 212 | 0.683 | 0.688 | 0.130 | 0.115 | 0.27 | 0.21 |
| 2014 | 4054 | 261 | 0.719 | 0.713 | 0.153 | 0.165 | 0.33 | 0.33 |
| 2016 | 4267 | 326 | 0.758 | 0.766 | 0.213 | 0.217 | 0.34 | 0.33 |
| 2018 | 4672 | 375 | 0.759 | 0.762 | 0.254 | 0.260 | 0.35 | 0.36 |
| 2020 | 5578 | 812 | 0.807 | 0.812 | 0.475 | 0.485 | 0.37 | 0.38 |
| 2022 | 3344 | 695 | 0.759 | 0.759 | 0.522 | 0.523 | 0.32 | 0.32 |

## Group-Split (keine Firm-Wiederholung)

- n=6176 pos=583
- Financial ROC=0.757 PR=0.327 Top10%=0.37
- Combined  ROC=0.763 PR=0.333 Top10%=0.37
