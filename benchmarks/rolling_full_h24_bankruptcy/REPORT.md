# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=36739 · positives=202
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.835 · combined=0.866
- mean fold PR  financial=0.066 · combined=0.073
- pooled ROC financial=0.786 [0.741, 0.819]
- pooled ROC combined=0.777 [0.730, 0.814]
- pooled PR financial=0.046 · combined=0.056 · Top10% fin=55.0% comb=57.9%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 5061 | 19 | 0.874 | 0.900 | 0.021 | 0.026 | 0.47 | 0.53 |
| 2014 | 5650 | 20 | 0.872 | 0.899 | 0.115 | 0.102 | 0.60 | 0.75 |
| 2016 | 6079 | 28 | 0.803 | 0.851 | 0.064 | 0.083 | 0.64 | 0.61 |
| 2018 | 6789 | 31 | 0.896 | 0.920 | 0.105 | 0.104 | 0.74 | 0.81 |
| 2020 | 8227 | 25 | 0.778 | 0.839 | 0.028 | 0.042 | 0.44 | 0.64 |
| 2022 | 4933 | 79 | 0.786 | 0.789 | 0.065 | 0.080 | 0.43 | 0.47 |

## Group-Split (keine Firm-Wiederholung)

- n=8453 pos=42
- Financial ROC=0.791 PR=0.032 Top10%=0.43
- Combined  ROC=0.846 PR=0.079 Top10%=0.55
