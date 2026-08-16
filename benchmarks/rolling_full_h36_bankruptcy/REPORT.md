# SEC-PD Rolling-Origin (Financial vs Combined)

- folds: 6 · pooled n=32492 · positives=316
- text features: llm_risk_sentiment, txt_going_concern, txt_liquidity_stress, txt_covenant, txt_restructuring, txt_bankruptcy_lang
- mean fold ROC financial=0.813 · combined=0.840
- mean fold PR  financial=0.138 · combined=0.168
- pooled ROC financial=0.779 [0.745, 0.809]
- pooled ROC combined=0.804 [0.772, 0.832]
- pooled PR financial=0.061 · combined=0.077 · Top10% fin=44.9% comb=53.2%

| cutoff | n | pos | ROC fin | ROC comb | PR fin | PR comb | Top10% fin | Top10% comb |
|--------|---|-----|---------|----------|--------|---------|------------|-------------|
| 2012 | 5061 | 32 | 0.807 | 0.806 | 0.030 | 0.064 | 0.47 | 0.38 |
| 2014 | 5650 | 29 | 0.874 | 0.921 | 0.186 | 0.209 | 0.59 | 0.76 |
| 2016 | 6079 | 50 | 0.833 | 0.878 | 0.082 | 0.154 | 0.62 | 0.66 |
| 2018 | 6789 | 40 | 0.866 | 0.890 | 0.098 | 0.090 | 0.68 | 0.72 |
| 2020 | 8227 | 68 | 0.688 | 0.719 | 0.025 | 0.039 | 0.35 | 0.38 |
| 2022 | 686 | 97 | 0.813 | 0.823 | 0.408 | 0.455 | 0.33 | 0.39 |

## Group-Split (keine Firm-Wiederholung)

- n=7479 pos=100
- Financial ROC=0.834 PR=0.131 Top10%=0.46
- Combined  ROC=0.848 PR=0.162 Top10%=0.50
