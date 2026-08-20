# SEC-PD — Ausfall-/Misconduct-Scoring aus SEC-Filings

Eigenständiges Python-Modul (1-Wochen-Praktikumsbeitrag, Credit Risk Innovation):
Vorhersage eines Risiko-Scores je Firm-Year aus **tabellarischen Finanzkennzahlen**
und **LLM-extrahierten Textmerkmalen** (10-K MD&A). Modular gebaut, offline-fähig
und unabhängig von einem bestimmten Zielsystem einsetzbar.

## Datenquelle (Zenodo)

Die 10-K-/AAER-Rohdaten liegen **nicht** in diesem Repo (Größe). Bitte den
öffentlichen Datensatz herunterladen:

**[10k-fraud-detection — Zenodo Record 17121948](https://zenodo.org/records/17121948)**  
DOI: [10.5281/zenodo.17121948](https://doi.org/10.5281/zenodo.17121948)

Benötigte Dateien aus dem Record:

| Datei | Zweck |
|---|---|
| `aaer_mark5.csv` | AAER-Zeitfenster (Fraud-Label) |
| `firm_years_labels.json` | Firm-Years + MD&A (~716 MB) |
| `firm_years.json` | optional, vollständiges Set (~4.9 GB) |

```bash
python scripts/fetch_zenodo.py --files aaer_mark5.csv firm_years_labels.json
```

## Architektur

```
                 ┌──────────────────────────────┐
 Zenodo 17121948 │ firm_years_labels.json (MD&A)│──┐
                 │ aaer_mark5.csv (AAER-Fenster)│  │  scripts/convert_zenodo.py
                 └──────────────────────────────┘  ▼  (Label = CIK × Zeitfenster-Join)
 SEC EDGAR       ┌──────────────────────────────┐  data/processed/zenodo_labeled.csv.gz
 (companyfacts)  │ scripts/fetch_edgar_financials│─────────────┐
                 └──────────────────────────────┘              ▼
                                                     train.py ──┬── features/financial.py  (pandas-Ratios)
                                                                ├── features/textual.py ── llm/ (Mock|Bank + Cache)
                                                                ├── splitting.py (temporal/group)
                                                                ├── models/pipeline.py (RF, opt. kalibriert)
                                                                ├── models/ensemble.py (Option B, Logit-Raum)
                                                                └── models/persistence.py (.joblib + Metadaten)
```

```
src/secpd/
├── cli/                 # interaktive Oberfläche (`python start.py`)
├── config.py            # ENV-basierte Settings (SECPD_LLM_MODE, …)
├── splitting.py         # temporal / group / random Splits (leakage-bewusst)
├── evaluation.py        # ROC-AUC, PR-AUC, Brier, Dezil-/Lift-Tabelle
├── data/
│   ├── zenodo.py        # Streaming-Loader, AAER-Parsing, Fraud-Label
│   ├── edgar.py         # XBRL companyfacts → kanonisches Finanz-Panel
│   ├── events.py        # 8-K-Events: Default-Label (Insolvenz) + PIT-Features
│   └── synthetic.py     # Demo-/Testdaten mit echtem Signalfluss
├── features/
│   ├── financial.py     # Leverage/Liquidität/Profitabilität/Coverage/Größe
│   └── textual.py       # prepare_text + precompute-then-join der LLM-Features
├── llm/
│   ├── schema.py        # TextRiskProfile (pydantic) — zentraler Vertrag
│   ├── base.py          # BaseLLMClient (ABC) + Batch-Fallback
│   ├── mock.py          # deterministischer Heuristik-Mock (stdlib-only)
│   ├── bank.py          # OpenAI-kompatibler Gateway-Client (# ADAPT-Marker)
│   └── cache.py         # Datei-Cache (committbar ⇒ Bank↔Home „Replay")
└── models/
    ├── pipeline.py      # RF-Pipeline (Option A) + Text-Only-LogReg (Option B)
    ├── ensemble.py      # gewichtete Logit-Aggregation (Option B)
    └── persistence.py   # joblib-Bundles mit Versions-Stempeln
```

## Quickstart (Home-Setup)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# End-to-End-Demo auf synthetischen Daten (kein Download nötig):
make train-combined        # = synth + Option A mit Mock-LLM
make train-ensemble        # = Option B (Logit-Ensemble)
make test                  # pytest

# Echte Daten:
python scripts/fetch_zenodo.py --files aaer_mark5.csv firm_years_labels.json
python scripts/convert_zenodo.py --firm-years data/raw/firm_years_labels.json \
    --aaer data/raw/aaer_mark5.csv --out data/processed/zenodo_labeled.csv.gz
export SECPD_SEC_UA="Commerzbank Praktikum vorname.nachname@example.com"
python scripts/fetch_edgar_financials.py --dataset data/processed/zenodo_labeled.csv.gz \
    --out data/raw/financials_panel.csv
python train.py --data data/processed/zenodo_labeled.csv.gz \
    --financials data/raw/financials_panel.csv --mode combined --llm mock --calibrate
# schreibt z. B. models/combined_fraud.joblib + models/financial_fraud.joblib
```

## Workflow Home → Bank (air-gapped)

1. **Home:** `bash scripts/build_offline_bundle.sh` (vorher `PY_VERSION` an den
   Bank-Server anpassen; robusteste Variante via Docker, siehe Skript-Header).
2. **Home:** committen: Code, `requirements.txt`, ggf. `models/*_*.joblib`.
   Rohdaten **nicht** committen — Zenodo + EDGAR-Skripte (siehe oben).
   Optional lokal: `vendor/wheels/`, `data/cache/llm/`.
3. **Bank:** `bash scripts/install_offline.sh` (venv, `--no-index`-Install,
   Editable-Install).
4. **Bank:** ENV setzen und **identisch** trainieren/scoren:

```bash
export SECPD_LLM_MODE=bank
export SECPD_LLM_ENDPOINT=<interne Gateway-URL>       # ADAPT
export SECPD_LLM_API_KEY=<interner Key>               # nie committen!
export SECPD_LLM_MODEL=<Modellname>
python train.py --data data/processed/zenodo_labeled.csv.gz \
    --financials data/raw/financials_panel.csv --mode ensemble --calibrate
```

Der LLM-Cache (`data/cache/llm/<namespace>/*.json`) kann anschließend zurück-
committet werden — damit sind die Bank-LLM-Profile zu Hause exakt reproduzierbar.

## ENV-Variablen

| Variable              | Zweck                                   | Default |
|-----------------------|-----------------------------------------|---------|
| `SECPD_LLM_MODE`     | `mock` \| `openai` \| `lmstudio` \| `bank` | `mock`  |
| `SECPD_LLM_ENDPOINT` | Gateway-/LM-Studio-URL oder Host        | OpenAI: api.openai.com |
| `SECPD_LLM_API_KEY`  | Key (auch in `start.py` → `.secpd.env`) | —       |
| `SECPD_LLM_MODEL`    | Modellname (`gpt-5.6-luna` / `auto`)    | je nach Modus |
| `SECPD_LLM_JSON_MODE`| `1`/`0` — `response_format=json_object` | openai: `1` |
| `SECPD_LLM_TIMEOUT`  | Request-Timeout (Sekunden)              | openai: `120` |
| `SECPD_SEC_UA`       | SEC-Pflicht-User-Agent für EDGAR (Home) | —       |
| `SECPD_FMP_API_KEY`  | FMP-Key (optional, `--source fmp`)      | —       |
| `SECPD_LLM_CACHE`    | Cache-Verzeichnis (optional)            | `data/cache/llm` |

### OpenAI / ChatGPT (empfohlen für Bulk)

In `python start.py` → **Einstellungen → LLM → OpenAI** den Dev-API-Key
eintragen (wird in `.secpd.env` gespeichert, gitignored). Beim Training:
**Cache nutzen** (schnell) oder **Neu bewerten** (`--llm-refresh`).

```bash
# Oder per ENV:
export SECPD_LLM_MODE=openai
export SECPD_LLM_API_KEY=sk-...          # nie committen!
# Default-Modell: gpt-5.6-luna

make ping-llm                            # nach ENV / .secpd.env
python scripts/precompute_llm_features.py \
  --data data/processed/zenodo_labeled.csv.gz --llm openai --sample 5

# Volles Sample cachen, danach Training mit Cache:
python scripts/precompute_llm_features.py \
  --data data/processed/zenodo_labeled.csv.gz \
  --financials data/raw/financials_panel.csv \
  --min-fyear 2009 --require-financials --llm openai \
  --out data/processed/llm_features_openai.csv

python train.py --data data/processed/zenodo_labeled.csv.gz \
  --financials data/raw/financials_panel.csv \
  --events data/raw/edgar_8k_events.csv \
  --label-source default --require-financials \
  --mode combined --llm openai --llm-cache-only --calibrate
# Neu bewerten: --llm-refresh (braucht API-Key)
# Combined nutzt llm_risk_sentiment + llm_complexity_score plus
# MD&A-Keyword-Zähler (going concern, liquidity, covenant, …).
```

### LM Studio (lokales LLM)

In LM Studio den Local Server starten (Bind `0.0.0.0`, Port `1234`) und ein
Modell laden. Vom Rechner im gleichen Netz:

```bash
export SECPD_LLM_MODE=lmstudio
export SECPD_LLM_ENDPOINT=http://172.16.3.164:1234
export SECPD_LLM_MODEL=auto          # oder exakte Modell-ID aus LM Studio

make ping-llm                        # /v1/models + optional --analyze
# Smoke-Test mit 5 Docs:
python scripts/precompute_llm_features.py \
  --data data/processed/zenodo_labeled.csv.gz --llm lmstudio --sample 5

# Volles Clean-Sample cachen (wiederaufnehmbar):
make precompute-llm

# Danach Training nutzt denselben Cache:
python train.py --data data/processed/zenodo_labeled.csv.gz \
  --financials data/raw/financials_panel.csv \
  --events data/raw/edgar_8k_events.csv \
  --label-source default --require-financials \
  --mode combined --llm lmstudio --calibrate
```

Cache-Dateien landen unter `data/cache/llm/bank-<modell>/` und werden bei
Re-Runs übersprungen (auch wenn der LLM-Host offline ist — dann nur Hits).

## 8-K-Events & Default-Label (Insolvenz)

Der Zenodo-Datensatz labelt Fraud, keinen Zahlungsausfall. Für eine echte
**Ausfallwahrscheinlichkeit** liefert die EDGAR-Submissions-API das
Insolvenzsignal direkt in den Filing-Metadaten: 8-K **Item 1.03**
("Bankruptcy or Receivership"), vor dem 2004-08-23 **Item 3** (altes
Nummerierungsregime — der Code matcht beide mit Datums-Guard, an echten
Fällen verifiziert: Enron 2001 via Item 3, Lehman 2008 / Delta 2005 /
PG&E 2019 / Hertz 2020 via 1.03).

```bash
# 1) 8-K-Events laden (Home-Setup; Cache macht den Lauf wiederaufnehmbar):
export SECPD_SEC_UA="Commerzbank Praktikum vorname.nachname@example.com"
python scripts/fetch_edgar_events.py \
    --dataset data/processed/zenodo_labeled.csv.gz \
    --out data/raw/edgar_8k_events.csv

# 2) Training mit Insolvenz-Label (1-Jahres-PD) + Event-Features:
python train.py --data data/processed/zenodo_labeled.csv.gz \
    --financials data/raw/financials_panel.csv \
    --events data/raw/edgar_8k_events.csv \
    --label-source default --default-horizon 12 --calibrate
# Artefakte: models/combined_default_h12.joblib + models/financial_default_h12.joblib
# (Label-Quelle + Horizont stecken im Dateinamen — kein Überschreiben fremder Läufe)

# Demo ohne Downloads:  make train-default
```

**Label-Definition:** `label_default = 1` ⟺ die Insolvenzmeldung liegt in
`(reporting_date, reporting_date + Horizont]`. Firm-Years **nach** der
Insolvenz werden gedroppt, ebenso 10-Ks mit `filing_date > bankruptcy_date`
(post-petition: Kodak, PG&E haben den 10-K erst ~6 Wochen nach dem Antrag
eingereicht — `reporting_date` liegt davor, die MD&A enthält aber schon
Chapter-11-Prosa). Rechtszensierte Firm-Years (Horizont ragt über das
Beobachtungsende hinaus, Label 0) ebenfalls. Die Insolvenz ist
ausschließlich **Zielvariable** — Item 1.03/alt-3 existiert bewusst in
keinem Feature (Leakage-Regel, per Test abgesichert).

**Delisting ist kein Default.** Die Policy `bankruptcy,delisting` gewinnt
einen Rolling-Vergleich nur, weil Delisting-8-Ks (Item 3.01) rund 13×
häufiger sind als Insolvenz-Meldungen. Delisting umfasst freiwillige
Abgänge, Merger und Exchange-Wechsel — das ist kein Kreditereignis.
Ausgelieferte Bundles sind bewusst **bankruptcy-only**. `delisting`
bleibt ein Event-Feature (`evt_n_delisting`), nicht Teil des Labels.

**Event-Features (`evt_*`, mit `--events` in allen Modi):** PIT-saubere
Zähler über `(filing_date_10K − 365 T, filing_date_10K]` — 8-K-Frequenz,
Auditor-Wechsel (4.01/alt-4), Officer-Abgänge (5.02/alt-6),
Covenant-Brüche (2.04), Impairments (2.06), Delisting-Notices (3.01).
Item 4.02 (Restatement) ist standardmäßig aus — zu nah am Fraud-Label.

**Caveats:** Nicht jede Insolvenz erzeugt ein 8-K (Untererfassung bei
Firmen, die vorher delisten oder Filing-Pflichten verletzen); die
Basisrate liegt deutlich unter der Fraud-Basisrate ⇒ **PR-AUC als
Leitmetrik**; Chapter 11 ist der übliche Default-Proxy, nicht identisch
mit einem Zahlungsausfall im Kreditvertragssinn.

## Ratings-Label (Agentur-Note, ordinal)

Dichtes Ranking-Target: **Issuer-Ratings** aus [ratingshistory.info](https://ratingshistory.info/)
(SEC Regulation 17g-7: Moody's, Fitch, Egan-Jones). Das Modell sagt die
**ordinale Note** (Notch 1–21, AAA … D) als primäre Zielvariable vorher;
die 12-Monats-PD bleibt sekundär. Die Note ist **nur Label**, nie Feature.
SIC-Division (1-stellig, One-Hot) und Größen-Buckets aus `fin_log_assets`
sind Features des ordinalen Modells — Text ändert die MAE kaum (Ratings
hängen an Fundamentaldaten und Größe).

FMP-Fundamentalnoten bleiben optional (`--source fmp` / `both`, Free-Tier
~250 Calls/Tag, `--max-requests`). Kaggle/HuggingFace-Samples (2010–2016,
wenige Firmen) lohnen sich nicht.

```bash
# 17g-7 Bulk (kein API-Limit, Cache unter data/raw/nrsro_ratings/):
python scripts/fetch_ratings.py \
    --dataset data/processed/zenodo_labeled.csv.gz \
    --out data/raw/ratings_panel.csv

# Optional FMP ergänzen (Key in .secpd.env, nie committen):
export SECPD_FMP_API_KEY=...
python scripts/fetch_ratings.py --source both --max-requests 250

# Ordinales Shadow-Rating (Default bei --label-source rating):
python train.py --data data/processed/zenodo_labeled.csv.gz \
    --financials data/raw/financials_panel.csv \
    --events data/raw/edgar_8k_events.csv \
    --ratings data/raw/ratings_panel.csv \
    --label-source rating --rating-target ordinal \
    --require-financials --mode combined --llm-cache-only

# HY-ähnlich / Downgrade bleiben als binäre Varianten:
python train.py ... --label-source rating --rating-target speculative --calibrate
```

Artefakte: `models/combined_rating_ordinal.joblib` (+ financial-Pendant
mit derselben `train_run_id`). Metriken: **MAE (Notches), Spearman,
±1-Notch-Treffer**. Unrated Firm-Years werden gedroppt. 17g-7-Daten
kommen mit ~12 Monaten Meldeverzug; der CIK-Join läuft über LEI und
normalisierte Emittentennamen. Das 8-K-Insolvenzlabel bleibt der
Produktions-Tail für die PD.

In `start.py` zeigt die Auswertung **Rating (Modell)** und **Rating
(Agentur)** plus sekundär die 12M-PD, sobald beide Bundles existieren.

## Alle Modelle auf einmal

`train.py --mode combined` schreibt financial + combined **im selben Lauf**
(gleiche `train_run_id`). Der Sweep trainiert default-h12/h24/h36 + fraud
(+ rating, falls Panel da) hintereinander oder parallel:

```bash
python scripts/train_all.py                 # hintereinander, Full-Universum
python scripts/train_all.py --jobs 2        # zwei Subprozesse parallel
python scripts/train_all.py --dry-run
# start.py → Trainieren → Label „alle“
```

## Methodische Hinweise (bewusst dokumentiert)

* **Label-Semantik:** Der Zenodo-Datensatz liefert AAER-basierte
  **Misstatement-/Fraud-Fenster**, keine Zahlungsausfälle. Das Modell schätzt
  damit primär ein **Misconduct-/Obfuskationsrisiko** — als eigenständiger
  Score bzw. PD-Overlay/Feature interpretierbar, nicht als kalibrierte PD im
  regulatorischen Sinn. Zudem gilt PU-Learning-Logik: Negative sind „nicht
  erwischt", nicht sicher „sauber".
* **Splits:** Default `auto` = temporal (letzte Geschäftsjahre als Test),
  Fallback GroupSplit über `cik` — verhindert Look-ahead und Firm-Leakage.
  Text-Features sind pro Dokument deterministisch (kein Fitting) und dürfen
  daher vor dem Split berechnet werden. Der temporale Holdout ist **optimistisch**, wenn Krisenjahre (z. B. 2019)
  im Trainingsset liegen: der Random Forest kann exakte Finanz-Kombinationen
  memorieren (Hertz: hohe PD bei unauffälligen Text-Features). **Ehrlicher
  Maßstab ist Rolling-Origin** (`benchmarks/rolling_full_h12_bankruptcy/`):
  pooled Combined ROC **0,79**, PR-AUC **0,055**, Top-10 % fängt ~**60 %**
  der Defaults.
* **Fraud-Bundles** (`combined_fraud` / `financial_fraud`) sind **experimentell**:
  wenige Positive, oft negativer Brier-Skill. Sie stehen nicht in der
  Standardauswahl von `start.py`.
* **Kalibrierung:** `--calibrate --calibrate-method auto` (sigmoid bei
  <100 Trainings-Positiven, sonst isotonic). Vergleich h24/h36:
  `benchmarks/calibration_h24_h36.md`.
* **Finanzdaten-Abdeckung:** XBRL-companyfacts ~ab GJ 2009. Firm-Years ohne
  `total_assets` (GM 2008, Delta 2004) sind in `start.py` **nicht scorebar** —
  keine Pseudo-PD aus Median-Imputation. Training filtert sie mit
  `--require-financials` / `min_fyear=2009`.
* **joblib-Portabilität:** Bundles stempeln sklearn-/Python-Versionen und
  warnen bei Drift — deshalb `requirements.txt` auf beiden Seiten identisch
  halten. Die Pins brauchen **Python 3.11+**; auf der Bank mit nur 3.14
  (`3.14.3`/`3.14.6`) die aktuelle `requirements.txt` verwenden — ältere
  Pins (numpy 1.26 / pandas 2.2 / sklearn 1.4) haben keine 3.14-Wheels.
  Offline-Bundle: `PY_VERSION=3.14 PLATFORM=manylinux_2_28_x86_64`.
  Nach einem sklearn-Bump Modelle neu trainieren, nicht alte `.joblib` laden.

## Schnittstellen für spätere Integration

Drei stabile Schnittstellen: (1) `TextRiskProfile` (pydantic-Schema) als
Austauschformat der Textanalyse, (2) `get_llm_client()` als Injektionspunkt
für beliebige eigene LLM-Infrastruktur, (3) `.joblib`-Bundles mit
Feature-Vertrag + Metadaten für das Serving. `predict.py` zeigt den
minimalen Inferenz-Pfad.
