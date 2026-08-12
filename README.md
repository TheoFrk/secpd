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
  --mode combined --llm openai --calibrate
# Neu bewerten: zusätzlich --llm-refresh
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
Insolvenz werden gedroppt; rechtszensierte Firm-Years (Horizont ragt über
das Beobachtungsende hinaus, Label 0) ebenfalls. Die Insolvenz ist
ausschließlich **Zielvariable** — Item 1.03/alt-3 existiert bewusst in
keinem Feature (Leakage-Regel, per Test abgesichert).

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
  daher vor dem Split berechnet werden.
* **Kalibrierung:** `--calibrate` (isotonic, CV=3) für sinnvolle
  Wahrscheinlichkeitsniveaus; Brier-Score wird immer mitreportet.
* **Finanzdaten-Abdeckung:** XBRL-companyfacts ~ab GJ 2009; ältere Firm-Years
  laufen über die Median-Imputation (oder vorab filtern).
* **joblib-Portabilität:** Bundles stempeln sklearn-/Python-Versionen und
  warnen bei Drift — deshalb `requirements.txt` auf beiden Seiten identisch
  halten.

## Schnittstellen für spätere Integration

Drei stabile Schnittstellen: (1) `TextRiskProfile` (pydantic-Schema) als
Austauschformat der Textanalyse, (2) `get_llm_client()` als Injektionspunkt
für beliebige eigene LLM-Infrastruktur, (3) `.joblib`-Bundles mit
Feature-Vertrag + Metadaten für das Serving. `predict.py` zeigt den
minimalen Inferenz-Pfad.
