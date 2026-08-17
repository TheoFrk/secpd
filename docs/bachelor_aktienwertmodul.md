# S5 — Aktienwertmodul (Bachelor-Praxis)

SEC-PD als **fünftes Szenario** neben S1 (Preis), S2 (Loughran-McDonald),
S3 (FinBERT) und S4 (News-LLM). Kein Ersatz der News-Pipeline: 10-K-PD ist
ein **langsames Fundamentals-/Distress-Signal**, News sind hochfrequent.

Die Praxis-Datenbank bleibt im iCloud-Repo
`Bachelorarbeit_Praxis` (nicht in dieses Git kopiert). Pfad:

```bash
export BACHELOR_PRAXIS_ROOT="…/Bachelorarbeit_Praxis"
```

## Mapping

| Bachelor | SEC-PD |
|---|---|
| Daily `z_s1_price_score` | — |
| News-Sentiment (S2–S4) | — |
| **S5** `z_secpd_quality` | `−z(PD)` je Firm-Year, PIT ab `filing_date+1` |
| Hybrid 50/50 | `0.5·z_s1 + 0.5·z_secpd_quality` (wie S2/S4) |

Join-Idee: Daily-Preispanel der Praxis × as-of PD (letzter 10-K vor dem Handelstag).
US-Namen (Apple, Tesla) sind Demos; STOXX-600 braucht ein CIK↔Yahoo-Mapping.

## Demo (dieses Repo)

```bash
python scripts/export_aktienwert.py \
  --data docs/demo/apple_10k.csv --ticker AAPL \
  --model models/combined_default_h36.joblib \
  --events data/raw/edgar_8k_events_full.csv \
  --out docs/demo/apple_aktienwert.csv

python scripts/export_aktienwert.py \
  --data docs/demo/tesla.json --ticker TSLA \
  --model models/combined_default_h36.joblib \
  --events data/raw/edgar_8k_events_full.csv \
  --out docs/demo/tesla_aktienwert.csv
```

In `start.py`: Datei laden → `docs/demo/apple_10k.csv` bzw. `docs/demo/tesla.json`.

## Code

| Datei | Rolle |
|---|---|
| `src/secpd/equity/overlay.py` | PD → Qualität / z / Hybrid |
| `src/secpd/equity/scoring.py` | Bundle-Inferenz |
| `scripts/export_aktienwert.py` | Export-Panel |
| `config/s5_aktienwert.yaml` | Gewichte und Praxis-Pfade |
| `tests/test_equity_overlay.py` | Overlay-Tests |

Konfig: `config/s5_aktienwert.yaml`.
