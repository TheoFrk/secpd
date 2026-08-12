.PHONY: synth train train-combined train-ensemble train-default score test bundle install-offline benchmark rolling

synth:
	python scripts/make_synthetic_data.py --n 1200 --out data/processed/synthetic.csv \
		--events-out data/processed/synthetic_events.csv

train-default: synth
	python train.py --data data/processed/synthetic.csv \
		--events data/processed/synthetic_events.csv \
		--label-source default --default-horizon 12 --split group --calibrate \
		--min-fyear 2009

train: synth
	python train.py --data data/processed/synthetic.csv --mode financial

train-combined: synth
	python train.py --data data/processed/synthetic.csv --mode combined --llm mock

train-ensemble: synth
	python train.py --data data/processed/synthetic.csv --mode ensemble --llm mock

score:
	python predict.py --model models/combined_default_h12.joblib \
		--data data/processed/synthetic.csv --out scores.csv --llm mock

benchmark:
	python scripts/freeze_benchmark.py \
		--data data/processed/zenodo_labeled.csv.gz \
		--financials data/raw/financials_panel.csv \
		--events data/raw/edgar_8k_events.csv \
		--financial-model models/financial_default_h12.joblib \
		--combined-model models/combined_default_h12.joblib \
		--out benchmarks/default_h12_clean

rolling:
	python scripts/rolling_eval.py \
		--data data/processed/zenodo_labeled.csv.gz \
		--financials data/raw/financials_panel.csv \
		--events data/raw/edgar_8k_events.csv \
		--out benchmarks/rolling_default_h12

test:
	python -m pytest -q

bundle:
	bash scripts/build_offline_bundle.sh

install-offline:
	bash scripts/install_offline.sh
