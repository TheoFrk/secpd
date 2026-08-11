.PHONY: synth train train-combined train-ensemble train-default score test bundle install-offline

synth:
	python scripts/make_synthetic_data.py --n 1200 --out data/processed/synthetic.csv \
		--events-out data/processed/synthetic_events.csv

train-default: synth
	python train.py --data data/processed/synthetic.csv \
		--events data/processed/synthetic_events.csv \
		--label-source default --default-horizon 12 --split group --calibrate

train: synth
	python train.py --data data/processed/synthetic.csv --mode financial

train-combined: synth
	python train.py --data data/processed/synthetic.csv --mode combined --llm mock

train-ensemble: synth
	python train.py --data data/processed/synthetic.csv --mode ensemble --llm mock

score:
	python predict.py --model models/combined_default_h12.joblib \
		--data data/processed/synthetic.csv --out scores.csv --llm mock

test:
	python -m pytest -q

bundle:
	bash scripts/build_offline_bundle.sh

install-offline:
	bash scripts/install_offline.sh
