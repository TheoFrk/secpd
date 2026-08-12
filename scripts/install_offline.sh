#!/usr/bin/env bash
# =============================================================================
# Installation auf dem air-gapped Bank-Server (GCP Workstation) — KEIN Internet.
# Voraussetzung: Repo inkl. vendor/wheels ist ausgecheckt und die
# Python-Version entspricht dem PY_VERSION des Bundles.
# =============================================================================
set -euo pipefail

echo "==> Python: $(python3 -V)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Build-Tooling aus dem lokalen Wheel-Verzeichnis"
pip install --no-index --find-links vendor/wheels --upgrade pip setuptools wheel

echo "==> Abhängigkeiten offline installieren"
pip install --no-index --find-links vendor/wheels -r requirements.txt

echo "==> Paket als Editable-Install (ohne Build-Isolation, offline-fähig)"
pip install --no-index --no-build-isolation -e .

python -c "import secpd, sklearn, pandas, pydantic; print('secpd', secpd.__version__, '| sklearn', sklearn.__version__, '| OK')"

cat <<'MSG'

Fertig. Auf dem Bank-Server zusätzlich setzen:
  export SECPD_LLM_MODE=bank
  export SECPD_LLM_ENDPOINT=<interne Gateway-URL>
  export SECPD_LLM_API_KEY=<interner Key>
  export SECPD_LLM_MODEL=<Modellname>
MSG
