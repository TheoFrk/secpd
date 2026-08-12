#!/usr/bin/env bash
# =============================================================================
# Offline-Bundle für den air-gapped Bank-Server (GCP Workstation, Linux x86_64).
# Läuft auf dem Home-Setup MIT Internet; Ergebnis wird via Git committet.
#
# WICHTIG — ABI-Match:
#   PY_VERSION muss der Python-Version des BANK-Servers entsprechen
#   (dort einmalig `python3 -V` prüfen). Kompilierte Wheels (numpy, pandas,
#   sklearn, pydantic-core) sind an Python-Minor-Version + Plattform gebunden.
#
# Robusteste Variante (garantiert passende ABI, umgeht Windows/macOS-Probleme):
#   docker run --rm -v "$PWD":/w -w /w python:3.11-slim \
#       bash scripts/build_offline_bundle.sh
# =============================================================================
set -euo pipefail

PY_VERSION="${PY_VERSION:-3.11}"
PLATFORM="${PLATFORM:-manylinux2014_x86_64}"   # Alternative: manylinux_2_28_x86_64
WHEEL_DIR="vendor/wheels"

mkdir -p "$WHEEL_DIR"

echo "==> 1/2 Wheels für Python ${PY_VERSION} / ${PLATFORM} nach ${WHEEL_DIR}"
python -m pip download -r requirements.txt -d "$WHEEL_DIR" \
    --only-binary=:all: \
    --platform "$PLATFORM" \
    --python-version "$PY_VERSION" \
    --implementation cp

echo "==> 2/2 Build-Tooling mitbündeln (pip/setuptools/wheel für die Offline-Box)"
python -m pip download pip setuptools wheel -d "$WHEEL_DIR" \
    --only-binary=:all: \
    --platform "$PLATFORM" \
    --python-version "$PY_VERSION" \
    --implementation cp

echo
echo "Bundle fertig:"
du -sh "$WHEEL_DIR" || true
echo "Jetzt committen: git add ${WHEEL_DIR} && git commit -m 'offline bundle'"
echo "(Bei Repo-Größenlimits: git lfs track '${WHEEL_DIR}/*.whl')"
