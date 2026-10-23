#!/usr/bin/env bash
set -euo pipefail
: "${TMOT_ROOT:?Set TMOT_ROOT to the external artifact directory}"
cd "$(dirname "$0")/.."
PYTHON_BIN="${TMOT_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
trap 'printf "pipeline_exit=%s\n" "$?"' EXIT
"$PYTHON_BIN" scripts/extract_features.py --root "$TMOT_ROOT" --datasets har uwave dailysports energy --weights "$TMOT_ROOT/models/MantisV2"
"$PYTHON_BIN" scripts/run_queue.py --root "$TMOT_ROOT" --config configs/pilot_cil.json --name pilot_cil_v1 --workers 2
"$PYTHON_BIN" scripts/run_queue.py --root "$TMOT_ROOT" --config configs/pilot_regression_mantis.json --name pilot_regression_mantis_v1 --workers 2
"$PYTHON_BIN" scripts/summarize.py --runs "$TMOT_ROOT/runs/pilot_cil_v1" --out "$TMOT_ROOT/reports/pilot_cil_v1"
"$PYTHON_BIN" scripts/summarize.py --runs "$TMOT_ROOT/runs/pilot_regression_mantis_v1" --out "$TMOT_ROOT/reports/pilot_regression_mantis_v1"
