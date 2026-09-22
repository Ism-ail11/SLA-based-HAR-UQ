#!/usr/bin/env bash
# Complete synthetic validation; WITH_TFLITE=1 also checks optional conversion.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_tests.sh
bash scripts/run_smoke.sh
python -m rateless_har grid --run outputs/demo/fixture --suite all --output outputs/validation_grid
python -m rateless_har report --input outputs/validation_grid --output outputs/validation_report
python scripts/matched_airtime.py --run outputs/demo/fixture --output outputs/matched
python scripts/export_integer.py --run outputs/synthetic/run --output outputs/export/model_constants.h
if [[ "${WITH_TFLITE:-0}" == "1" ]]; then
  python scripts/export_tflite.py --run outputs/synthetic/run --data outputs/synthetic/windows.npz --output outputs/export
  python scripts/extract_tflite_features.py --run outputs/synthetic/run --data outputs/synthetic/windows.npz --export outputs/export --output outputs/tflite_run
  python -m rateless_har evaluate --run outputs/tflite_run --output outputs/tflite_evaluation
fi
