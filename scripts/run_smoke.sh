#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m rateless_har demo --output outputs/demo --windows 120
python -m rateless_har synthetic-windows --output outputs/synthetic/windows.npz --per-subject 12
python -m rateless_har train --data outputs/synthetic/windows.npz --output outputs/synthetic/run --epochs 2
python -m rateless_har evaluate --run outputs/synthetic/run --output outputs/synthetic/evaluation
python scripts/check_theory.py --output outputs/theory
