#!/usr/bin/env bash
# All three datasets must already have been prepared; see docs/DATA.md.
set -euo pipefail
cd "$(dirname "$0")/.."
for dataset in wisdm pamap2 realworld; do
  bash scripts/reproduce_dataset.sh "$dataset" "data/prepared/$dataset.npz"
done
python scripts/check_theory.py --output outputs/theory
