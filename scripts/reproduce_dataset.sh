#!/usr/bin/env bash
# Usage: bash scripts/reproduce_dataset.sh DATASET PREPARED_NPZ [CONFIG_JSON]
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:?dataset required: wisdm, pamap2, or realworld}"
prepared="${2:?prepared NPZ required}"
config="${3:-configs/$dataset.json}"
for seed in 7 19 42; do
  run="outputs/$dataset/training/seed_$seed"
  python -m rateless_har train --data "$prepared" --config "$config" --seed "$seed" --output "$run"
  python -m rateless_har grid --run "$run" --config "$config" --seed "$seed" --suite all --output "outputs/$dataset/experiments/seed_$seed"
  python scripts/matched_airtime.py --run "$run" --config "$config" --seed "$seed" --output "outputs/$dataset/matched/seed_$seed"
done
python -m rateless_har report --input "outputs/$dataset/experiments" --output "outputs/$dataset/report"
