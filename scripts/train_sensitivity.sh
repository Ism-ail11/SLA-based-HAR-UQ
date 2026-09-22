#!/usr/bin/env bash
# Data-rate and stride changes require window preparation AND model retraining.
# Usage: bash scripts/train_sensitivity.sh DATASET RAW_PATH_OR_MANIFEST
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:?dataset required}"
raw="${2:?raw path or RealWorld manifest required}"
for hz in 50 100; do
  for stride in 1.0 0.5; do
    name="${dataset}_hz${hz}_stride${stride}"
    config="outputs/configs/$name.json"
    python scripts/write_config.py --base "configs/$dataset.json" --output "$config" --set "stride_s=$stride"
    python -m rateless_har prepare --dataset "$dataset" --input "$raw" --hz "$hz" --stride "$stride" --output "data/prepared/$name.npz"
    bash scripts/reproduce_dataset.sh "$name" "data/prepared/$name.npz" "$config"
  done
done
