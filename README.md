# SLA-aware rateless uncertainty packetization for TinyML HAR

A documented **reference reimplementation** of the method described in *SLA-Aware Rateless Uncertainty Packetization for TinyML-Based Human Activity Recognition at the IoT Edge*.

This repository implements data preparation, quantization-aware CNN training, integer classification, sparse signed rateless packet encoding, peeling, conformal calibration, causal SLA control, link simulation/replay, baselines, ablations, sensitivity studies, plots, and measurement-log analysis. It includes a portable C encoder/head and an optional all-integer TFLite exporter.

**Scientific status:** this is not the authors' original executable artifact. Original subject lists, checkpoints, packet traces, board firmware, measurement logs, and some implementation choices were not supplied. Consequently, the paper's exact numerical tables and hardware claims remain unverified. `docs/REPRODUCIBILITY.md` explains the choices, inconsistencies, and limits explicitly. No paper results are hard-coded; synthetic checks are labeled synthetic.

## Install

Tested on Linux CPU with Python 3.12; supported Python range is 3.10–3.12. A C compiler is needed for Python/C parity tests. Run commands from this repository's root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU PyTorch avoids installing an unnecessary CUDA runtime on Linux.
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
bash scripts/run_tests.sh
bash scripts/run_smoke.sh
```

On Windows, use WSL for the shell runners, or run the equivalent Python commands individually. On other platforms, install an appropriate PyTorch 2.2.2 build. A clean environment is recommended; NumPy is deliberately pinned below version 2 for this PyTorch release.

The smoke script creates synthetic features, trains for two epochs on synthetic windows, evaluates the controller, generates example figures, and checks the reliability calculations. It is **not** an experiment on WISDM, PAMAP2, or RealWorld. Fresh validation evidence is in `reports/VALIDATION.md`.

## Run the datasets

See [data preparation](docs/DATA.md) for official sources, exact sensor choices, archive extraction, RealWorld manifests, and label mappings. The raw datasets are not bundled.

```bash
python scripts/download_data.py wisdm --extract --nested
python -m rateless_har prepare --dataset wisdm \
  --input data/raw/wisdm/extracted --output data/prepared/wisdm.npz
bash scripts/reproduce_dataset.sh wisdm data/prepared/wisdm.npz
```

Once all three prepared datasets exist:

```bash
bash scripts/reproduce_all.sh
```

The default runner trains seeds **7, 19, 42** and runs **252 configurations per seed**: 224 main comparisons, 12 ablations/debt settings, and 16 packet/score/histogram sensitivities. Across three datasets this is 2,268 scenario runs, plus matched-airtime studies. Model training and real-dataset runs can take substantial time. No automatic GPU, cloud account, or paid service is required.

To inspect a single run:

```bash
python -m rateless_har evaluate --run outputs/wisdm/training/seed_7 \
  --config configs/wisdm.json --method ours --seed 7 --output outputs/single
python -m rateless_har report --input outputs/wisdm/experiments --output outputs/wisdm/report
```

Every evaluation writes window-level CSV records, calibration histograms, per-class/per-packet metrics, selective-risk curves, a policy-tuning curve, and a JSON summary with the complete configuration and feature-file SHA-256. The reporter produces CSV/LaTeX tables and PNG/PDF figures. A limited grid is visibly marked as limited in `grid_manifest.json`.

## Repository map

| Location | Purpose |
|---|---|
| `rateless_har/data.py` | Dataset adapters, subject splits, gap-safe windows, normalization |
| `model.py`, `train.py` | QAT CNN, masked-feature training, temperature scaling, feature export |
| `codec.py`, `integer.py` | Wire format, sparse projections, peeling, integer head and softmax |
| `conformal.py`, `controller.py` | Histograms, finite-sample quantiles, budget sizing and debt |
| `link.py`, `experiment.py` | Simulated/replayed links and experiment execution |
| `metrics.py`, `report.py` | Metrics, seed-level intervals, tables and figures |
| `scripts/` | Shell reproduction runners, downloads, export and supplemental experiments |
| `embedded/` | Portable C encoder/head; not board-specific firmware |
| `tests/` | Numerical, parser, causality, integration and C parity tests |
| `configs/` | Explicit assumptions and dataset defaults |
| `docs/` | Protocol, reproduction boundaries, experiments, hardware, GitHub steps |
| `reports/` | Validation performed for this archive |

## Integer export and hardware

```bash
python scripts/export_integer.py --run outputs/synthetic/run --output outputs/export/model_constants.h
python -m pip install -r requirements-export.txt
python scripts/export_tflite.py --run outputs/synthetic/run \
  --data outputs/synthetic/windows.npz --output outputs/export
python scripts/extract_tflite_features.py --run outputs/synthetic/run \
  --data outputs/synthetic/windows.npz --export outputs/export --output outputs/tflite_run
python -m rateless_har evaluate --run outputs/tflite_run --output outputs/tflite_evaluation
```

TFLite conversion changes activation quantization. The converted model therefore needs feature re-extraction and fresh conformal calibration. The exporter uses only training subjects as representatives. See [hardware scope](docs/HARDWARE.md) for current-trace analysis, packet replay, firmware integration, and measurements that cannot be inferred from software tests.

## Publish on GitHub

See [GitHub instructions](docs/GITHUB.md). Choose a license, complete `CITATION.cff.example` (rename to `CITATION.cff` after completing it), and review scientific caveats before publication. Dataset licenses and attribution requirements remain those of their providers. No license or author identity is invented for you.
