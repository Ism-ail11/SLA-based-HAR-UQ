# Included runnable scripts

The main Python entry point is `python -m rateless_har`. Its subcommands are `prepare`, `train`, `evaluate`, `grid`, `report`, `demo`, and `synthetic-windows`. Use `--help` after any subcommand.

| Script | Function |
|---|---|
| `run_tests.sh` | Static checks and automated tests |
| `run_smoke.sh` | Synthetic demo, two-epoch CNN training, evaluation and theory checks |
| `validate_release.sh` | Full synthetic scenario grid and optional TFLite validation |
| `reproduce_dataset.sh` | Three-seed training, grids and matched-airtime study for one prepared dataset |
| `reproduce_all.sh` | Run all three prepared datasets |
| `train_sensitivity.sh` | Reprepare/retrain for four sampling-rate/stride choices |
| `download_data.py` | Official-source download, SHA-256 recording and safe extraction |
| `make_realworld_manifest.py` | Strict accelerometer/gyroscope pairing |
| `write_config.py` | Create explicit config variations from dotted JSON assignments |
| `matched_airtime.py` | Validation-only baseline packet-size search |
| `placement_shift.py` | ID-placement training/calibration and held-out placement evaluation |
| `check_theory.py` | Reliability checks, rank limits and monotonicity counterexample |
| `export_integer.py` | C arrays for the trained heads and integer lookup tables |
| `export_tflite.py` | Convert a trained checkpoint to an integer TFLite model |
| `extract_tflite_features.py` | Re-extract converted-model features before fresh calibration |
| `capture_serial.py` | Capture board JSON lines with host timestamps |
| `analyze_hardware.py` | Analyze synchronized prediction/current measurements |

Run Python scripts from the repository root after the editable installation. Shell scripts enter the repository root themselves; arguments are interpreted relative to that root. No script uploads results or pushes to GitHub.
