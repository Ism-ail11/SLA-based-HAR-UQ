# Validation of this archive

Validation used Linux x86-64, Python 3.12.14, CPU PyTorch 2.2.2, TensorFlow 2.16.2, and the pinned scientific Python dependencies. See `environment.json` and the command logs in this directory.

| Check | Observed result |
|---|---|
| Automated tests | **109 passed**, no failures or skips, 16.24 seconds |
| Ruff static checks | Passed |
| Packaging | Built an installable wheel with bundled default config; independent wheel demo passed |
| Shell script syntax | All included `.sh` files passed `bash -n` |
| Python/C packet compatibility | 45 parameter combinations across 3 dimensions, included in pytest |
| Python/C integer-head compatibility | 50 random input vectors, included in pytest |
| Conformal quantiles | Conservative finite-sample thresholds over 21 sample-size/bin settings |
| Reliability sizing | 20 test settings plus the standalone analytical sweep |
| Causal policy | Altering test labels did not change predictions, packet decisions, debt, latency, or energy |
| All methods | All 14 methods exercised in pytest |
| Full scenario grid | **252/252 completed**, seed 7, 24 synthetic windows per role |
| Report generation | Read all 252 runs and generated CSV, LaTeX, PNG, PDF outputs |
| Synthetic training | Two QAT epochs on 120 synthetic windows across ten subjects, followed by evaluation |
| TFLite conversion | **38,168-byte** FlatBuffer; every tensor integer; sample inference passed |
| TFLite feature comparison | Maximum absolute difference 0.015934 or less on 16 representative training windows |
| Converted-feature path | All split features re-extracted, then evaluated with fresh calibration |
| Matched airtime | Validation-only search completed for both fixed and ordered baselines; residual mismatch recorded |
| Integer constant export | Completed for the trained synthetic checkpoint |

These are execution and numerical checks, not guarantees that software has no defects. The synthetic CNN was trained for only two epochs to exercise the pipeline; its accuracy is not a benchmark result. The full scenario check used synthetic features and one seed, not three fully trained real datasets.

Not performed: full official dataset download/training, reproduction of the manuscript's exact tables, board flashing, phone/edge deployment, BLE measurements, tensor-arena measurement, physical energy experiments, or validation on every supported OS/device. Source-specific parser fixtures were tested; unobserved archive layouts may need an explicit manifest.

For a fresh local check, run:

```bash
bash scripts/validate_release.sh
# With optional export dependencies installed:
WITH_TFLITE=1 bash scripts/validate_release.sh
```

The release validator's default demo uses 120 synthetic windows per role. The included 252-scenario validation used 24 per role to validate every branch quickly. Neither setting establishes paper-level statistical results.
