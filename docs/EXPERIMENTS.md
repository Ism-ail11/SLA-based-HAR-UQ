# Experiment map

All configurations generate their own calibration and metrics. No numbers are copied from the manuscript.

| Study | Invocation | Scope |
|---|---|---|
| Main comparisons | `python -m rateless_har grid ... --suite main` | 7 methods × 4 regimes × 4 deadlines × 2 alpha = 224 scenarios |
| Ablations | `grid ... --suite ablations` | 10 moderate-link variants + ours/no-debt on bursty = 12 |
| Score/histogram/packet sensitivity | `grid ... --suite sensitivity` | 3 bin counts + 4 scores + 3 byte sizes + 3 projection counts + 3 degree caps = 16 |
| Full software grid | `grid ... --suite all` | All 252 configurations |
| Three seeds | `bash scripts/reproduce_dataset.sh DATASET PREPARED_NPZ` | Seeds 7, 19, 42; training and grid, plus matched-airtime study |
| Three datasets | `bash scripts/reproduce_all.sh` | WISDM, PAMAP2, RealWorld |
| Sampling/stride sensitivity | `bash scripts/train_sensitivity.sh DATASET RAW_INPUT` | 50/100 Hz × 1.0/0.5 s stride; reprepare and retrain |
| Matched airtime | `python scripts/matched_airtime.py --run RUN --config CONFIG --output OUT` | Baseline size search on validation traffic only |
| Placement shift | `python scripts/placement_shift.py --data PREPARED --id-placement chest --output OUT` | Train/calibrate one placement; held-out subjects with all supplied placements |
| Analytical checks | `python scripts/check_theory.py --output OUT` | Chernoff vs binomial, conservative integer sizing, rank and monotonicity counterexample |
| Recorded transport | `evaluate ... --trace timing.csv` | Strict replay of supplied application opportunities |
| Measured energy/latency | `python scripts/analyze_hardware.py --windows windows.csv --current current.csv --output OUT` | Integrates user-supplied synchronized measurement logs |

Replace ellipses above with `--run RUN --config CONFIG --seed 7 --output OUT`. All CLI commands have `--help`.

## Methods

- `ours`: sparse rateless projections, masked-QAT features, per-realized-k conformal calibration, integer head, causal debt.
- `local_only`: local full-feature argmax, no radio and no conformal set.
- `local_conformal`: local full-feature conformal set.
- `full_offload`: one full quantized feature-vector frame, singleton output if delivered.
- `fixed_chunk`: one importance-ranked fixed subset; conformal set over its reconstruction.
- `ordered`: importance-ranked disjoint feature chunks, with a bounded ACK-driven schedule.
- `mc_dropout`: full-feature offload followed by 20 passes through a separately trained dropout head, with validation-selected confidence rejection.
- `pooled`: one calibration histogram across realized packet counts.
- `no_debt`: the same policy with uncertainty debt disabled.
- `float_head`: float head/softmax, same int8 reconstructed features.
- `phone_only`, `edge_assist`: configured processing-delay scenarios; no actual phone/edge runtime.
- `balanced_degree`: lower singleton/degree-two masses (.10 each), more tail mass.
- `direct_margin`: auxiliary learned margin confidence converted to explicit class scores.

MC dropout is a software baseline; its cost is not measured on target hardware. Adapt the configured compute costs before making hardware energy comparisons. `fixed_chunk` and full offload retain their one-transmission definitions even if the controller's sufficient reliability budget would require more transmissions; budget infeasibility must not be read as a measured baseline reliability certificate.

Matched-airtime search tunes frame size from a discrete candidate list using only the recorded validation stream. Ordered chunks retain 40 coordinates and may add padding. It reports the actual residual rather than claiming exact equality. Candidate test outputs are generated but are never read by the selector; scientific selection is exclusively validation-based. Matching may be impossible, especially when ours transmits zero frames.

## Outputs and interpretation

`windows.csv` includes true label, predicted set, realized and offered evidence counts, packet bytes, resolved coordinates, latency, energy components, debt before observation, bound, budget feasibility/reason, and source. `summary.json` records the entire config, seed, feature checksum and calibration group counts. `calibration.npz` stores counts and quantile settings. `policy_tuning_curve.csv`, `per_k.csv`, `per_class.csv`, and `selective_risk.csv` retain the statistics behind plots.

`coverage` is empirical set coverage. `coverage_at_deadline` also requires on-time output. `sla_met` additionally requires modeled energy within its cap; it does not require singleton output, so always inspect mean set size and target feasibility alongside it. `deadline_success` means the selected evidence target was met; it is false when no validation target existed. Classification accuracy of a zero-feature fallback may be defined even when an offload baseline abstains; prediction-set coverage and acceptance are the relevant abstention metrics.

Rates per day/hour assume one window per configured stride and scale average modeled cost. They are projections, not continuous-day measurements. Confidence intervals are Student-t over seeds, with no interval for one seed. The report retains scenario details in `runs.csv`, aggregated `summary.csv/.tex`, and paired deltas. PNG and vector PDF figures cover deadlines, regimes, latency distributions, energy breakdown, rolling coverage, and realized-k coverage/set sizes.

For publication, execute real data and attach the full output directories or an archival link. The archive's included reports validate code execution on synthetic fixtures, not the manuscript's benchmark tables.
