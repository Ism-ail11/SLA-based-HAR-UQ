# Datasets and exact preparation choices

Download only from the official providers and follow their terms. The downloader records URL and SHA-256; an observed hash documents provenance, while `--sha256` additionally verifies a known expected hash. Archive extraction rejects traversal, links, and special files. Large archives require sufficient local disk space; nested extraction is explicit.

| Dataset | Official source | Default input |
|---|---|---|
| WISDM | https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset | Phone accelerometer; 3 channels, native 20 Hz |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | Protocol subset, hand accelerometer ±16g and gyroscope; 6 channels, native 100 Hz |
| RealWorld | https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/ | Paired accelerometer/gyroscope, chest by default; 6 channels, native about 50 Hz |

WISDM is ambiguous in the manuscript: its accelerometer description does not uniquely determine the older WISDM collection versus UCI 507. This implementation uses the cited UCI 507 phone accelerometer files. Do not compare its results to a different WISDM release without changing and recording the adapter.

```bash
python scripts/download_data.py wisdm --extract --nested
python scripts/download_data.py pamap2 --extract --nested
python scripts/download_data.py realworld --extract --nested

python -m rateless_har prepare --dataset wisdm --input data/raw/wisdm/extracted \
  --placement phone --output data/prepared/wisdm.npz
python -m rateless_har prepare --dataset pamap2 --input data/raw/pamap2/extracted \
  --placement hand --output data/prepared/pamap2.npz
python scripts/make_realworld_manifest.py --input data/raw/realworld/extracted \
  --placement chest --output data/raw/realworld/manifest.json
python -m rateless_har prepare --dataset realworld --input data/raw/realworld/manifest.json \
  --output data/prepared/realworld.npz
```

RealWorld extraction layouts vary. The manifest helper rejects missing/ambiguous sensor pairs instead of guessing. Review its output. If a layout is unsupported, construct an explicit manifest:

```json
[
  {
    "subject": "1",
    "activity": "walking",
    "placement": "chest",
    "session": "walking_trial_1",
    "acc": "relative/path/acc_walking_chest.csv",
    "gyro": "relative/path/Gyr_walking_chest.csv"
  }
]
```

CSV sensor columns may be `attr_time,attr_x,attr_y,attr_z` or `timestamp,x,y,z`. Time is milliseconds. Pairing uses the overlapping time interval and interpolates the gyro onto accelerometer timestamps. Gyro gaps above 250 ms become invalid values and break windows. WISDM timestamps are nanoseconds; PAMAP2 timestamps are seconds. PAMAP2 label 0 is excluded by default.

For another layout, use a normalized CSV manifest with `csv`, `subject`, `session`, `placement`, `source_hz`, and optional `channels`. CSV columns are `timestamp_s,label,ax,ay,az,gx,gy,gz`. Use `--dataset manifest`.

Preparation regularizes each uninterrupted segment, resamples with a polyphase filter, and produces 2 s windows with 1 s stride at 50 Hz. Windows require at least 80% label purity. Invalid values, gaps above 250 ms, clock resets, and recording boundaries break segments. Interpolation is offline; acquisition/streaming filter latency is not measured by this pipeline. Data-rate and stride changes require retraining:

```bash
bash scripts/train_sensitivity.sh pamap2 data/raw/pamap2/extracted
```

Native class vocabularies are retained. The paper does not provide a complete cross-dataset harmonized label map. Supply a JSON map with every native label using `--label-map mapping.json` when a specific taxonomy is required. Mapping changes are part of the experiment specification; do not silently collapse activities.

Subjects, not overlapping windows, define splits. The seeded outer split reserves approximately 20% test and 10% calibration subjects; validation subjects come from the remaining training pool. The original exact subject IDs were not supplied. `train --split splits.json` accepts complete disjoint `train`, `val`, `cal`, `test` subject lists. Training subjects must contain every class. Normalization uses training windows only; early stopping and temperature scaling use validation only. The final calibration split is untouched by model fitting.

Prepared NPZ keys: `x[N,C,T]` float32, `y[N]` int64, `subject`, `session`, `placement`, `start_s`, `classes`, `sample_hz`. Bundles contain no pickle objects. Real-dataset preparation has parser fixture tests; full dataset downloads/training were not run to validate this archive.
