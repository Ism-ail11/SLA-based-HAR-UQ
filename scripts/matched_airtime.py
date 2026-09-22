"""Tune baseline packet sizes on validation traffic; report residual mismatch.

Test labels never enter the search. Matching is discrete and may be infeasible.
"""

import argparse
import copy
import json
from pathlib import Path
import pandas as pd
from rateless_har.cli import load_config
from rateless_har.experiment import evaluate


def matched(run, output, config, seed=7):
    out = Path(output)
    cfg = copy.deepcopy(config)
    cfg["record_policy_tuning_stream"] = True
    cfg["variant"] = "matched_airtime"
    evaluate(run, out / "ours", cfg, "ours", seed)
    target = float(pd.read_csv(out / "ours/tuning_windows.csv").airtime_ms.mean())
    records = []
    for method in ("fixed_chunk", "ordered"):
        candidates = []
        for size in (32, 40, 48, 64, 80, 96, 128, 152, 192, 256):
            trial = copy.deepcopy(cfg)
            if method == "fixed_chunk":
                trial["fixed_chunk_bytes"] = size
            else:
                trial.update(ordered_coordinates=40, ordered_packet_bytes=size)
                if size < 64:  # 24 metadata bytes + 40 coordinates
                    continue
            location = out / "tuning" / method / f"bytes_{size}"
            evaluate(run, location, trial, method, seed)
            airtime = float(pd.read_csv(location / "tuning_windows.csv").airtime_ms.mean())
            candidates.append((abs(airtime - target), size, airtime, trial))
        _, size, airtime, selected = min(candidates, key=lambda row: (row[0], row[1]))
        # The selection above uses ONLY the tuning_windows file, never test metrics.
        evaluate(run, out / method, selected, method, seed)
        records.append(
            dict(
                method=method,
                bytes=size,
                validation_airtime_ms=airtime,
                target_airtime_ms=target,
                residual_ms=airtime - target,
                selection="validation_only",
            )
        )
    pd.DataFrame(records).to_csv(out / "matching.csv", index=False)
    (out / "matching.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(matched(a.run, a.output, load_config(a.config), a.seed), indent=2))


if __name__ == "__main__":
    main()
