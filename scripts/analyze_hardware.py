import argparse
import json
from pathlib import Path
import pandas as pd
from rateless_har.hardware import analyze_windows


def main():
    p = argparse.ArgumentParser(
        description="Analyze real measurement logs with shared-clock timestamps"
    )
    p.add_argument("--windows", required=True)
    p.add_argument("--current")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    frame = analyze_windows(pd.read_csv(a.windows), pd.read_csv(a.current) if a.current else None)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "measured_windows.csv", index=False)
    result = dict(
        source="user_supplied_measurement_logs",
        n=len(frame),
        coverage=float(frame.covered.mean()),
        sla_success=float(frame.sla_success.mean()),
        latency_p90_ms=None
        if frame.latency_ms.dropna().empty
        else float(frame.latency_ms.quantile(0.9)),
    )
    if "energy_uj" in frame:
        result["mean_energy_uj"] = float(frame.energy_uj.mean())
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
