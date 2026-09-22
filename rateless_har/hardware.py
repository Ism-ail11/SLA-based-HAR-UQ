"""Analyze measured current traces and application event logs, without inventing data."""

import numpy as np
import pandas as pd


def integrate_energy(trace, start_s, end_s):
    """Integrate voltage*current; trace columns time_s, voltage_v, current_a."""
    if not {"time_s", "voltage_v", "current_a"} <= set(trace):
        raise ValueError("expected time_s, voltage_v, current_a")
    t = trace.time_s.to_numpy(float)
    power = trace.voltage_v.to_numpy(float) * trace.current_a.to_numpy(float)
    if (
        len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(power).all()
        or (np.diff(t) <= 0).any()
    ):
        raise ValueError("trace must be finite and strictly increasing")
    if not t[0] <= start_s < end_s <= t[-1]:
        raise ValueError("measurement interval lies outside trace")
    ix = (t > start_s) & (t < end_s)
    sample_t = np.r_[start_s, t[ix], end_s]
    sample_p = np.r_[np.interp(start_s, t, power), power[ix], np.interp(end_s, t, power)]
    return float(np.trapz(sample_p, sample_t) * 1e6)


def analyze_windows(windows, trace=None):
    required = {"window_id", "ready_s", "prediction_s", "deadline_ms", "covered", "set_size"}
    if not required <= set(windows):
        raise ValueError(f"missing fields: {sorted(required - set(windows))}")
    if windows.window_id.duplicated().any() or (windows.deadline_ms <= 0).any():
        raise ValueError("invalid IDs/deadlines")
    result = windows.copy()
    result["latency_ms"] = (result.prediction_s - result.ready_s) * 1000
    if (result.latency_ms.dropna() < 0).any():
        raise ValueError("prediction precedes ready timestamp")
    result["on_time"] = result.latency_ms.notna() & (result.latency_ms <= result.deadline_ms + 1e-9)
    result["sla_success"] = result.on_time & result.covered.astype(bool)
    if trace is not None:
        # Integrate each measured interval, including the full wait on failed predictions.
        result["energy_uj"] = [
            integrate_energy(
                trace,
                r.ready_s,
                r.prediction_s if pd.notna(r.prediction_s) else r.ready_s + r.deadline_ms / 1000,
            )
            for r in result.itertuples()
        ]
    return result
