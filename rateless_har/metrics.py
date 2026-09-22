"""Metrics computed from window records; uncertainty is aggregated over seeds."""

import numpy as np
from scipy.stats import rankdata, t


def auroc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid AUROC inputs")
    pos, neg = labels.sum(), (~labels).sum()
    if not pos or not neg:
        return None
    return float((rankdata(scores)[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarize(rows, stride_s=1.0):
    if not rows or stride_s <= 0:
        raise ValueError("empty evaluation or invalid stride")

    def a(key):
        return np.array([r[key] for r in rows], float)

    cov, lat, energy = a("covered"), a("latency_ms"), a("energy_uj")
    ontime = lat <= a("deadline_ms") + 1e-9
    met = (cov > 0) & ontime & (energy <= a("energy_cap_uj") + 1e-9)
    result = dict(
        windows=len(rows),
        coverage=float(cov.mean()),
        coverage_at_deadline=float(((cov > 0) & ontime).mean()),
        mean_set_size=float(a("set_size").mean()),
        accuracy=float(a("correct").mean()),
        latency_median_ms=float(np.median(lat)),
        latency_p90_ms=float(np.quantile(lat, 0.9)),
        energy_mean_uj=float(energy.mean()),
        energy_p90_uj=float(np.quantile(energy, 0.9)),
        energy_per_hour_j=float(energy.mean() * 3600 / stride_s / 1e6),
        bytes_per_day=float(a("bytes_sent").mean() * 86400 / stride_s),
        airtime_per_day_s=float(a("airtime_ms").mean() * 86400 / stride_s / 1000),
        sla_met=float(met.mean()),
    )
    for metric, key in [
        ("emitted_mean", "emitted"),
        ("received_mean", "k"),
        ("deadline_success", "evidence_success"),
        ("bound_mean", "bound"),
        ("budget_feasible_fraction", "budget_feasible"),
    ]:
        result[metric] = float(a(key).mean())
    return result


def seed_interval(values):
    v = np.asarray(values, float)
    if not len(v):
        raise ValueError("empty runs")
    return dict(
        mean=float(v.mean()),
        ci95_half_width=None
        if len(v) < 2
        else float(t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))),
        n_seeds=len(v),
    )


def selective_risk(rows):
    order = np.argsort([r["uncertainty"] for r in rows], kind="stable")
    miss = 1 - np.array([r["covered"] for r in rows])[order]
    error = 1 - np.array([r["correct"] for r in rows])[order]
    count = np.arange(1, len(rows) + 1)
    return [
        dict(
            kept_fraction=float(n / len(rows)),
            set_miscoverage=float(m),
            classification_error=float(e),
        )
        for n, m, e in zip(count, miss.cumsum() / count, error.cumsum() / count)
    ]
