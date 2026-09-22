"""Scientific figures and seed-level tables from computed experiment records."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics import seed_interval


def build_report(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    runs, windows, per_k = [], [], []
    keys = [
        "dataset",
        "method",
        "regime",
        "deadline_ms",
        "alpha",
        "bins",
        "score",
        "packet_bytes",
        "source",
        "variant",
    ]
    for file in sorted(Path(root).rglob("summary.json")):
        meta = json.loads(file.read_text())
        if "metrics" not in meta or "config" not in meta:
            continue
        c = meta["config"]
        common = dict(
            dataset=c.get("dataset", "unspecified"),
            method=meta["method"],
            seed=meta["seed"],
            regime=c["regime"],
            deadline_ms=c["deadline_ms"],
            alpha=c["alpha"],
            bins=c["histogram_bins"],
            score=c["score"],
            packet_bytes=c["encoder"]["packet_bytes"],
            source=meta["source"],
            variant=c.get("variant", "main"),
        )
        runs.append({**common, **meta["metrics"]})
        if (
            c["deadline_ms"] == 200
            and c["regime"] == "moderate"
            and c.get("variant", "main") == "main"
        ):
            f = pd.read_csv(file.parent / "windows.csv")
            for k, v in common.items():
                f[k] = v
            windows.append(f)
        f = pd.read_csv(file.parent / "per_k.csv")
        for k, v in common.items():
            f[k] = v
        per_k.append(f)
    if not runs:
        raise ValueError("no experiment summaries")
    frame, counts = pd.DataFrame(runs), pd.concat(per_k)
    records = (
        pd.concat(windows)
        if windows
        else pd.DataFrame(columns=["dataset", "alpha", "deadline_ms", "regime", "variant"])
    )
    frame.to_csv(out / "runs.csv", index=False)
    metrics = [c for c in frame.select_dtypes("number") if c not in keys + ["seed"]]
    summary = []
    for values, group in frame.groupby(keys, dropna=False):
        if group.seed.duplicated().any():
            raise ValueError("duplicate seed/scenario: avoid double-counting repeated runs")
        row = dict(zip(keys, values))
        row["n_seeds"] = len(group)
        for metric in metrics:
            v = group[metric].dropna().to_numpy()
            ci = seed_interval(v) if len(v) else dict(mean=None, ci95_half_width=None)
            row[metric], row[metric + "_ci95"] = ci["mean"], ci["ci95_half_width"]
        summary.append(row)
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    table.to_latex(out / "summary.tex", index=False, float_format="%.4f", longtable=True)
    join = [k for k in keys if k != "method"] + ["seed"]
    paired = frame[frame.method != "ours"].merge(
        frame[frame.method == "ours"], on=join, suffixes=("", "_ours")
    )
    for metric in metrics:
        if metric + "_ours" in paired:
            paired[metric + "_delta"] = paired[metric] - paired[metric + "_ours"]
    paired.to_csv(out / "paired_deltas.csv", index=False)
    for (dataset, alpha), data in frame[frame.variant == "main"].groupby(["dataset", "alpha"]):
        tag = f"{dataset}_alpha_{alpha:g}"

        def finish(name, xlabel, ylabel):
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"{dataset} — simulation/replay", fontsize=9)
            plt.legend(fontsize=7)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(out / f"{tag}_{name}.pdf")
            plt.savefig(out / f"{tag}_{name}.png", dpi=160)
            plt.close()

        plt.figure(figsize=(6, 4))
        for method, g in data[data.regime == "moderate"].groupby("method"):
            v = g.groupby("deadline_ms").coverage_at_deadline.mean()
            plt.plot(v.index, v, marker="o", label=method)
        finish("coverage_deadline", "Post-window deadline (ms)", "Coverage at deadline")
        plt.figure(figsize=(7, 4))
        for method, g in data[data.deadline_ms == 200].groupby("method"):
            v = g.groupby("regime").sla_met.mean().reindex(["good", "moderate", "poor", "bursty"])
            plt.plot(v.index, v * 100, marker="o", label=method)
        finish("sla_regime", "Link regime", "Joint SLA met (%)")
        selected = records[(records.dataset == dataset) & (records.alpha == alpha)]
        if len(selected):
            plt.figure(figsize=(6, 4))
            for method, g in selected.groupby("method"):
                x = np.sort(g.latency_ms)
                plt.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=method)
            finish("latency_cdf", "Post-window latency (ms)", "Empirical CDF")
            plt.figure(figsize=(7, 4))
            energy = selected.groupby("method")[
                ["compute_energy_uj", "radio_energy_uj", "idle_energy_uj"]
            ].mean()
            bottom = np.zeros(len(energy))
            for col in energy:
                plt.bar(
                    energy.index, energy[col], bottom=bottom, label=col.replace("_energy_uj", "")
                )
                bottom += energy[col]
            plt.xticks(rotation=40, ha="right")
            finish("energy", "Method", "Modeled energy (µJ/window)")
            plt.figure(figsize=(6, 4))
            for method, g in selected.groupby("method"):
                g = g[g.seed == g.seed.min()].sort_values("window_id")
                plt.plot(g.window_id, g.covered.rolling(25, min_periods=1).mean(), label=method)
            finish("rolling_coverage", "Window index", "Rolling coverage, 25 windows")
        pk = counts[
            (counts.dataset == dataset)
            & (counts.alpha == alpha)
            & (counts.deadline_ms == 200)
            & (counts.regime == "moderate")
            & (counts.variant == "main")
        ]
        for col in ("coverage", "set_size"):
            if len(pk):
                plt.figure(figsize=(6, 4))
                for method, g in pk.groupby("method"):
                    v = g.groupby("k")[col].mean()
                    plt.plot(v.index, v, marker="o", label=method)
                finish(col + "_k", "Realized received packets", col.replace("_", " "))
    (out / "README.md").write_text(
        "# Computed experiment report\n\nAll values come from window records. Simulation energy/timing are assumptions, not hardware measurements. CIs use independent seeds; one seed has no CI. No paper values are hard-coded.\n"
    )
    return len(runs)
