"""Command line tools; run python -m rateless_har --help."""

import argparse
import copy
import itertools
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not 0 < c["alpha"] < 1 or c["deadline_ms"] <= 0 or c["max_k"] < 1:
        raise ValueError("invalid alpha/deadline/packet count")
    return c


def grid_jobs(config, suite):
    jobs = []
    if suite in ("main", "all"):
        for method, regime, deadline, alpha in itertools.product(
            [
                "ours",
                "local_only",
                "local_conformal",
                "full_offload",
                "fixed_chunk",
                "ordered",
                "mc_dropout",
            ],
            ["good", "moderate", "poor", "bursty"],
            [100, 150, 200, 300],
            [0.1, 0.05],
        ):
            c = copy.deepcopy(config)
            c.update(regime=regime, deadline_ms=deadline, alpha=alpha, variant="main")
            jobs.append((f"main/{method}/{regime}/d{deadline}_a{alpha}", method, c))
    if suite in ("ablations", "all"):
        for method in [
            "ours",
            "fixed_chunk",
            "ordered",
            "pooled",
            "no_debt",
            "float_head",
            "phone_only",
            "edge_assist",
            "balanced_degree",
            "direct_margin",
        ]:
            c = copy.deepcopy(config)
            c.update(regime="moderate", deadline_ms=200, variant="ablation")
            jobs.append((f"ablations/{method}", method, c))
        for method in ["ours", "no_debt"]:
            c = copy.deepcopy(config)
            c.update(regime="bursty", variant="debt_bursty")
            jobs.append((f"debt_bursty/{method}", method, c))
    if suite in ("sensitivity", "all"):
        for key, values in [
            ("histogram_bins", [128, 256, 512]),
            ("score", ["margin", "probability", "entropy", "energy"]),
        ]:
            for value in values:
                c = copy.deepcopy(config)
                c[key], c["variant"] = value, f"{key}_{value}"
                jobs.append((f"sensitivity/{key}_{value}", "ours", c))
        for key, values in [
            ("packet_bytes", [48, 64, 96]),
            ("projections", [1, 2, 4]),
            ("cap", [8, 16, 32]),
        ]:
            for value in values:
                c = copy.deepcopy(config)
                c["encoder"][key], c["variant"] = value, f"{key}_{value}"
                jobs.append((f"sensitivity/{key}_{value}", "ours", c))
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--dataset", choices=["wisdm", "pamap2", "realworld", "manifest"], required=True
    )
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--placement")
    prepare.add_argument("--hz", type=int, default=50)
    prepare.add_argument("--window", type=float, default=2.0)
    prepare.add_argument("--stride", type=float, default=1.0)
    prepare.add_argument("--label-map")
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--config", default=str(DEFAULT_CONFIG))
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--epochs", type=int)
    tr.add_argument("--device")
    tr.add_argument("--split")
    for name in ["evaluate", "grid"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
        cmd.add_argument("--seed", type=int, default=7)
        if name == "evaluate":
            cmd.add_argument("--method", default="ours")
            cmd.add_argument("--trace")
        else:
            cmd.add_argument(
                "--suite", choices=["main", "ablations", "sensitivity", "all"], default="all"
            )
            cmd.add_argument("--limit", type=int)
    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--windows", type=int, default=120)
    synth = sub.add_parser("synthetic-windows")
    synth.add_argument("--output", required=True)
    synth.add_argument("--per-subject", type=int, default=24)
    a = p.parse_args(argv)
    if a.command == "prepare":
        from .data import read_wisdm, read_pamap2, read_manifest, build_windows

        raw = Path(a.input)
        if a.dataset == "wisdm":
            placement = a.placement or "phone"
            files = (
                [raw]
                if raw.is_file()
                else [f for f in sorted(raw.rglob("*_accel.txt")) if placement in f.parts]
            )
            recordings = (r for f in files for r in read_wisdm(f, placement))
        elif a.dataset == "pamap2":
            files = [raw] if raw.is_file() else sorted(raw.rglob("subject*.dat"))
            if raw.is_dir() and any("Protocol" in f.parts for f in files):
                files = [f for f in files if "Protocol" in f.parts]
            recordings = (r for f in files for r in read_pamap2(f, a.placement or "hand"))
        else:
            recordings = read_manifest(raw)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        mapping = json.loads(Path(a.label_map).read_text()) if a.label_map else None
        info = build_windows(
            recordings,
            a.output,
            target_hz=a.hz,
            window_s=a.window,
            stride_s=a.stride,
            label_map=mapping,
            excluded_labels=("0",) if a.dataset == "pamap2" and mapping is None else (),
        )
        Path(a.output + ".json").write_text(json.dumps({**vars(a), **info}, indent=2) + "\n")
        print(json.dumps(info))
    elif a.command == "train":
        from .train import train

        c = load_config(a.config)["training"]
        if a.epochs is not None:
            c["epochs"] = a.epochs
        if a.device is not None:
            c["device"] = a.device
        train(
            a.data, a.output, c, a.seed, json.loads(Path(a.split).read_text()) if a.split else None
        )
    elif a.command == "evaluate":
        from .experiment import evaluate

        print(
            json.dumps(
                evaluate(a.run, a.output, load_config(a.config), a.method, a.seed, a.trace)[
                    "metrics"
                ],
                indent=2,
            )
        )
    elif a.command == "grid":
        from .experiment import evaluate

        jobs = grid_jobs(load_config(a.config), a.suite)
        selected = jobs if a.limit is None else jobs[: a.limit]
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "grid_manifest.json").write_text(
            json.dumps(
                dict(
                    seed=a.seed,
                    planned=len(jobs),
                    executed=len(selected),
                    limited=a.limit is not None,
                    jobs=[j[0] for j in selected],
                ),
                indent=2,
            )
            + "\n"
        )
        for i, (name, method, c) in enumerate(selected):
            print(f"[{i + 1}/{len(selected)}] {name}", flush=True)
            evaluate(a.run, out / name, c, method, a.seed)
    elif a.command == "report":
        from .report import build_report

        print(f"Rendered {build_report(a.input, a.output)} runs")
    elif a.command == "demo":
        from .demo import make_features
        from .experiment import evaluate
        from .report import build_report

        out = Path(a.output)
        make_features(out / "fixture", per_role=a.windows)
        c = load_config(DEFAULT_CONFIG)
        c["dataset"] = "SYNTHETIC_SMOKE_TEST"
        for method in ["ours", "local_conformal", "fixed_chunk", "ordered", "no_debt"]:
            evaluate(out / "fixture", out / "experiments" / method, c, method)
        build_report(out / "experiments", out / "report")
        print(f"Synthetic smoke run complete: {out}")
    elif a.command == "synthetic-windows":
        from .demo import make_windows

        make_windows(a.output, windows_per_subject=a.per_subject)
