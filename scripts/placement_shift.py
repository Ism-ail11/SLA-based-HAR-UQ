"""Train/calibrate on one placement; evaluate all held-out subject placements."""

import argparse
from pathlib import Path
from rateless_har.cli import load_config
from rateless_har.train import train
from rateless_har.experiment import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--id-placement", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    cfg = load_config(a.config)
    cfg.update(id_placement=a.id_placement, variant="placement_shift")
    cfg["training"]["id_placement"] = a.id_placement
    out = Path(a.output)
    train(a.data, out / "run", cfg["training"], a.seed)
    for method in ("ours", "local_conformal", "ordered", "mc_dropout"):
        evaluate(out / "run", out / "experiments" / method, cfg, method, a.seed)


if __name__ == "__main__":
    main()
