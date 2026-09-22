"""Create a config variation without silently mutating a published config."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="configs/default.json")
    p.add_argument("--output", required=True)
    p.add_argument("--set", action="append", default=[], metavar="DOTTED.KEY=JSON_VALUE")
    a = p.parse_args()
    cfg = json.loads(Path(a.base).read_text())
    for item in a.set:
        key, value = item.split("=", 1)
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = obj[part]
        obj[parts[-1]] = json.loads(value)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
