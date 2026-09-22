"""Pair RealWorld CSVs by subject, activity, placement and acquisition suffix.

Review the generated manifest before using it. Ambiguous pairings are rejected.
"""

import argparse
import json
import re
from pathlib import Path


def build_manifest(root, placement="chest"):
    entries = {}
    for path in sorted(Path(root).resolve().rglob("*.csv")):
        name = path.stem.lower()
        match = re.fullmatch(
            r"(acc|gyr|gyro)_(climbingdown|climbingup|jumping|lying|running|sitting|standing|walking)_(chest|forearm|head|shin|thigh|upperarm|waist)(.*)",
            name,
        )
        if match is None:
            continue
        sensor, activity, location, suffix = match.groups()
        if placement != "all" and location != placement:
            continue
        subject = next(
            (
                m.group(1)
                for part in path.parts
                if (m := re.fullmatch(r"proband[_-]?(\d+)", part.lower()))
            ),
            None,
        )
        if subject is None:
            raise ValueError(f"cannot infer subject for {path}; use an explicit manifest")
        # Nested stair recordings sometimes identify trials only in parent directories.
        parent_trials = tuple(
            part.lower()
            for part in path.parent.parts
            if re.fullmatch(r"(?:trial|session|climbingup|climbingdown)[_-]?\d+", part.lower())
        )
        key = (subject, activity, location, suffix, parent_trials)
        group = entries.setdefault(key, {})
        sensor = "acc" if sensor == "acc" else "gyro"
        if sensor in group:
            raise ValueError(
                f"ambiguous {sensor} pairing: {group[sensor]} and {path}; create an explicit manifest"
            )
        group[sensor] = str(path)
    result = []
    for (subject, activity, location, suffix, trials), pair in sorted(entries.items()):
        if set(pair) != {"acc", "gyro"}:
            raise ValueError(f"missing sensor pair: {pair}")
        result.append(
            dict(
                subject=subject,
                activity=activity,
                placement=location,
                session="_".join([activity, location, suffix, *trials]).strip("_"),
                **pair,
            )
        )
    if not result:
        raise ValueError("no paired RealWorld CSVs found; unpack nested archives first")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--placement", default="chest")
    a = p.parse_args()
    result = build_manifest(a.input, a.placement)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result)} pairs; review {a.output}")


if __name__ == "__main__":
    main()
