"""Capture board-produced JSON lines; schema/clock semantics belong to firmware."""

import argparse
import json
import time
from pathlib import Path


def main():
    import serial

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.seconds <= 0:
        p.error("--seconds must be positive")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(a.port, a.baud, timeout=1) as port, open(a.output, "x") as dst:
        end = time.monotonic() + a.seconds
        while time.monotonic() < end:
            raw = port.readline()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = {"unparsed_hex": raw.hex()}
            dst.write(json.dumps(dict(host_monotonic_s=time.monotonic(), board=payload)) + "\n")
            dst.flush()


if __name__ == "__main__":
    main()
