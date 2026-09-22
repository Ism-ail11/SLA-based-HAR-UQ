"""Numerically check reliability sizing and exhibit rank/monotonicity limits."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
from rateless_har.controller import required_emissions, success_lower_bound, integer_budget
from rateless_har.codec import Encoder, Packet, equations


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in (1, 2, 4, 6, 12):
        for q in (0.1, 0.5, 0.75, 0.9, 0.98, 1.0):
            n = required_emissions(k, q, 0.05)
            lower = success_lower_bound(n, q, k)
            exact = float(binom.sf(k - 1, n, q))
            fixed = integer_budget(k, math.floor(q * 32768), math.ceil(math.log(20) * 65536))
            assert 0.95 - 1e-12 <= lower <= exact + 1e-12
            assert fixed >= n
            rows.append(
                dict(
                    k=k,
                    q=q,
                    emissions=n,
                    integer_emissions=fixed,
                    lower_bound=lower,
                    exact_iid_probability=exact,
                )
            )
    pd.DataFrame(rows).to_csv(out / "reliability.csv", index=False)
    encoder = Encoder()
    matrix = []
    for i in range(6):
        packet = Packet.from_bytes(encoder.encode(np.zeros(128, np.int8), i + 1, i))
        for equation in equations(packet.seed, packet.dimension, packet.degrees):
            row = np.zeros(128)
            for j, sign in equation:
                row[j] = sign
            matrix.append(row)
    summary = dict(
        matrix_shape=list(np.shape(matrix)),
        rank=int(np.linalg.matrix_rank(matrix)),
        full_reconstruction_possible=False,
        lipschitz_counterexample=dict(
            phi="identity",
            ideal=0,
            errors=[-2, -1],
            error_norms=[2, 1],
            scores=[-2, -1],
            conclusion="smaller error norm does not imply a decreasing signed score",
        ),
    )
    (out / "limits.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/theory")
    print(json.dumps(run(p.parse_args().output), indent=2))
