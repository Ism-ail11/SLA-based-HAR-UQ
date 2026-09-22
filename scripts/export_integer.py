"""Export trained integer heads, exp LUT and degree CDF as C constants."""

import argparse
from pathlib import Path
import numpy as np
from rateless_har.integer import IntegerHead, IntegerSoftmax
from rateless_har.codec import degree_cdf
from rateless_har.cli import load_config


def declaration(name, array, ctype):
    flat = np.asarray(array).reshape(-1)
    rows = [", ".join(str(int(v)) for v in flat[i : i + 12]) for i in range(0, len(flat), 12)]
    return f"static const {ctype} {name}[{len(flat)}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


def export(run, output, config):
    head = IntegerHead.load(Path(run) / "integer_head.npz")
    margin = IntegerHead.load(Path(run) / "margin_head.npz")
    result = "#ifndef RQ_MODEL_CONSTANTS_H\n#define RQ_MODEL_CONSTANTS_H\n#include <stdint.h>\n"
    result += (
        f"#define RQ_FEATURES {head.weights.shape[1]}\n#define RQ_CLASSES {head.weights.shape[0]}\n"
    )
    for prefix, obj in [("rq_head", head), ("rq_margin", margin)]:
        result += f"#define {prefix.upper()}_SHIFT {obj.shift}\n"
        for key, dtype in [("weights", "int8_t"), ("bias", "int32_t"), ("multiplier", "int64_t")]:
            result += declaration(f"{prefix}_{key}", getattr(obj, key), dtype)
    e = config["encoder"]
    result += declaration(
        "rq_degree_cdf",
        degree_cdf(min(e["cap"], head.weights.shape[1]), e["lambda1"], e["lambda2"]),
        "uint32_t",
    )
    result += declaration("rq_exp_lut", IntegerSoftmax().lut, "int32_t")
    result += "#endif\n"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    export(a.run, a.output, load_config(a.config))
