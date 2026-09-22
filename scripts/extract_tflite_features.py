"""Create a separate feature bundle using the converted integer backbone."""

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np


def extract(run, data_path, export_dir, output):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    run, exp, out = Path(run), Path(export_dir), Path(output)
    if out.resolve() == run.resolve():
        raise ValueError("output must differ from original training directory")
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((exp / "export.json").read_text())
    training = json.loads((run / "training.json").read_text())
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != training["data_sha256"]:
        raise ValueError("prepared data differs from the training input")
    split = json.loads((run / "split.json").read_text())
    with np.load(run / "features.npz", allow_pickle=False) as original:
        bundle = {k: original[k] for k in original.files}
    with (
        np.load(data_path, allow_pickle=False) as raw,
        np.load(run / "normalization.npz", allow_pickle=False) as norm,
    ):
        interpreter = tf.lite.Interpreter(model_path=str(exp / "model_int8.tflite"), num_threads=2)
        interpreter.allocate_tensors()
        for role, subjects in split.items():
            mask = np.isin(raw["subject"].astype(str), subjects)
            if role != "test" and training["config"].get("id_placement"):
                mask &= raw["placement"] == training["config"]["id_placement"]
            indices = np.flatnonzero(mask)
            if len(indices) != len(bundle[f"z_{role}"]):
                raise ValueError("role size differs from training bundle")
            values = []
            for i in indices:
                sample = ((raw["x"][i : i + 1] - norm["mean"]) / norm["std"]).astype(np.float32)
                q = np.clip(
                    np.rint(sample[:, :, :, None] / info["input_scale"]) + info["input_zero"],
                    -128,
                    127,
                ).astype(np.int8)
                interpreter.set_tensor(info["input_index"], q)
                interpreter.invoke()
                real = (
                    interpreter.get_tensor(info["feature_index"]).astype(float)
                    - info["feature_zero"]
                ) * info["feature_scale"]
                values.append(
                    np.clip(np.rint(real[0] / info["reference_feature_scale"]), -128, 127).astype(
                        np.int8
                    )
                )
            bundle[f"z_{role}"] = np.stack(values)
        del interpreter
    np.savez_compressed(out / "features.npz", **bundle)
    for filename in (
        "integer_head.npz",
        "margin_head.npz",
        "split.json",
        "normalization.npz",
        "training.json",
    ):
        shutil.copy2(run / filename, out / filename)
    info["note"] = (
        "Converted backbone features requantized to original head scale. Run evaluate/grid for fresh calibration."
    )
    (out / "tflite_provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    gc.collect()
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--export", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    extract(a.run, a.data, a.export, a.output)
