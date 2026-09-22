"""Export a QAT checkpoint to an all-integer TFLite feature backbone and head.

Conversion recalibrates activations from TRAIN subjects only. Re-extract features
and recalibrate conformal thresholds before evaluating the converted model.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np


def export(run, data_path, output, representatives=128):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import torch
    import tensorflow as tf
    from torch.ao.quantization import disable_observer
    from rateless_har.model import TinyHAR

    run, out = Path(run), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = TinyHAR(checkpoint["classes"], checkpoint["feature_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.apply(disable_observer)
    with (
        np.load(data_path, allow_pickle=False) as raw,
        np.load(run / "normalization.npz", allow_pickle=False) as norm,
    ):
        split = json.loads((run / "split.json").read_text())
        take = np.isin(raw["subject"].astype(str), split["train"])
        training = json.loads((run / "training.json").read_text())
        if training["config"].get("id_placement"):
            take &= raw["placement"] == training["config"]["id_placement"]
        x = ((raw["x"][take] - norm["mean"]) / norm["std"]).astype(np.float32)
    if representatives < 1 or not len(x):
        raise ValueError("nonempty representative TRAIN data required")
    x = x[:representatives]
    constants = []
    with torch.no_grad():
        for block in model.blocks:
            w = block.weight_quant(block.conv.weight).numpy()
            bn = block.bn
            scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).numpy()
            bias = (bn.bias - bn.running_mean * torch.from_numpy(scale)).numpy()
            w = w * scale[:, None, None, None]
            depthwise = block.conv.groups == block.conv.in_channels and block.conv.groups > 1
            kernel = w.transpose(2, 3, 0, 1) if depthwise else w.transpose(2, 3, 1, 0)
            constants.append(
                (
                    tf.constant(kernel),
                    tf.constant(bias),
                    depthwise,
                    block.conv.stride,
                    block.conv.padding,
                )
            )
        projection_w = tf.constant(
            model.projection.weight_quant(model.projection.linear.weight).numpy().T
        )
        projection_b = tf.constant(model.projection.linear.bias.numpy())
        head_w = tf.constant(model.head.weight_quant(model.head.linear.weight).numpy().T)
        head_b = tf.constant(model.head.linear.bias.numpy())
    channels, samples = x.shape[1:]

    class ExportModule(tf.Module):
        @tf.function(
            input_signature=[tf.TensorSpec([1, channels, samples, 1], tf.float32, name="imu")]
        )
        def infer(self, inputs):
            value = inputs
            for kernel, bias, depthwise, stride, padding in constants:
                ph, pw = padding
                value = tf.pad(value, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
                strides = [1, stride[0], stride[1], 1]
                if depthwise:
                    value = tf.nn.depthwise_conv2d(value, kernel, strides, "VALID")
                else:
                    value = tf.nn.conv2d(value, kernel, strides, "VALID")
                value = tf.nn.relu6(tf.nn.bias_add(value, bias))
            features = tf.matmul(tf.reduce_mean(value, axis=[1, 2]), projection_w) + projection_b
            logits = (tf.matmul(features, head_w) + head_b) / float(checkpoint["temperature"])
            return dict(features=features, logits=logits)

    module = ExportModule()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.infer.get_concrete_function()], module
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None, :, :, None]] for sample in x)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    (out / "model_int8.tflite").write_bytes(blob)
    byte_rows = [", ".join(f"0x{v:02x}" for v in blob[i : i + 16]) for i in range(0, len(blob), 16)]
    (out / "model_data.cc").write_text(
        "#include <cstddef>\n#include <cstdint>\nalignas(16) const uint8_t rq_model[] = {\n"
        + ",\n".join(byte_rows)
        + "\n};\nconst size_t rq_model_len = sizeof(rq_model);\n"
    )
    interpreter = tf.lite.Interpreter(model_content=blob, num_threads=2)
    interpreter.allocate_tensors()
    tensors = interpreter.get_tensor_details()
    if any(np.issubdtype(t["dtype"], np.floating) for t in tensors):
        raise RuntimeError("conversion retained floating-point tensors")
    signatures = interpreter._get_full_signature_list()
    signature = signatures[next(iter(signatures))]
    feature_index = signature["outputs"]["features"]
    inp = interpreter.get_input_details()[0]
    feature = next(t for t in tensors if t["index"] == feature_index)
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = feature["quantization"]
    errors = []
    for sample in x[:16]:
        q = np.clip(np.rint(sample[None, :, :, None] / input_scale) + input_zero, -128, 127).astype(
            np.int8
        )
        interpreter.set_tensor(inp["index"], q)
        interpreter.invoke()
        converted = (
            interpreter.get_tensor(feature_index).astype(float) - output_zero
        ) * output_scale
        with torch.no_grad():
            reference = model.features(torch.tensor(sample[None, None])).numpy()
        errors.append(float(np.max(np.abs(converted - reference))))
    info = dict(
        format="TFLite int8",
        bytes=len(blob),
        all_tensors_integer=True,
        representative_role="train",
        representative_windows=len(x),
        input_index=int(inp["index"]),
        feature_index=int(feature_index),
        input_scale=input_scale,
        input_zero=input_zero,
        feature_scale=output_scale,
        feature_zero=output_zero,
        reference_feature_scale=training["feature_scale"],
        max_feature_difference=max(errors),
        requires_feature_reextraction_and_recalibration=True,
        tensorflow=tf.__version__,
    )
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    # Clear function closures before TensorFlow's interpreter shutdown.
    del interpreter, converter, module, ExportModule
    tf.keras.backend.clear_session()
    gc.collect()
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representatives", type=int, default=128)
    a = p.parse_args()
    print(json.dumps(export(a.run, a.data, a.output, a.representatives), indent=2))
