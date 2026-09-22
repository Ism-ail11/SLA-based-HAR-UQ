"""Synthetic fixtures only: these are not benchmark measurements."""

from pathlib import Path
import json
import numpy as np
from .integer import IntegerHead


def make_windows(path, subjects=10, windows_per_subject=24, channels=6, samples=100, seed=7):
    rng = np.random.default_rng(seed)
    x, y, users = [], [], []
    time = np.arange(samples) / 50
    for user in range(subjects):
        for i in range(windows_per_subject):
            label = i % 3
            signal = np.stack(
                [np.sin(2 * np.pi * (label + 1) * time + c * 0.3) for c in range(channels)]
            )
            signal += rng.normal(0, 0.12, signal.shape) + rng.normal(0, 0.05, (channels, 1))
            x.append(signal)
            y.append(label)
            users.append(f"{user:02d}")
    n = len(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.array(x, np.float32),
        y=np.array(y, np.int64),
        subject=np.array(users),
        session=np.repeat("synthetic", n),
        placement=np.array(["wrist" if i % 2 else "pocket" for i in range(n)]),
        start_s=np.tile(np.arange(windows_per_subject), subjects),
        classes=np.array(["synthetic_0", "synthetic_1", "synthetic_2"]),
        sample_hz=50,
    )


def make_features(directory, dimension=128, per_role=120, seed=7):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, b = rng.normal(0, 0.15, (3, dimension)), np.zeros(3)
    data = dict(
        weights_float=w,
        bias_float=b,
        feature_scale=0.05,
        temperature=1.0,
        classes=np.array(["synthetic_0", "synthetic_1", "synthetic_2"]),
        dropout_weights=w,
        dropout_bias=b,
    )
    for j, role in enumerate(("train", "val", "cal", "test")):
        y = rng.integers(0, 3, per_role)
        z = np.clip(np.rint(70 * w[y] + rng.normal(0, 7, (per_role, dimension))), -128, 127).astype(
            np.int8
        )
        data.update(
            {
                f"z_{role}": z,
                f"y_{role}": y,
                f"subject_{role}": np.repeat(f"synthetic_subject_{j}", per_role),
                f"placement_{role}": np.where(np.arange(per_role) % 2, "wrist", "pocket"),
            }
        )
    np.savez_compressed(out / "features.npz", **data)
    IntegerHead.from_float(w, b, 0.05).save(out / "integer_head.npz")
    IntegerHead.from_float(np.zeros((1, dimension)), np.array([1.0]), 0.05).save(
        out / "margin_head.npz"
    )
    (out / "fixture.json").write_text(json.dumps(dict(synthetic=True, seed=seed)) + "\n")
