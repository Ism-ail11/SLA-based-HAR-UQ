"""Subject-safe IMU adapters and window preparation; never normalize test data by fitting it."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


@dataclass
class Recording:
    subject: str
    session: str
    placement: str
    time: np.ndarray
    values: np.ndarray
    labels: np.ndarray
    source_hz: float


def read_wisdm(path, placement="phone"):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        for record in line.split(";"):
            if not record.strip():
                continue
            parts = record.strip().split(",")
            if len(parts) != 6:
                raise ValueError(f"{path}:{number}: expected six columns")
            rows.append(
                (parts[0].strip(), parts[1].strip(), int(parts[2]), *[float(v) for v in parts[3:]])
            )
    f = pd.DataFrame(rows, columns=["subject", "label", "time", "x", "y", "z"])
    for subject, g in f.groupby("subject", sort=True):
        t = g.time.to_numpy(np.int64)
        yield Recording(
            str(subject),
            Path(path).stem,
            placement,
            (t - t[0]).astype(float) / 1e9,
            g[["x", "y", "z"]].to_numpy(float),
            g.label.to_numpy(str),
            20.0,
        )


def read_pamap2(path, placement="hand"):
    offsets = {"hand": 3, "chest": 20, "ankle": 37}
    if placement not in offsets:
        raise ValueError("invalid PAMAP2 placement")
    x = np.loadtxt(path, ndmin=2)
    if x.shape[1] != 54:
        raise ValueError("expected 54 PAMAP2 columns")
    m = re.search(r"subject(\d+)", Path(path).stem, re.I)
    if m is None:
        raise ValueError("cannot infer subject from filename")
    channels = [offsets[placement] + i for i in (1, 2, 3, 7, 8, 9)]
    yield Recording(
        m[1],
        Path(path).stem,
        placement,
        x[:, 0],
        x[:, channels],
        x[:, 1].astype(int).astype(str),
        100.0,
    )


def read_realworld(acc, gyro, subject, activity, placement, session=None):
    def load(path):
        f = pd.read_csv(path)
        f.columns = [c.strip().lower() for c in f.columns]
        cols = ["attr_time", "attr_x", "attr_y", "attr_z"]
        if not set(cols) <= set(f.columns):
            cols = ["timestamp", "x", "y", "z"]
        return f[cols].to_numpy(float)

    a, g = load(acc), load(gyro)
    if len(a) < 2 or len(g) < 2 or (np.diff(a[:, 0]) <= 0).any() or (np.diff(g[:, 0]) <= 0).any():
        raise ValueError("timestamps must increase within each CSV")
    a = a[(a[:, 0] >= g[0, 0]) & (a[:, 0] <= g[-1, 0])]
    if len(a) < 2:
        raise ValueError("no shared sensor interval")
    aligned = np.stack([np.interp(a[:, 0], g[:, 0], g[:, j]) for j in (1, 2, 3)], -1)
    right = np.clip(np.searchsorted(g[:, 0], a[:, 0], side="right"), 1, len(g) - 1)
    aligned[g[right, 0] - g[right - 1, 0] > 250] = np.nan
    yield Recording(
        str(subject),
        session or Path(acc).stem,
        placement,
        (a[:, 0] - a[0, 0]) / 1000,
        np.c_[a[:, 1:4], aligned],
        np.repeat(str(activity), len(a)),
        50.0,
    )


def read_manifest(path):
    path = Path(path)
    for item in json.loads(path.read_text()):

        def locate(key):
            value = Path(item[key])
            return value if value.is_absolute() else path.parent / value

        if "acc" in item:
            yield from read_realworld(
                locate("acc"),
                locate("gyro"),
                item["subject"],
                item["activity"],
                item["placement"],
                item.get("session"),
            )
        else:
            f = pd.read_csv(locate("csv"))
            yield Recording(
                str(item["subject"]),
                item["session"],
                item["placement"],
                f.timestamp_s.to_numpy(float),
                f[item.get("channels", ["ax", "ay", "az", "gx", "gy", "gz"])].to_numpy(float),
                f.label.to_numpy(str),
                float(item["source_hz"]),
            )


def window_recording(
    rec,
    target_hz=50,
    window_s=2.0,
    stride_s=1.0,
    purity=0.8,
    max_gap_s=0.25,
    label_map=None,
    excluded_labels=(),
):
    if min(target_hz, window_s, stride_s, rec.source_hz) <= 0 or not 0 < purity <= 1:
        raise ValueError("invalid window settings")
    t, x, y = rec.time, rec.values, rec.labels.astype(str)
    if len(t) != len(x) or len(t) != len(y) or x.ndim != 2:
        raise ValueError("recording shape mismatch")
    if label_map is not None:
        if set(y) - set(label_map):
            raise ValueError("label mapping must cover every native label")
        y = np.array([label_map[v] for v in y], str)
    valid = np.isfinite(x).all(-1) & np.isfinite(t) & ~np.isin(y, list(excluded_labels))
    breaks = np.r_[True, (np.diff(t) <= 0) | (np.diff(t) > max_gap_s)]
    segments, start = [], None
    for i in range(len(t)):
        if (breaks[i] or not valid[i]) and start is not None:
            segments.append((start, i))
            start = None
        if valid[i] and start is None:
            start = i
    if start is not None:
        segments.append((start, len(t)))
    width, stride = round(window_s * target_hz), round(stride_s * target_hz)
    if min(width, stride) < 1:
        raise ValueError("window/stride rounds to zero")
    ratio = Fraction(target_hz / rec.source_hz).limit_denominator(1000)
    for begin, end in segments:
        st, sx, sy = t[begin:end], x[begin:end], y[begin:end]
        if len(st) < 2 or st[-1] - st[0] < window_s - 1 / rec.source_hz - 1e-8:
            continue
        nt = int(np.floor((st[-1] - st[0]) * rec.source_hz + 1e-7)) + 1
        native = st[0] + np.arange(nt) / rec.source_hz
        regular = np.stack([np.interp(native, st, sx[:, j]) for j in range(x.shape[1])], -1)
        values = resample_poly(regular, ratio.numerator, ratio.denominator, axis=0)
        times = st[0] + np.arange(len(values)) / target_hz
        right = np.clip(np.searchsorted(st, times), 0, len(st) - 1)
        left = np.maximum(0, right - 1)
        labels = sy[np.where(abs(times - st[left]) <= abs(times - st[right]), left, right)]
        for i in range(0, len(values) - width + 1, stride):
            choices, counts = np.unique(labels[i : i + width], return_counts=True)
            winner = counts.argmax()
            if counts[winner] / width >= purity:
                yield (
                    values[i : i + width].T.astype(np.float32),
                    choices[winner],
                    rec.subject,
                    rec.session,
                    rec.placement,
                    float(times[i]),
                )


def build_windows(recordings, output, **kwargs):
    rows = [row for rec in recordings for row in window_recording(rec, **kwargs)]
    if not rows:
        raise ValueError("no usable windows; inspect timestamps/segments/channels")
    vocab = sorted({r[1] for r in rows})
    x = np.stack([r[0] for r in rows])
    np.savez_compressed(
        output,
        x=x,
        y=np.array([vocab.index(r[1]) for r in rows], np.int64),
        subject=np.array([r[2] for r in rows]),
        session=np.array([r[3] for r in rows]),
        placement=np.array([r[4] for r in rows]),
        start_s=np.array([r[5] for r in rows]),
        classes=np.array(vocab),
        sample_hz=kwargs.get("target_hz", 50),
    )
    return dict(
        windows=len(rows), subjects=len(set(r[2] for r in rows)), classes=vocab, shape=list(x.shape)
    )


def subject_split(subjects, seed=7, explicit=None):
    unique = np.array(sorted(set(map(str, subjects))))
    if len(unique) < 5:
        raise ValueError("at least five subjects required")
    if explicit is None:
        order = np.random.default_rng(seed).permutation(unique)
        nt, nc = max(1, round(0.2 * len(unique))), max(1, round(0.1 * len(unique)))
        nv = max(1, round(0.1 * (len(unique) - nt - nc)))
        explicit = dict(
            test=order[:nt].tolist(),
            cal=order[nt : nt + nc].tolist(),
            val=order[nt + nc : nt + nc + nv].tolist(),
            train=order[nt + nc + nv :].tolist(),
        )
    if set(explicit) != {"train", "val", "cal", "test"}:
        raise ValueError("need train/val/cal/test subject lists")
    assigned = [str(s) for group in explicit.values() for s in group]
    if (
        len(assigned) != len(set(assigned))
        or set(assigned) != set(unique)
        or any(not g for g in explicit.values())
    ):
        raise ValueError("overlapping, incomplete, or empty split")
    return {k: list(map(str, v)) for k, v in explicit.items()}


def normalization(x):
    if not len(x):
        raise ValueError("empty training data")
    x = np.asarray(x, float)
    return x.mean((0, 2), keepdims=True).astype(np.float32), np.maximum(
        x.std((0, 2), keepdims=True), 1e-6
    ).astype(np.float32)
