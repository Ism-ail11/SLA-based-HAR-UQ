"""Integer inference with offline quantizer and lookup-table construction."""

from dataclasses import dataclass
import numpy as np

Q15 = 32768


@dataclass
class IntegerHead:
    weights: np.ndarray
    bias: np.ndarray
    multiplier: np.ndarray
    shift: int = 24

    @classmethod
    def from_float(cls, weights, bias, feature_scale, temperature=1.0):
        w, b = np.asarray(weights, float), np.asarray(bias, float)
        if w.ndim != 2 or b.shape != (w.shape[0],) or min(feature_scale, temperature) <= 0:
            raise ValueError("invalid head dimensions/scales")
        ws = np.maximum(
            abs(w).max(1) / 127,
            np.maximum(
                abs(b) / (feature_scale * (2**30 - 1)), temperature / (feature_scale * 256 * 2**24)
            ),
        )
        wq = np.clip(np.rint(w / ws[:, None]), -127, 127).astype(np.int8)
        bq = np.rint(b / (ws * feature_scale)).astype(np.int64)
        if np.any(abs(bq) > 2**31 - 1):
            raise OverflowError("bias outside int32")
        multiplier = np.rint(ws * feature_scale / temperature * 256 * 2**24).astype(np.int64)
        return cls(wq, bq.astype(np.int32), multiplier)

    def logits(self, z):
        z = np.asarray(z)
        if z.dtype != np.int8 or z.shape[-1] != self.weights.shape[1]:
            raise ValueError("expected int8 features")
        acc = z.astype(np.int64) @ self.weights.astype(np.int64).T + self.bias
        if np.any(abs(acc) > 2**31 - 1):
            raise OverflowError("int32 accumulator overflow")
        if np.any(
            abs(acc) > (2**63 - 1 - 2 ** (self.shift - 1)) // np.maximum(abs(self.multiplier), 1)
        ):
            raise OverflowError("requantization overflow")
        return np.clip(
            (acc * self.multiplier + 2 ** (self.shift - 1)) >> self.shift, -(2**31), 2**31 - 1
        ).astype(np.int32)

    def save(self, path):
        np.savez(
            path, weights=self.weights, bias=self.bias, multiplier=self.multiplier, shift=self.shift
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f["weights"], f["bias"], f["multiplier"], int(f["shift"]))


class IntegerSoftmax:
    def __init__(self):
        self.lut = np.rint(np.exp(-np.arange(4097) / 256) * 2**20).astype(np.int64)

    def __call__(self, logits):
        x = np.asarray(logits)
        if not np.issubdtype(x.dtype, np.integer) or x.shape[-1] < 2:
            raise ValueError("integer multiclass logits required")
        shape = x.shape
        x = x.reshape(-1, shape[-1]).astype(np.int64)
        delta = x.max(1, keepdims=True) - x
        values = self.lut[np.clip(delta, 0, 4096)]
        values[delta > 4096] = 0
        total = values.sum(1, keepdims=True)
        numerator = values * Q15
        p, remainder = numerator // total, numerator % total
        for i in range(len(p)):
            order = np.argsort(-remainder[i], kind="stable")
            p[i, order[: Q15 - int(p[i].sum())]] += 1
        return p.reshape(shape).astype(np.int32)


def top2_uncertainty(p):
    top = np.sort(p, axis=-1)[..., -2:]
    return Q15 - (top[..., 1] - top[..., 0])


def class_scores(p, kind="margin", logits=None, direct_confidence=None):
    p = np.asarray(p, np.int64)
    if p.shape[-1] < 2 or (p < 0).any() or (p > Q15).any():
        raise ValueError("invalid Q15 probabilities")
    if kind == "probability":
        return (Q15 - p).astype(np.int32)
    if kind == "margin":
        order = np.argsort(p, axis=-1, kind="stable")
        largest = np.take_along_axis(p, order[..., -1:], axis=-1)
        second = np.take_along_axis(p, order[..., -2:-1], axis=-1)
        rival = np.broadcast_to(largest, p.shape).copy()
        np.put_along_axis(rival, order[..., -1:], second, -1)
        return ((Q15 + rival - p + 1) // 2).astype(np.int32)
    if kind == "entropy":
        prob = p.astype(float) / Q15
        h = -(prob * np.log(np.maximum(prob, 1e-15))).sum(-1) / np.log(p.shape[-1])
        return np.rint((h[..., None] + 1 - prob) / 2 * Q15).astype(np.int32)
    if kind == "energy":
        if logits is None:
            raise ValueError("energy score needs logits")
        return np.rint(Q15 / (1 + np.exp(np.clip(np.asarray(logits) / 256, -32, 32)))).astype(
            np.int32
        )
    if kind == "direct_margin":
        if direct_confidence is None:
            raise ValueError("direct confidence missing")
        confidence = np.asarray(direct_confidence, np.int64)
        score = np.broadcast_to(confidence[..., None], p.shape).copy()
        np.put_along_axis(score, p.argmax(-1)[..., None], (Q15 - confidence)[..., None], -1)
        return score.astype(np.int32)
    raise ValueError(f"unknown score: {kind}")
