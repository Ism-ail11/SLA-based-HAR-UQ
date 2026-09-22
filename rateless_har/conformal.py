"""Frozen split-conformal histograms with conservative finite-sample quantiles."""

import math
import numpy as np
from .integer import Q15


class HistogramCalibrator:
    def __init__(self, max_k=6, bins=256, alpha=0.05, pooled=False):
        if max_k < 0 or not 2 <= bins <= Q15 or not 0 < alpha < 1:
            raise ValueError("invalid calibration configuration")
        self.max_k, self.bins, self.alpha, self.pooled = max_k, bins, alpha, pooled
        self.hist = np.zeros((1 if pooled else max_k + 1, bins), np.uint64)

    def fit(self, scores, labels, realized_k=None):
        scores, labels = np.asarray(scores), np.asarray(labels)
        if scores.ndim != 3 or scores.shape[1] != self.max_k + 1 or labels.shape != (len(scores),):
            raise ValueError("calibration shape mismatch")
        if (
            (scores < 0).any()
            or (scores > Q15).any()
            or (len(labels) and (labels.min() < 0 or labels.max() >= scores.shape[2]))
        ):
            raise ValueError("invalid scores/labels")
        if realized_k is None and self.pooled:
            realized_k = np.random.default_rng(1729).integers(0, self.max_k + 1, len(scores))
        if realized_k is not None:
            realized_k = np.asarray(realized_k, int)
            if (
                realized_k.shape != labels.shape
                or (realized_k < 0).any()
                or (realized_k > self.max_k).any()
            ):
                raise ValueError("invalid realized packet counts")
        self.hist.fill(0)
        for k in range(self.max_k + 1):
            take = np.arange(len(labels)) if realized_k is None else np.flatnonzero(realized_k == k)
            s = scores[take, k, labels[take]]
            bins = np.minimum(s.astype(np.int64) * self.bins // (Q15 + 1), self.bins - 1)
            self.hist[0 if self.pooled else k] += np.bincount(bins, minlength=self.bins).astype(
                np.uint64
            )
        return self

    def threshold(self, k):
        if not 0 <= k <= self.max_k:
            raise ValueError("uncalibrated packet index")
        h = self.hist[0 if self.pooled else k]
        n = int(h.sum())
        rank = math.ceil((n + 1) * (1 - self.alpha) - 1e-12)
        if not n or rank > n:
            return Q15
        b = int(np.searchsorted(h.cumsum(), rank, side="left"))
        return min(Q15, ((b + 1) * (Q15 + 1) + self.bins - 1) // self.bins - 1)

    def predict(self, scores, k):
        return np.asarray(scores) <= self.threshold(int(k))

    def thresholds(self):
        return np.array([self.threshold(k) for k in range(self.max_k + 1)], np.int32)

    def save(self, path):
        np.savez(
            path,
            hist=self.hist,
            max_k=self.max_k,
            bins=self.bins,
            alpha=self.alpha,
            pooled=self.pooled,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            obj = cls(int(f["max_k"]), int(f["bins"]), float(f["alpha"]), bool(f["pooled"]))
            obj.hist = f["hist"]
        return obj


class AdaptiveHistogram:
    """Integer EWMA research option; not a finite-sample conformal guarantee."""

    def __init__(self, bins=256, decay_q16=655):
        if not 0 <= decay_q16 < 65536 or bins < 2:
            raise ValueError("invalid histogram")
        self.counts, self.decay = np.zeros(bins, np.int64), decay_q16

    def update(self, score):
        if not 0 <= score <= Q15:
            raise ValueError("invalid labeled score")
        self.counts = (self.counts * (65536 - self.decay) + 32768) >> 16
        b = min(len(self.counts) - 1, score * len(self.counts) // (Q15 + 1))
        self.counts[b] = min(2**31 - 1, self.counts[b] + 65536)


def calibration_curve(cal, scores, labels):
    rows = []
    for k in range(scores.shape[1]):
        sets = cal.predict(scores[:, k], k)
        rows.append(
            dict(
                k=k,
                coverage=float(sets[np.arange(len(labels)), labels].mean()),
                mean_set_size=float(sets.sum(-1).mean()),
                n=len(labels),
            )
        )
    return rows


def select_k_star(curve, target, max_size):
    return next(
        (r["k"] for r in curve if r["coverage"] >= target and r["mean_set_size"] <= max_size), None
    )
