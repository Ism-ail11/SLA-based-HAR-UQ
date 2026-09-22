import math
import numpy as np
import pytest
from rateless_har.integer import IntegerHead, IntegerSoftmax, Q15, class_scores
from rateless_har.conformal import HistogramCalibrator, AdaptiveHistogram, select_k_star


def test_integer_softmax_extremes():
    logits = np.array([[0, 0, 0], [-(2**31), 2**31 - 1, 0], [100, 0, -100]], np.int32)
    p = IntegerSoftmax()(logits)
    np.testing.assert_array_equal(p.sum(1), [Q15] * 3)
    assert p[1, 1] == Q15 and (p >= 0).all()
    assert p[2, 0] > p[2, 1] > p[2, 2]


def test_integer_head_accuracy_and_constant_bias():
    rng = np.random.default_rng(19)
    w, b = rng.normal(0, 0.1, (5, 128)), rng.normal(0, 0.1, 5)
    z = rng.integers(-128, 128, (30, 128), dtype=np.int8)
    head = IntegerHead.from_float(w, b, 0.025)
    error = abs(head.logits(z) / 256 - (z.astype(float) * 0.025 @ w.T + b))
    assert error.max() < 0.08
    constant = IntegerHead.from_float(np.zeros((2, 128)), np.array([1.0, -1.0]), 0.05)
    np.testing.assert_allclose(constant.logits(z)[0] / 256, [1, -1], atol=0.004)


@pytest.mark.parametrize("n", [0, 1, 18, 19, 20, 100, 500])
@pytest.mark.parametrize("bins", [128, 256, 512])
def test_conservative_histogram_quantile(n, bins):
    rng = np.random.default_rng(n)
    s = rng.integers(0, Q15 + 1, n)
    scores = np.stack([s, Q15 - s], axis=-1)[:, None, :]
    cal = HistogramCalibrator(0, bins, 0.05).fit(scores, np.zeros(n, int))
    rank = math.ceil((n + 1) * 0.95 - 1e-12)
    exact = Q15 if rank > n or n == 0 else sorted(s)[rank - 1]
    assert cal.threshold(0) >= exact
    if rank <= n:
        assert cal.threshold(0) - exact <= math.ceil((Q15 + 1) / bins)


def test_pooled_counts_once_and_realized_groups():
    scores = np.full((100, 7, 3), 1000, np.int32)
    labels = np.zeros(100, int)
    assert HistogramCalibrator(pooled=True).fit(scores, labels).hist.sum() == 100
    cal = HistogramCalibrator().fit(scores, labels, np.repeat(2, 100))
    assert cal.hist[2].sum() == 100 and cal.hist.sum() == 100
    assert cal.threshold(1) == Q15
    assert cal.predict([0, 10000, Q15], 1).all()


def test_exchangeable_coverage_sanity():
    rng = np.random.default_rng(7)
    scores = rng.integers(0, Q15 + 1, (20000, 1, 2))
    labels = rng.integers(0, 2, 20000)
    cal = HistogramCalibrator(0).fit(scores[:2000], labels[:2000])
    sets = cal.predict(scores[2000:, 0], 0)
    coverage = sets[np.arange(18000), labels[2000:]].mean()
    assert 0.93 < coverage < 0.98


def test_scores_and_unattainable_policy():
    p = np.array([30000, 2000, 768])
    scores = class_scores(p)
    assert scores.argmin() == p.argmax()
    assert (scores >= 0).all() and (scores <= Q15).all()
    assert select_k_star([dict(k=0, coverage=1.0, mean_set_size=3.0)], 0.95, 1.5) is None
    hist = AdaptiveHistogram()
    hist.update(0)
    assert hist.counts[0] == 65536
