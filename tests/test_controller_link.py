import math
import numpy as np
import pytest
from scipy.stats import binom
from rateless_har.controller import (
    required_emissions,
    success_lower_bound,
    integer_budget,
    SLAController,
    wilson_lower,
)
from rateless_har.link import Event, Link, receive


@pytest.mark.parametrize("k", [0, 1, 2, 6, 20])
@pytest.mark.parametrize("q", [0.1, 0.75, 0.9, 1.0])
def test_reliability_bound(k, q):
    n = required_emissions(k, q, 0.05)
    bound = success_lower_bound(n, q, k)
    assert bound >= 0.95 - 1e-12
    assert bound <= binom.sf(k - 1, n, q) + 1e-12
    if n:
        assert success_lower_bound(n - 1, q, k) < 0.95
    assert integer_budget(k, math.floor(q * 32768), math.ceil(math.log(20) * 65536)) >= n


def test_controller_infeasibility_and_causality():
    assert required_emissions(2, 0.9, 0.05) == 11
    assert required_emissions(2, 0, 0.05) is None
    ctl = SLAController()
    initial = ctl.decide(2, 0.9, 60, 1000)
    assert initial.feasible
    for _ in range(30):
        ctl.observe(4, 1.5)
    later = ctl.decide(2, 0.9, 60, 1000)
    assert later.target_k > initial.target_k and later.emissions >= initial.emissions
    assert not ctl.decide(None, 0.9, 60, 1000).feasible
    assert not SLAController(energy_cap_uj=10).decide(1, 0.9, 60, 1000).feasible
    ctl = SLAController(debt_enabled=False)
    ctl.observe(6, 1.0)
    assert ctl.debt == 0
    assert wilson_lower(0, 0) == 0 and 0 < wilson_lower(90, 100) < 0.9


def test_ack_counts_inflight_and_deadline():
    events = [Event(0, 0, 15, 64), Event(1, 10, 25, 64), Event(2, 20, 35, 64)]
    got, sent, latency = receive(events, 100, 1, 1, 1)
    assert [e.packet_id for e in got] == [0]
    assert [e.packet_id for e in sent] == [0, 1]
    assert latency == 16
    got, _, latency = receive(events, 15, 1, 1, 1)
    assert got == [] and latency == 15


def test_common_potential_stream():
    left, right = Link("bursty", 19), Link("bursty", 19)
    for i in range(8):
        a, b = left.generate(i, 2, 20), right.generate(i, 8, 20)
        assert a == b[:2]


def test_trace_strictness(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text("window_id,packet_id,tx_ms,arrival_ms,bytes\n0,0,0,10,64\n0,1,10,,64\n")
    link = Link(trace=p)
    events = link.generate(0, 2, 0)
    assert events[1].arrival_ms is None
    with pytest.raises(ValueError):
        link.generate(0, 3, 0)
    with pytest.raises(ValueError):
        link.generate(0, 1, 0, 48)
    with pytest.raises(ValueError):
        link.generate(3, 1, 0)
    assert np.isfinite(events[0].arrival_ms)
