"""Reproducible potential-opportunity link streams and application trace replay."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

REGIMES = {
    "good": (80.0, 0.98),
    "moderate": (60.0, 0.90),
    "poor": (40.0, 0.75),
    "bursty": (60.0, 0.90),
}


@dataclass(frozen=True)
class Event:
    packet_id: int
    tx_ms: float
    arrival_ms: float | None
    bytes: int


class Link:
    def __init__(self, regime="moderate", seed=7, jitter_ms=2.0, trace=None):
        if regime not in REGIMES or jitter_ms < 0:
            raise ValueError("invalid link")
        self.regime, self.rate, self.q = regime, *REGIMES[regime]
        self.rng, self.jitter, self.bad = np.random.default_rng(seed), jitter_ms, False
        self.trace = None
        if trace is not None:
            f = pd.read_csv(trace)
            if not {"window_id", "packet_id", "tx_ms", "arrival_ms", "bytes"} <= set(f):
                raise ValueError("invalid trace schema")
            if (
                f.duplicated(["window_id", "packet_id"]).any()
                or (f.tx_ms < 0).any()
                or (f["bytes"] <= 0).any()
            ):
                raise ValueError("duplicate IDs or invalid trace values")
            valid = f.arrival_ms.notna()
            if (f.loc[valid, "arrival_ms"] < f.loc[valid, "tx_ms"]).any():
                raise ValueError("arrival precedes transmission")
            self.trace = {int(k): g for k, g in f.groupby("window_id")}

    def generate(self, window_id, n, compute_ms, packet_bytes=64):
        if n < 0 or compute_ms < 0 or packet_bytes < 1:
            raise ValueError("invalid transmission request")
        if self.trace is not None:
            if window_id not in self.trace:
                raise ValueError(f"trace has no window {window_id}")
            f = self.trace[window_id].sort_values("packet_id").iloc[:n]
            if len(f) < n or not (f["bytes"] == packet_bytes).all():
                raise ValueError("trace budget/packet size mismatch")
            return [
                Event(
                    int(r.packet_id),
                    float(r.tx_ms),
                    None if pd.isna(r.arrival_ms) else float(r.arrival_ms),
                    int(r.bytes),
                )
                for r in f.itertuples()
            ]
        duration = 1000 / self.rate * packet_bytes / 64
        events = []
        # Fixed potential stream makes paired policies comparable despite different budgets.
        for i in range(max(64, n)):
            state, loss, jitter = self.rng.random(3)
            if self.regime == "bursty":
                self.bad = not (state < 0.25) if self.bad else state < 0.08
                success = loss < (0.25 if self.bad else 0.98)
            else:
                success = loss < self.q
            tx = compute_ms + i * duration
            if i < n:
                events.append(
                    Event(
                        i,
                        tx,
                        tx + duration + jitter * self.jitter if success else None,
                        packet_bytes,
                    )
                )
        return events


def receive(events, deadline_ms, target_k, decode_ms=0.0, ack_delay_ms=0.0, max_k=6):
    if min(deadline_ms, decode_ms, ack_delay_ms, target_k) < 0:
        raise ValueError("invalid timing")
    if target_k == 0:
        return [], [], 0.0
    attempts = [e for e in events if e.tx_ms < deadline_ms]
    candidates = sorted(
        (
            e
            for e in attempts
            if e.arrival_ms is not None and e.arrival_ms + decode_ms <= deadline_ms
        ),
        key=lambda e: (e.arrival_ms, e.packet_id),
    )
    target = min(target_k, max_k)
    useful, seen, response = [], set(), deadline_ms
    for event in candidates:
        if event.packet_id in seen:
            continue
        useful.append(event)
        seen.add(event.packet_id)
        if len(useful) >= target:
            response = event.arrival_ms + decode_ms
            break
    stop = min(deadline_ms, response + ack_delay_ms) if len(useful) >= target else deadline_ms
    return useful[:max_k], [e for e in attempts if e.tx_ms < stop], response
