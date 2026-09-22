"""Portable, integer-only sparse packet encoder and transactional peel decoder."""

from __future__ import annotations
import binascii
import struct
from dataclasses import dataclass
from collections import deque
import numpy as np

HEADER = struct.Struct("<2sBBIHHIB")


class XorShift32:
    def __init__(self, seed):
        if not 0 < int(seed) <= 0xFFFFFFFF:
            raise ValueError("seed must be a nonzero uint32")
        self.state = int(seed)

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def below(self, n):
        if n <= 0:
            raise ValueError("positive range required")
        return self.next() % n


def degree_cdf(cap=16, lambda1=0.25, lambda2=0.2):
    """Offline float setup; runtime samples the resulting integer CDF."""
    if not 2 <= cap <= 255 or min(lambda1, lambda2) < 0 or lambda1 + lambda2 > 1:
        raise ValueError("invalid degree distribution")
    if cap == 2:
        if lambda1 + lambda2 == 0:
            raise ValueError("empty distribution")
        p = np.array([lambda1, lambda2]) / (lambda1 + lambda2)
    else:
        tail = np.array([1 / (k * (k - 1)) for k in range(3, cap + 1)])
        p = np.r_[lambda1, lambda2, tail / tail.sum() * (1 - lambda1 - lambda2)]
    return tuple(np.rint(p.cumsum() * 65536).astype(int)[:-1]) + (65536,)


def equations(seed, dimension, degrees):
    if not 1 <= dimension <= 65535:
        raise ValueError("invalid dimension")
    rng, rows = XorShift32(seed), []
    for degree in degrees:
        if not 1 <= degree <= min(255, dimension):
            raise ValueError("invalid degree")
        indices = list(range(degree))
        for j in range(degree, dimension):
            r = rng.below(j + 1)
            if r < degree:
                indices[r] = j
        rows.append(tuple((j, 1 if rng.next() & 1 else -1) for j in indices))
    return tuple(rows)


def support_hash(rows):
    return binascii.crc_hqx(
        b"".join(struct.pack("<Hb", j, w) for row in rows for j, w in row), 0xFFFF
    )


@dataclass(frozen=True)
class Packet:
    window_id: int
    packet_id: int
    dimension: int
    seed: int
    degrees: tuple
    projections: tuple
    sketch: bytes = bytes(16)

    def to_bytes(self, size=64):
        m = len(self.degrees)
        if (
            m not in (1, 2, 4)
            or len(self.projections) != m
            or len(self.sketch) not in (0, 8, 16, 32)
        ):
            raise ValueError("invalid payload shape")
        rows = equations(self.seed, self.dimension, self.degrees)
        body = HEADER.pack(
            b"RQ", 1, m, self.window_id, self.packet_id, self.dimension, self.seed, len(self.sketch)
        )
        body += bytes(self.degrees) + struct.pack("<H", support_hash(rows))
        body += struct.pack("<" + "h" * m, *self.projections) + self.sketch
        if len(body) + 2 > size:
            raise ValueError(f"packet needs {len(body) + 2} bytes, budget {size}")
        body += bytes(size - len(body) - 2)
        return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))

    @classmethod
    def from_bytes(cls, data):
        if len(data) < HEADER.size + 7:
            raise ValueError("truncated packet")
        if binascii.crc_hqx(data[:-2], 0xFFFF) != struct.unpack("<H", data[-2:])[0]:
            raise ValueError("CRC mismatch")
        magic, version, m, wid, pid, d, seed, slen = HEADER.unpack_from(data)
        if magic != b"RQ" or version != 1 or m not in (1, 2, 4) or slen not in (0, 8, 16, 32):
            raise ValueError("invalid header")
        off = HEADER.size
        end = off + 3 * m + 2 + slen
        if end + 2 > len(data):
            raise ValueError("truncated payload")
        degrees = tuple(data[off : off + m])
        rows = equations(seed, d, degrees)
        if support_hash(rows) != struct.unpack_from("<H", data, off + m)[0]:
            raise ValueError("support hash mismatch")
        values = struct.unpack_from("<" + "h" * m, data, off + m + 2)
        if any(data[end:-2]):
            raise ValueError("nonzero padding")
        return cls(wid, pid, d, seed, degrees, values, data[off + 3 * m + 2 : end])


class Encoder:
    def __init__(
        self,
        dimension=128,
        projections=2,
        cap=16,
        lambda1=0.25,
        lambda2=0.2,
        packet_bytes=64,
        tile_size=256,
    ):
        if not 2 <= dimension <= 65535 or projections not in (1, 2, 4) or tile_size < 1:
            raise ValueError("invalid encoder configuration")
        self.dimension, self.projections, self.packet_bytes, self.tile_size = (
            dimension,
            projections,
            packet_bytes,
            tile_size,
        )
        self.cdf = degree_cdf(min(cap, dimension), lambda1, lambda2)

    def encode(self, z, seed, packet_id=0, window_id=0, sketch=bytes(16)):
        z = np.asarray(z)
        if z.shape != (self.dimension,) or z.dtype != np.int8:
            raise ValueError("expected int8 feature vector")
        rng = XorShift32(seed ^ 0x9E3779B9 or 1)
        draws = [rng.below(65536) for _ in range(self.projections)]
        degrees = tuple(next(j + 1 for j, edge in enumerate(self.cdf) if r < edge) for r in draws)
        rows = equations(seed, self.dimension, degrees)
        values = []
        for row in rows:
            value = 0
            for start in range(0, self.dimension, self.tile_size):
                value += sum(w * int(z[j]) for j, w in row if start <= j < start + self.tile_size)
            if not -32768 <= value <= 32767:
                raise OverflowError("int16 projection overflow")
            values.append(value)
        return Packet(
            window_id, packet_id, self.dimension, seed, degrees, tuple(values), sketch
        ).to_bytes(self.packet_bytes)


class PeelDecoder:
    def __init__(self, dimension=128, tolerance=2, max_packets=256):
        if dimension < 1 or tolerance < 0 or max_packets < 1:
            raise ValueError("invalid decoder configuration")
        self.z = np.zeros(dimension, np.int8)
        self.known = np.zeros(dimension, bool)
        self.tolerance, self.max_packets = tolerance, max_packets
        self.rows, self.seen, self.seeds = [], set(), set()
        self.window_id, self.sketch = None, bytes(16)

    @property
    def k(self):
        return len(self.seen)

    def add(self, wire):
        p = Packet.from_bytes(wire)
        if p.dimension != len(self.z):
            raise ValueError("dimension mismatch")
        if self.window_id is not None and p.window_id != self.window_id:
            raise ValueError("different window")
        if p.packet_id in self.seen or p.seed in self.seeds:
            return False
        if self.k >= self.max_packets:
            raise ValueError("packet capacity exceeded")
        z, known = self.z.copy(), self.known.copy()
        rows = [(dict(a), b) for a, b in self.rows]
        rows.extend(
            (dict(row), y)
            for row, y in zip(equations(p.seed, p.dimension, p.degrees), p.projections)
        )
        changed = True
        while changed:
            changed, pending = False, []
            for a, value in rows:
                for j in list(a):
                    if known[j]:
                        value -= a.pop(j) * int(z[j])
                if not a:
                    if abs(value) > self.tolerance:
                        raise ValueError("inconsistent equation")
                elif len(a) == 1:
                    j, sign = next(iter(a.items()))
                    candidate = value * sign
                    if not -128 <= candidate <= 127:
                        raise ValueError("singleton outside int8")
                    z[j], known[j], changed = candidate, True, True
                else:
                    pending.append((a, value))
            rows = pending
        self.z, self.known, self.rows = z, known, rows
        self.seen.add(p.packet_id)
        self.seeds.add(p.seed)
        self.window_id, self.sketch = p.window_id, p.sketch
        return True


class UncertaintySketch:
    def __init__(self, bins=16, history=255):
        self.counts, self.queue, self.history = np.zeros(bins, np.uint16), deque(), history

    def update(self, score_q15):
        if not 0 <= score_q15 <= 32768:
            raise ValueError("invalid Q15 score")
        b = min(len(self.counts) - 1, score_q15 * len(self.counts) // 32768)
        self.queue.append(b)
        self.counts[b] += 1
        if len(self.queue) > self.history:
            self.counts[self.queue.popleft()] -= 1
        return np.minimum(self.counts, 255).astype(np.uint8).tobytes()
