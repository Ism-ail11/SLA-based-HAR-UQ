import ctypes
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from rateless_har.codec import (
    XorShift32,
    Encoder,
    Packet,
    PeelDecoder,
    equations,
    degree_cdf,
    UncertaintySketch,
)
from rateless_har.integer import IntegerHead


def test_prng_known_vector():
    rng = XorShift32(1)
    assert [rng.next() for _ in range(3)] == [270369, 67634689, 2647435461]
    with pytest.raises(ValueError):
        XorShift32(0)


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("size", [48, 64, 96])
def test_packet_roundtrip(m, size):
    z = np.random.default_rng(7).integers(-128, 128, 128, dtype=np.int8)
    sketch = bytes(8) if m == 4 and size == 48 else bytes(range(16))
    enc = Encoder(projections=m, packet_bytes=size)
    wire = enc.encode(z, 123, 9, 42, sketch)
    packet = Packet.from_bytes(wire)
    assert packet.to_bytes(size) == wire
    assert packet.window_id == 42 and packet.packet_id == 9 and packet.sketch == sketch
    assert len(wire) == size
    for equation, projection in zip(
        equations(packet.seed, 128, packet.degrees), packet.projections
    ):
        assert len({j for j, _ in equation}) == len(equation)
        assert projection == sum(int(z[j]) * s for j, s in equation)


def test_invalid_size_and_corruption():
    with pytest.raises(ValueError, match="needs 49"):
        Encoder(projections=4, packet_bytes=48).encode(np.zeros(128, np.int8), 1)
    wire = bytearray(Encoder().encode(np.zeros(128, np.int8), 1))
    wire[22] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        Packet.from_bytes(wire)
    with pytest.raises(ValueError):
        Packet.from_bytes(b"RQ")


def singleton(z, index, packet_id=0, window_id=0):
    for seed in range(1, 100000):
        row = equations(seed, len(z), (1,))[0]
        if row[0][0] == index:
            return Packet(
                window_id, packet_id, len(z), seed, (1,), (int(z[index]) * row[0][1],)
            ).to_bytes()
    raise AssertionError("seed search failed")


@pytest.mark.parametrize("value", [-128, -1, 0, 1, 127])
def test_exact_singletons(value):
    z = np.array([value, 20, -4, 0], np.int8)
    decoder = PeelDecoder(4)
    wire = singleton(z, 0)
    assert decoder.add(wire)
    assert decoder.known[0] and decoder.z[0] == value
    assert not decoder.add(wire) and decoder.k == 1


def test_cascade_and_transaction():
    z = np.array([-128, 0], np.int8)
    row = equations(19, 2, (2,))[0]
    pair = Packet(0, 0, 2, 19, (2,), (sum(int(z[j]) * s for j, s in row),)).to_bytes()
    d = PeelDecoder(2, tolerance=0, max_packets=2)
    d.add(pair)
    assert not d.known.any()
    d.add(singleton(z, 1, 1))
    np.testing.assert_array_equal(d.z, z)
    assert d.known.all()
    before = d.z.copy()
    with pytest.raises(ValueError, match="different window"):
        d.add(singleton(z, 0, 2, 1))
    np.testing.assert_array_equal(d.z, before)
    with pytest.raises(ValueError, match="capacity"):
        d.add(singleton(z, 0, 2))


def test_invalid_singleton_rolls_back():
    d = PeelDecoder(2)
    with pytest.raises(ValueError, match="outside int8"):
        d.add(Packet(0, 0, 2, 7, (1,), (300,)).to_bytes())
    assert d.k == 0 and not d.known.any() and d.window_id is None


def test_degree_and_sketch():
    cdf = degree_cdf()
    assert cdf[-1] == 65536 and all(a <= b for a, b in zip(cdf, cdf[1:]))
    sketch = UncertaintySketch()
    for _ in range(1000):
        result = sketch.update(32768)
    assert sum(result) == 255 and result[-1] == 255


@pytest.fixture(scope="session")
def c_library(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc needed for portable C cross-check")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("c") / "codec.so"
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(root / "embedded/rateless_codec.c"),
            "-o",
            str(output),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(output))
    lib.rq_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_uint32,
        ctypes.c_uint16,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.rq_head.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_void_p,
    ]
    return lib


@pytest.mark.parametrize("seed", [1, 7, 19, 42, 0xFFFFFFFF])
@pytest.mark.parametrize("m", [1, 2, 4])
def test_c_python_packet_parity(c_library, seed, m):
    rng = np.random.default_rng(seed)
    for dimension in (8, 128, 513):
        z = rng.integers(-128, 128, dimension, dtype=np.int8)
        enc = Encoder(dimension, projections=m)
        wire = enc.encode(z, seed, 7, 1234)
        p = Packet.from_bytes(wire)
        degrees = np.array(p.degrees, np.uint8)
        sketch = np.zeros(16, np.uint8)
        output = np.zeros(64, np.uint8)
        result = c_library.rq_encode(
            z.ctypes.data,
            dimension,
            seed,
            7,
            1234,
            degrees.ctypes.data,
            m,
            sketch.ctypes.data,
            16,
            output.ctypes.data,
            64,
        )
        assert result == 64 and output.tobytes() == wire


def test_c_python_head_parity(c_library):
    rng = np.random.default_rng(19)
    head = IntegerHead.from_float(rng.normal(0, 0.1, (6, 128)), rng.normal(0, 0.2, 6), 0.037, 1.3)
    for _ in range(50):
        z = rng.integers(-128, 128, 128, dtype=np.int8)
        output = np.zeros(6, np.int32)
        result = c_library.rq_head(
            z.ctypes.data,
            128,
            6,
            head.weights.ctypes.data,
            head.bias.ctypes.data,
            head.multiplier.ctypes.data,
            head.shift,
            output.ctypes.data,
        )
        assert result == 0
        np.testing.assert_array_equal(output, head.logits(z))
