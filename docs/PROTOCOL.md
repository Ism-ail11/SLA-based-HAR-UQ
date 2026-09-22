# Reference packet protocol, version 1

All multibyte values are little-endian. A packet is exactly the configured payload size (64 bytes by default). CRC-16 is CCITT polynomial `0x1021`, initial `0xffff`, no reflection or final XOR; its two wire bytes are little-endian. The support checksum uses the same algorithm.

| Offset | Bytes | Field |
|---|---:|---|
| 0 | 2 | ASCII `RQ` |
| 2 | 1 | Version 1 |
| 3 | 1 | Number of projections `m`, one of 1, 2, 4 |
| 4 | 4 | Window ID, uint32 |
| 8 | 2 | Packet ID, uint16 |
| 10 | 2 | Feature dimension, uint16 |
| 12 | 4 | Nonzero xorshift32 seed |
| 16 | 1 | Sketch size, one of 0, 8, 16, 32 |
| 17 | m | Degree of each projection, uint8 |
| 17+m | 2 | Support/sign checksum |
| 19+m | 2m | Signed int16 projections |
| 19+3m | sketch size | Sketch bytes |
| next | variable | Zero padding |
| size−2 | 2 | CRC over every preceding byte |

Minimum size is `21 + 3*m + sketch_size`. Four projections and a 16-byte sketch require 49 bytes, so a 48-byte packet cannot hold that combination. The sensitivity grid changes one parameter at a time and avoids that invalid combination; serialization rejects it explicitly.

The PRNG is xorshift32 with left13, right17, left5, uint32 wraparound. A bounded draw is `next() % n`. Given a degree, reservoir sampling produces distinct coordinate indices using O(degree) storage; one subsequent PRNG draw per index chooses sign +1 for an odd low bit and −1 otherwise. Each projection consumes the same sequential PRNG stream. The support checksum concatenates each index as uint16 plus each sign as int8 in projection/index order. These choices are deterministic, portable, and tested against C; modulo reduction is a small deterministic sampling bias, not cryptographic randomness.

Degrees are sampled with a separate PRNG initialized from `seed XOR 0x9e3779b9` (replace zero with one). Its draw modulo 65536 selects from a precomputed integer CDF. Default probabilities are 0.25 for degree 1, 0.20 for degree 2, and normalized `1/(d*(d-1))` for degrees 3…16 over the remaining mass. The C encoder accepts preselected degrees; exported constants include the CDF for a firmware-side sampler.

The recent uncertainty sketch has 16 uint8 bins over top-two Q15 uncertainty. It counts at most the last 255 completed windows; it is updated once per window and copied into that window's packets. It is not a calibrated distribution of candidate labels and is not fused into phone scores without an explicit fusion rule.

A decoder belongs to one window, rejects incompatible dimensions/window IDs, ignores duplicate packet IDs or seeds, and enforces capacity. Decoding a new packet is transactional: CRC/header/support validation and consistency checks finish before state is committed. Resolved coordinates are int8; unresolved coordinates remain zero. Zero-valued resolved coordinates remain distinguishable through the `known` mask.
