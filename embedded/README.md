# Portable C kernels

Compile on a host to check the C99 implementation:

```bash
gcc -std=c99 -O2 -Wall -Wextra -Werror -c embedded/rateless_codec.c -o rateless_codec.o
```

`rq_encode` packs a caller-selected set of projection degrees into a version-1 packet. `rq_head` computes int32 Q8 logits from int8 features, class-major int8 weights, int32 bias, and fixed-point multipliers. Both return explicit negative error codes for invalid inputs or overflow. Their caller owns every buffer; neither allocates heap memory.

`tests/test_codec.py` compares these kernels with the Python implementation. `scripts/export_integer.py` writes compatible model constants. See `docs/PROTOCOL.md` for byte order and PRNG behavior, and `docs/HARDWARE.md` for the unimplemented board-specific integration work.
