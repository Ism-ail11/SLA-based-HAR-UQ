#ifndef RATELESS_CODEC_H
#define RATELESS_CODEC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t rq_xorshift32(uint32_t *state);
uint16_t rq_crc16(const uint8_t *data, size_t length);
/* Caller supplies sampled degrees; output is zero-padded to capacity bytes.
 * Returns encoded byte count, or a negative error. No heap allocation. */
int rq_encode(const int8_t *z, uint16_t d, uint32_t seed, uint16_t packet_id,
              uint32_t window_id, const uint8_t *degrees, uint8_t m,
              const uint8_t *sketch, uint8_t sketch_length,
              uint8_t *output, size_t capacity);
/* Weights are class-major, logits are signed Q8. Returns 0, or negative error. */
int rq_head(const int8_t *z, uint16_t features, uint16_t classes,
            const int8_t *weights, const int32_t *bias,
            const int64_t *multiplier, uint8_t shift, int32_t *logits);
#ifdef __cplusplus
}
#endif
#endif
