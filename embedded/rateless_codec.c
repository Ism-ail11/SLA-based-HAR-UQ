#include "rateless_codec.h"
#include <limits.h>
#include <string.h>

uint32_t rq_xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}
static uint16_t crc_byte(uint16_t crc, uint8_t value) {
    crc ^= (uint16_t)value << 8;
    for (int i = 0; i < 8; ++i)
        crc = (uint16_t)((crc & 0x8000u) ? ((uint32_t)crc << 1) ^ 0x1021u : (uint32_t)crc << 1);
    return crc;
}
uint16_t rq_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xffffu;
    for (size_t i = 0; i < length; ++i) crc = crc_byte(crc, data[i]);
    return crc;
}
static void put16(uint8_t *p, uint16_t x) { p[0] = (uint8_t)x; p[1] = (uint8_t)(x >> 8); }
static void put32(uint8_t *p, uint32_t x) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(x >> (8 * i));
}
int rq_encode(const int8_t *z, uint16_t d, uint32_t seed, uint16_t packet_id,
              uint32_t window_id, const uint8_t *degrees, uint8_t m,
              const uint8_t *sketch, uint8_t slen, uint8_t *out, size_t capacity) {
    if (!z || !degrees || !out || !d || !seed || (m != 1 && m != 2 && m != 4)) return -1;
    if ((slen != 0 && slen != 8 && slen != 16 && slen != 32) || (slen && !sketch)) return -1;
    if (capacity < 21u + 3u * m + slen || capacity > 65535u) return -2;
    memset(out, 0, capacity);
    out[0] = 'R'; out[1] = 'Q'; out[2] = 1; out[3] = m;
    put32(out + 4, window_id); put16(out + 8, packet_id);
    put16(out + 10, d); put32(out + 12, seed); out[16] = slen;
    uint32_t state = seed;
    uint16_t support_crc = 0xffffu;
    for (uint8_t row = 0; row < m; ++row) {
        uint16_t indices[255];
        uint16_t degree = degrees[row];
        if (!degree || degree > d) return -3;
        out[17 + row] = (uint8_t)degree;
        for (uint16_t j = 0; j < degree; ++j) indices[j] = j;
        for (uint32_t j = degree; j < d; ++j) {
            uint32_t r = rq_xorshift32(&state) % (j + 1);
            if (r < degree) indices[r] = (uint16_t)j;
        }
        int32_t value = 0;
        for (uint16_t j = 0; j < degree; ++j) {
            int8_t sign = (rq_xorshift32(&state) & 1u) ? 1 : -1;
            value += sign * (int32_t)z[indices[j]];
            support_crc = crc_byte(support_crc, (uint8_t)indices[j]);
            support_crc = crc_byte(support_crc, (uint8_t)(indices[j] >> 8));
            support_crc = crc_byte(support_crc, (uint8_t)sign);
        }
        if (value < INT16_MIN || value > INT16_MAX) return -4;
        put16(out + 19 + m + 2 * row, (uint16_t)(int16_t)value);
    }
    put16(out + 17 + m, support_crc);
    if (slen) memcpy(out + 19 + 3 * m, sketch, slen);
    put16(out + capacity - 2, rq_crc16(out, capacity - 2));
    return (int)capacity;
}
int rq_head(const int8_t *z, uint16_t features, uint16_t classes,
            const int8_t *weights, const int32_t *bias,
            const int64_t *multiplier, uint8_t shift, int32_t *logits) {
    if (!z || !weights || !bias || !multiplier || !logits || !features || !classes || shift < 1 || shift > 62) return -1;
    int64_t denom = (int64_t)1 << shift;
    for (uint16_t c = 0; c < classes; ++c) {
        int64_t acc = bias[c];
        for (uint16_t j = 0; j < features; ++j) acc += (int32_t)z[j] * weights[(size_t)c * features + j];
        if (acc < INT32_MIN || acc > INT32_MAX || multiplier[c] < 0) return -2;
        int64_t magnitude = acc < 0 ? -acc : acc;
        if (multiplier[c] && magnitude > (INT64_MAX - denom / 2) / multiplier[c]) return -3;
        int64_t numerator = acc * multiplier[c] + denom / 2;
        int64_t rounded = numerator / denom;
        if (numerator < 0 && numerator % denom) --rounded;
        logits[c] = rounded > INT32_MAX ? INT32_MAX : (rounded < INT32_MIN ? INT32_MIN : (int32_t)rounded);
    }
    return 0;
}
