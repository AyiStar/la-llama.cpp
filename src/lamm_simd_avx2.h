#ifndef LAMM_SIMD_AVX2_H
#define LAMM_SIMD_AVX2_H

#pragma GCC diagnostic ignored "-Wpedantic"

#include "lamm_ggml_type_trait.h"
#include "loongarch_matmul.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <array>
#include <cassert>
#include <iostream>

#include <immintrin.h>

// abstraction for loongarch_asx SIMD intrinsics
namespace simd {

constexpr int kNumVecReg = 32;
constexpr int kVecWidth = 256;
constexpr int kF32PerVec = kVecWidth / 32;

using vreg_t = __m256;    // vector register type
using ivreg_t = __m256i;  // integer vector register type
using hvreg_t = __m128;   // half vector register type
using hivreg_t = __m128i; // half integer register type

LA_INLINE vreg_t vset(const float f) { return _mm256_set1_ps(f); }
LA_INLINE ivreg_t ivset(const char i) { return _mm256_set1_epi8(i); }
LA_INLINE hivreg_t hivset(const char i) { return _mm_set1_epi8(i); }

LA_INLINE ivreg_t extend(hivreg_t h) { return _mm256_cvtepi8_epi16(h); }
LA_INLINE hivreg_t trunc(ivreg_t i, int select) {
  return _mm256_extracti128_si256(i, select);
}
LA_INLINE vreg_t to_float(ivreg_t i) { return _mm256_cvtepi32_ps(i); }

// x + y: f32
LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return _mm256_add_ps(x, y); }

// x + y: int32
LA_INLINE ivreg_t add(ivreg_t x, ivreg_t y) { return _mm256_add_epi32(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return _mm256_fmadd_ps(x, y, z);
}

// x - y: f32
LA_INLINE vreg_t sub(vreg_t x, vreg_t y) { return _mm256_sub_ps(x, y); }

// x - y: int
LA_INLINE ivreg_t sub(ivreg_t x, ivreg_t y) { return _mm256_sub_epi8(x, y); }

// x * y: f32
LA_INLINE vreg_t mul(vreg_t x, vreg_t y) { return _mm256_mul_ps(x, y); }

// x: int8 * y: int8 -> int16
LA_INLINE ivreg_t mul(ivreg_t ax, ivreg_t sy) {
  return _mm256_madd_epi16(ax, sy);
};

// x: uint8 * y: int8 -> int16
LA_INLINE ivreg_t mul_ubs(ivreg_t ax, ivreg_t sy) {
  return _mm256_maddubs_epi16(ax, sy);
};

// x & y
LA_INLINE ivreg_t _and(ivreg_t x, ivreg_t y) { return _mm256_and_si256(x, y); }
LA_INLINE hivreg_t _and(hivreg_t x, hivreg_t y) { return _mm_and_si128(x, y); }

// (~x) & y
LA_INLINE ivreg_t andnot(ivreg_t x, ivreg_t y) {
  return _mm256_andnot_si256(x, y);
}

// x | y
LA_INLINE ivreg_t _or(ivreg_t x, ivreg_t y) { return _mm256_or_si256(x, y); }

// x: int16 >> n
LA_INLINE ivreg_t logic_shift_right(ivreg_t x, int n) {
  return _mm256_srli_epi16(x, n);
}
LA_INLINE hivreg_t logic_shift_right(hivreg_t x, int n) {
  return _mm_srli_epi16(x, n);
}

LA_INLINE ivreg_t shuffle(ivreg_t x, ivreg_t y) {
  return _mm256_shuffle_epi8(x, y);
}

#define MM256_SET_M128I(a, b)                                                  \
  _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

LA_INLINE ivreg_t concat(hivreg_t a, hivreg_t b) {
  return _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1);
}

// 32 bits -> 256 bits
LA_INLINE ivreg_t spread_bits(const uint8_t *x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m256i shuf_mask =
      _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                        0x0101010101010101, 0x0000000000000000);
  __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
  const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytes = _mm256_or_si256(bytes, bit_mask);
  return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// sum 4 f32 -> f32
LA_INLINE float reduce_sum(__m128 x) {
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

// sum 8 f32 -> f32
LA_INLINE float reduce_sum(vreg_t x) {
  return reduce_sum(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

// load from float*
LA_INLINE vreg_t load(const float *p) { return _mm256_loadu_ps(p); }

// load from uint8_t*
LA_INLINE ivreg_t load(const char *p) {
  return _mm256_loadu_si256((const ivreg_t *)p);
}
LA_INLINE hivreg_t loadh(const char *p) {
  return _mm_loadu_si128((const hivreg_t *)p);
}

// load from quantized block

// Q4_0
LA_INLINE ivreg_t load_quants(const block_q4_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q4_0>::super_block_size == 32);
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q4_1
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q4_1>::super_block_size == 32);
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q5_0
LA_INLINE ivreg_t load_quants(const block_q5_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q5_0>::super_block_size == 32);
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q5_1
LA_INLINE ivreg_t load_quants(const block_q5_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q5_1>::super_block_size == 32);
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q8_0
LA_INLINE ivreg_t load_quants(const block_q8_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q8_0>::super_block_size == 32);
  return _mm256_loadu_si256((const __m256i *)(p->qs));
};

// Q8_1
LA_INLINE ivreg_t load_quants(const block_q8_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q8_1>::super_block_size == 32);
  return _mm256_loadu_si256((const __m256i *)(p->qs));
};

// add int16_t pairwise and return as float vector
inline vreg_t sum_i16_pairs_float(const ivreg_t x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return to_float(summed_pairs);
}

inline vreg_t mul_sum_us8_pairs_float(const ivreg_t ax, const ivreg_t sy) {
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
}

inline vreg_t mul_sum_i8_pairs_float(const ivreg_t x, const ivreg_t y) {
  // Perform multiplication and create 16-bit values
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_float(ax, sy);
}

} // namespace simd

#endif // LAMM_SIMD_AVX2_H