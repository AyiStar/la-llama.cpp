#ifndef LAMM_SIMD_H
#define LAMM_SIMD_H

#pragma GCC diagnostic ignored "-Wpedantic"

#include "loongarch_matmul.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <array>
#include <cassert>
#include <iostream>

//// plaform
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// debug control
#if defined(LAMM_DEBUG)
constexpr bool kDebug = true;
#else
constexpr bool kDebug = false;
#endif

// abstraction for loongarch_asx SIMD intrinsics
namespace simd {

constexpr int kNumVecReg = 32;
constexpr int kVecWidth = 256;
constexpr int kF32PerVec = kVecWidth / 32;

using vreg_t = __m256;   // vector register type
using ivreg_t = __m256i; // integer vector register type

#if defined(__loongarch_asx)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS                                                             \
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27," \
  "28,29,30,31"

typedef union {
  int32_t i;
  float f;
} FloatInt;

LA_INLINE vreg_t vset(const float val) {
  FloatInt fi_tmpval = {.f = val};
  return (__m256)__lasx_xvreplgr2vr_w(fi_tmpval.i);
}

LA_INLINE ivreg_t lasx_set_q(__m128i inhi, __m128i inlo) {
  __m256i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".ifnc %[out], %[hi]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
                   "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out), [hi] "+f"(inhi)
                   : [lo] "f"(inlo));
  return out;
}

// x + y: f32
LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return __lasx_xvfadd_s(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return __lasx_xvfmadd_s(x, y, z);
}

// x - y: f32
LA_INLINE vreg_t sub(vreg_t x, vreg_t y) { return __lasx_xvfsub_s(x, y); }

// x * y: f32
LA_INLINE vreg_t mul(vreg_t x, vreg_t y) { return __lasx_xvfmul_s(x, y); }

// Convert __m256i low part to __m128i
LA_INLINE __m128i lasx_extracti128_lo(__m256i in) {
  __m128i out;
  __asm__ volatile(".ifnc %[out], %[in]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    vori.b $vr\\i, $vr\\j, 0        \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}

// Convert __m256i high part to __m128i
LA_INLINE __m128i lasx_extracti128_hi(__m256i in) {
  __m128i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}

LA_INLINE __m128 lasx_extractf128(__m256 a, int pos) {
  __m128 ret;
  if (pos == 0) {
    ret = (__m128)lasx_extracti128_lo((__m256i)a);
  } else {
    ret = (__m128)lasx_extracti128_hi((__m256i)a);
  }
  return ret;
}

// vector -> f32
LA_INLINE float reduce_sum(vreg_t x) {
  __m128 res = lasx_extractf128(x, 1);
  FloatInt tmp;
  res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(
                               __lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
  tmp.i = __lsx_vpickve2gr_w(res, 0);
  return tmp.f;
}

// load from float*
LA_INLINE vreg_t load(const float *p) { return (vreg_t)__lasx_xvld(p, 0); }
// load from quantized block

// Q4_0
LA_INLINE ivreg_t load_quants(const block_q4_0 *p) {
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q4_1
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q8_0
LA_INLINE ivreg_t load_quants(const block_q8_0 *p) {
  return __lasx_xvld((const __m256i *)(p->qs), 0);
}

// Q8_1
LA_INLINE ivreg_t load_quants(const block_q8_1 *p) {
  return __lasx_xvld((const __m256i *)(p->qs), 0);
}

LA_INLINE vreg_t sum_i16_pairs_float(const ivreg_t x) {
  ivreg_t v = __lasx_xvpackod_h(x, x);
  ivreg_t summed_pairs = __lasx_xvaddwev_w_h(x, v);
  return __lasx_xvffint_s_w(summed_pairs);
}

LA_INLINE ivreg_t lasx_maddubs_h(ivreg_t a, ivreg_t b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_h_b(a, b);
  tmp2 = __lasx_xvmulwod_h_b(a, b);
  return __lasx_xvsadd_h(tmp1, tmp2);
}

LA_INLINE vreg_t mul_sum_us8_pairs_float(const ivreg_t ax, const ivreg_t sy) {
  // Perform multiplication and create 16-bit values
  const ivreg_t dot = lasx_maddubs_h(ax, sy);
  return sum_i16_pairs_float(dot);
}

#elif defined(__AVX2__)

LA_INLINE vreg_t vset(const float f) { return _mm256_set1_ps(f); }

// x + y: f32
LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return _mm256_add_ps(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return _mm256_fmadd_ps(x, y, z);
}

// x - y: f32
LA_INLINE vreg_t sub(vreg_t x, vreg_t y) { return _mm256_sub_ps(x, y); }

// x * y: f32
LA_INLINE vreg_t mul(vreg_t x, vreg_t y) { return _mm256_mul_ps(x, y); }

// vector -> f32
LA_INLINE float reduce_sum(__m128 x) {
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

LA_INLINE float reduce_sum(vreg_t x) {
  return reduce_sum(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

// load from float*
LA_INLINE vreg_t load(const float *p) { return _mm256_loadu_ps(p); }

// load from quantized block

// Q4_0
LA_INLINE ivreg_t load_quants(const block_q4_0 *p) {
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q4_1
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}

// Q8_0
LA_INLINE ivreg_t load_quants(const block_q8_0 *p) {
  return _mm256_loadu_si256((const __m256i *)(p->qs));
};

// Q8_1
LA_INLINE ivreg_t load_quants(const block_q8_1 *p) {
  return _mm256_loadu_si256((const __m256i *)(p->qs));
};

#define MM256_SET_M128I(a, b)                                                  \
  _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// add int16_t pairwise and return as float vector
inline vreg_t sum_i16_pairs_float(const ivreg_t x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return _mm256_cvtepi32_ps(summed_pairs);
}

inline vreg_t mul_sum_us8_pairs_float(const ivreg_t ax, const ivreg_t sy) {
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
}

#endif

} // namespace simd

#endif // LAMM_SIMD_H