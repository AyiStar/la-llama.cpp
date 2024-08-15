#ifndef LAMM_SIMD_LOONGARCH_H
#define LAMM_SIMD_LOONGARCH_H

#pragma GCC diagnostic ignored "-Wpedantic"

#include "loongarch_matmul.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <array>
#include <cassert>
#include <iostream>

#include <lasxintrin.h>
#include <lsxintrin.h>

// abstraction for loongarch_asx SIMD intrinsics
namespace simd {

constexpr int kNumVecReg = 32;
constexpr int kVecWidth = 256;
constexpr int kF32PerVec = kVecWidth / 32;

using vreg_t = __m256;    // vector register type
using ivreg_t = __m256i;  // integer vector register type
using hvreg_t = __m128;   // half vector register type
using hivreg_t = __m128i; // half integer register type

/*
Part of the work below belongs to Loongson Corp.
https://github.com/ggerganov/llama.cpp/pull/6454

*/

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

// Convert two __m128i to __m256i
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

LA_INLINE __m256i lasx_set_d(int64_t a, int64_t b, int64_t c, int64_t d) {
  v4i64 __ret = {d, c, b, a};
  return (__m256i)__ret;
}

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

LA_INLINE __m256i lasx_shuffle_b(__m256i a, __m256i b) {
  __m256i mask_f, zero, tmp0, tmp2, mask;
  int f = 0x8f;
  mask_f = __lasx_xvreplgr2vr_b(f);
  zero = __lasx_xvldi(0);
  tmp0 = __lasx_xvand_v(b, mask_f); // get mask with low 4 bit and sign bits
  tmp0 = __lasx_xvori_b(
      tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
  mask = __lasx_xvsle_b(zero, tmp0); // if mask >= 0, set mask
  tmp2 = __lasx_xvand_v(tmp0, mask); // maskout the in2 < ones
  return __lasx_xvshuf_b(a, zero, tmp2);
}

LA_INLINE vreg_t vset(const float val) {
  FloatInt fi_tmpval = {.f = val};
  return (__m256)__lasx_xvreplgr2vr_w(fi_tmpval.i);
}

LA_INLINE ivreg_t ivset(const char i) { return __lasx_xvreplgr2vr_b(i); }
LA_INLINE hivreg_t hivset(const char i) { return __lsx_vreplgr2vr_b(i); }

LA_INLINE ivreg_t extend(hivreg_t h) {
  __m128i sign = __lsx_vslti_b(h, 0);
  __m128i vlo = __lsx_vilvl_b(sign, h);
  __m128i vhi = __lsx_vilvh_b(sign, h);
  return lasx_set_q(vhi, vlo);
}

LA_INLINE hivreg_t trunc(ivreg_t i, int select) {
  return (select) ? lasx_extracti128_hi(i) : lasx_extracti128_lo(i);
}

LA_INLINE vreg_t to_float(ivreg_t i) { return __lasx_xvffint_s_w(i); }

// x + y: f32
LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return __lasx_xvfadd_s(x, y); }

// x + y: int32
LA_INLINE ivreg_t add(ivreg_t x, ivreg_t y) { return __lasx_xvadd_w(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return __lasx_xvfmadd_s(x, y, z);
}

// x - y: f32
LA_INLINE vreg_t sub(vreg_t x, vreg_t y) { return __lasx_xvfsub_s(x, y); }

// x - y: int8
LA_INLINE ivreg_t sub(ivreg_t x, ivreg_t y) { return __lasx_xvsub_b(x, y); }

// x * y: f32
LA_INLINE vreg_t mul(vreg_t x, vreg_t y) { return __lasx_xvfmul_s(x, y); }

// x: int8 * y: int8 -> int16
LA_INLINE ivreg_t mul(ivreg_t a, ivreg_t b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_w_h(a, b);
  tmp2 = __lasx_xvmulwod_w_h(a, b);
  return __lasx_xvadd_w(tmp1, tmp2);
};

// x: uint8 * y: int8 -> int16
LA_INLINE ivreg_t mul_ubs(ivreg_t a, ivreg_t b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_h_b(a, b);
  tmp2 = __lasx_xvmulwod_h_b(a, b);
  return __lasx_xvsadd_h(tmp1, tmp2);
};

// x & y
LA_INLINE ivreg_t _and(ivreg_t x, ivreg_t y) { return __lasx_xvand_v(x, y); }
LA_INLINE hivreg_t _and(hivreg_t x, hivreg_t y) { return __lsx_vand_v(x, y); }

// (~x) & y: int
LA_INLINE ivreg_t andnot(ivreg_t x, ivreg_t y) { return __lasx_xvandn_v(x, y); }

// x | y
LA_INLINE ivreg_t _or(ivreg_t x, ivreg_t y) { return __lasx_xvor_v(x, y); }

// x: int16 >> n
LA_INLINE ivreg_t logic_shift_right(ivreg_t x, int n) {
  return __lasx_xvsrli_h(x, n);
}
LA_INLINE hivreg_t logic_shift_right(hivreg_t x, int n) {
  return __lsx_vsrli_h(x, n);
}

// TODO
LA_INLINE ivreg_t shuffle(ivreg_t x, ivreg_t y) { return lasx_shuffle_b(x, y); }

LA_INLINE __m256i lasx_insertf128(__m128i x, __m128i y) {
  return lasx_set_q(x, y);
}
LA_INLINE ivreg_t concat(hivreg_t a, hivreg_t b) {
  return lasx_insertf128(a, b);
}

// 32 bits -> 256 bits
LA_INLINE ivreg_t spread_bits(const uint8_t *x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m256i shuf_mask = lasx_set_d(0x0303030303030303, 0x0202020202020202,
                                       0x0101010101010101, 0x0000000000000000);

  __m256i bytes = lasx_shuffle_b(__lasx_xvreplgr2vr_w(x32), shuf_mask);
  const __m256i bit_mask = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
  bytes = __lasx_xvor_v(bytes, bit_mask);
  return __lasx_xvseq_b(bytes, __lasx_xvreplgr2vr_d(-1));
}

// sum 4 f32 -> f32
LA_INLINE __m128 lasx_extractf128(__m256 a, int pos) {
  __m128 ret;
  if (pos == 0) {
    ret = (__m128)lasx_extracti128_lo((__m256i)a);
  } else {
    ret = (__m128)lasx_extracti128_hi((__m256i)a);
  }
  return ret;
}

// sum 8 f32 -> f32
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
// load from uint8_t*
LA_INLINE ivreg_t load(const char *p) {
  return __lasx_xvld((const __m256i *)p, 0);
}
LA_INLINE hivreg_t loadh(const char *p) {
  return __lsx_vld((const __m128i *)p, 0);
}

// load from quantized block
// Q4_0
LA_INLINE ivreg_t load_quants(const block_q4_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q4_0>::super_block_size == 32);
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q4_1
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q4_1>::super_block_size == 32);
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q5_0
LA_INLINE ivreg_t load_quants(const block_q5_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q5_0>::super_block_size == 32);
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q5_1
LA_INLINE ivreg_t load_quants(const block_q5_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q5_1>::super_block_size == 32);
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}

// Q8_0
LA_INLINE ivreg_t load_quants(const block_q8_0 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q8_0>::super_block_size == 32);
  return __lasx_xvld((const __m256i *)(p->qs), 0);
}

// Q8_1
LA_INLINE ivreg_t load_quants(const block_q8_1 *p) {
  static_assert(ggml_type_trait<GGML_TYPE_Q8_1>::super_block_size == 32);
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

LA_INLINE vreg_t mul_sum_i8_pairs_float(const ivreg_t x, const ivreg_t y) {

  // Get absolute values of x vectors
  const __m256i ax = __lasx_xvsigncov_b(x, x);
  // Sign the values of the y vectors
  const __m256i sy = __lasx_xvsigncov_b(x, y);

  return mul_sum_us8_pairs_float(ax, sy);
}

} // namespace simd

#endif // LAMM_SIMD_LOONGARCH_H
