#ifndef LAMM_KERNEL_Q4_1_HPP
#define LAMM_KERNEL_Q4_1_HPP

#include "lamm_common.h"
#include <cstdint>

LA_INLINE void lamm_naive_kernel(const block_q4_1 *a, const block_q8_1 *b,
                                 float *c, int64_t lda, int64_t ldb,
                                 int64_t ldc, int i, int j, int K) {
  constexpr int Q = QK8_1;
  float sum = 0.0;
  for (int k = 0; k < K; k++) {
    const auto *aik = a + (i * lda + k);
    const auto *bjk = b + (j * ldb + k);
    int sumi = 0;
    for (int h = 0; h < Q / 2; h++) {
      sumi += (aik->qs[h] & 0x0F) * (bjk->qs[h]);
      sumi += (aik->qs[h] >> 4) * (bjk->qs[h + Q / 2]);
    }
    sum += (GGML_FP16_TO_FP32(aik->d) * GGML_FP16_TO_FP32(bjk->d)) * sumi +
           GGML_FP16_TO_FP32(aik->m) * GGML_FP16_TO_FP32(bjk->s);
    // printf("sumi = %d, aik->d=%.4f, bjk->d=%.4f, aik->m=%.4f, bjk->s=%.4f at
    // (i=%d, j=%d, k=%d)\n", sumi, GGML_FP16_TO_FP32(aik->d),
    // GGML_FP16_TO_FP32(bjk->d), GGML_FP16_TO_FP32(aik->m),
    // GGML_FP16_TO_FP32(bjk->s), i, j, k);
    // printf("ldc = %ld\n", ldc);
  }
  c[j * ldc + i] = sum;
}

LA_INLINE void lamm_simd_kernel(const block_q4_1 *a, const block_q8_1 *b,
                                float *c, int64_t lda, int64_t ldb, int64_t ldc,
                                int i, int j, int K) {
  float summs = 0;
  simd::vreg_t acc = {0};
  const auto *ai = a + (i * lda);
  const auto *bj = b + (j * ldb);
  for (int k = 0; k < K; k++, ai++, bj++) {
    summs += GGML_FP16_TO_FP32(ai->m) * GGML_FP16_TO_FP32(bj->s);
    const simd::vreg_t ad = simd::vset(GGML_FP16_TO_FP32(ai->d));
    const simd::vreg_t bd = simd::vset(GGML_FP16_TO_FP32(bj->d));
    const __m256 adbd = simd::mul(ad, bd);
    simd::ivreg_t va_qs = simd::load_quants(ai);
    simd::ivreg_t vb_qs = simd::load_quants(bj);
    const simd::vreg_t xy = simd::mul_sum_us8_pairs_float(va_qs, vb_qs);
    acc = simd::madd(adbd, xy, acc);
  }
  c[j * ldc + i] = simd::reduce_sum(acc) + summs;
}

template <int B0, int B1>
LA_INLINE void lamm_simd_block_kernel(const block_q4_1 *a, const block_q8_1 *b,
                                      float *c, int64_t lda, int64_t ldb,
                                      int64_t ldc, int i, int j, int K) {

  static_assert(B0 > 0 && B0 <= 4);
  static_assert(B1 > 0 && B1 <= 4);

  using namespace simd;

  ivreg_t va_qs = {0};
  simd::vreg_t vad = {0};
  [[maybe_unused]] ivreg_t vb0_qs = {0}, vb1_qs = {0}, vb2_qs = {0},
                           vb3_qs = {0};
  [[maybe_unused]] simd::vreg_t vbd0, vbd1, vbd2, vbd3;
  [[maybe_unused]] simd::vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0};
  [[maybe_unused]] simd::vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0};
  [[maybe_unused]] simd::vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0};
  [[maybe_unused]] simd::vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0};

  float summs[B0][B1] = {0};
  [[maybe_unused]] const auto *ai0{a + ((i + 0) * lda)};
  [[maybe_unused]] const auto *ai1{a + ((i + 1) * lda)};
  [[maybe_unused]] const auto *ai2{a + ((i + 2) * lda)};
  [[maybe_unused]] const auto *ai3{a + ((i + 3) * lda)};
  [[maybe_unused]] const auto *bj0{b + ((j + 0) * ldb)};
  [[maybe_unused]] const auto *bj1{b + ((j + 1) * ldb)};
  [[maybe_unused]] const auto *bj2{b + ((j + 2) * ldb)};
  [[maybe_unused]] const auto *bj3{b + ((j + 3) * ldb)};

  for (int k = 0; k < K; k++) {

    if constexpr (B1 > 0) {
      vb0_qs = load_quants(bj0 + k);
      vbd0 = vset(GGML_FP16_TO_FP32(bj0[k].d));
    }
    if constexpr (B1 > 1) {
      vb1_qs = load_quants(bj1 + k);
      vbd1 = vset(GGML_FP16_TO_FP32(bj1[k].d));
    }
    if constexpr (B1 > 2) {
      vb2_qs = load_quants(bj2 + k);
      vbd2 = vset(GGML_FP16_TO_FP32(bj2[k].d));
    }
    if constexpr (B1 > 3) {
      vb3_qs = load_quants(bj3 + k);
      vbd3 = vset(GGML_FP16_TO_FP32(bj3[k].d));
    }

    if constexpr (B0 > 0) {
      va_qs = load_quants(ai0 + k);
      vad = vset(GGML_FP16_TO_FP32(ai0[k].d));
      if constexpr (B1 > 0) {
        summs[0][0] +=
            GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc00 =
            madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs), vc00);
      }
      if constexpr (B1 > 1) {
        summs[0][1] +=
            GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc01 =
            madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs), vc01);
      }
      if constexpr (B1 > 2) {
        summs[0][2] +=
            GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc02 =
            madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs), vc02);
      }
      if constexpr (B1 > 3) {
        summs[0][3] +=
            GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc03 =
            madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs), vc03);
      }
    }

    if constexpr (B0 > 1) {
      va_qs = load_quants(ai1 + k);
      vad = vset(GGML_FP16_TO_FP32(ai1[k].d));
      if constexpr (B1 > 0) {
        summs[1][0] +=
            GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc10 =
            madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs), vc10);
      }
      if constexpr (B1 > 1) {
        summs[1][1] +=
            GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc11 =
            madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs), vc11);
      }
      if constexpr (B1 > 2) {
        summs[1][2] +=
            GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc12 =
            madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs), vc12);
      }
      if constexpr (B1 > 3) {
        summs[1][3] +=
            GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc13 =
            madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs), vc13);
      }
    }

    if constexpr (B0 > 2) {
      va_qs = load_quants(ai2 + k);
      vad = vset(GGML_FP16_TO_FP32(ai2[k].d));
      if constexpr (B1 > 0) {
        summs[2][0] +=
            GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc20 =
            madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs), vc20);
      }
      if constexpr (B1 > 1) {
        summs[2][1] +=
            GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc21 =
            madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs), vc21);
      }
      if constexpr (B1 > 2) {
        summs[2][2] +=
            GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc22 =
            madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs), vc22);
      }
      if constexpr (B1 > 3) {
        summs[2][3] +=
            GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc23 =
            madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs), vc23);
      }
    }

    if constexpr (B0 > 3) {
      va_qs = load_quants(ai3 + k);
      vad = vset(GGML_FP16_TO_FP32(ai3[k].d));
      if constexpr (B1 > 0) {
        summs[3][0] +=
            GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc30 =
            madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs), vc30);
      }
      if constexpr (B1 > 1) {
        summs[3][1] +=
            GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc31 =
            madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs), vc31);
      }
      if constexpr (B1 > 2) {
        summs[3][2] +=
            GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc32 =
            madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs), vc32);
      }
      if constexpr (B1 > 3) {
        summs[3][3] +=
            GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc33 =
            madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs), vc33);
      }
    }
  }

  if constexpr (B0 > 0) {
    if constexpr (B1 > 0) {
      c[ldc * (j + 0) + (i + 0)] = reduce_sum(vc00) + summs[0][0];
    }
    if constexpr (B1 > 1) {
      c[ldc * (j + 1) + (i + 0)] = reduce_sum(vc01) + summs[0][1];
    }
    if constexpr (B1 > 2) {
      c[ldc * (j + 2) + (i + 0)] = reduce_sum(vc02) + summs[0][2];
    }
    if constexpr (B1 > 3) {
      c[ldc * (j + 3) + (i + 0)] = reduce_sum(vc03) + summs[0][3];
    }
  }
  if constexpr (B0 > 1) {
    if constexpr (B1 > 0) {
      c[ldc * (j + 0) + (i + 1)] = reduce_sum(vc10) + summs[1][0];
    }
    if constexpr (B1 > 1) {
      c[ldc * (j + 1) + (i + 1)] = reduce_sum(vc11) + summs[1][1];
    }
    if constexpr (B1 > 2) {
      c[ldc * (j + 2) + (i + 1)] = reduce_sum(vc12) + summs[1][2];
    }
    if constexpr (B1 > 3) {
      c[ldc * (j + 3) + (i + 1)] = reduce_sum(vc13) + summs[1][3];
    }
  }
  if constexpr (B0 > 2) {
    if constexpr (B1 > 0) {
      c[ldc * (j + 0) + (i + 2)] = reduce_sum(vc20) + summs[2][0];
    }
    if constexpr (B1 > 1) {
      c[ldc * (j + 1) + (i + 2)] = reduce_sum(vc21) + summs[2][1];
    }
    if constexpr (B1 > 2) {
      c[ldc * (j + 2) + (i + 2)] = reduce_sum(vc22) + summs[2][2];
    }
    if constexpr (B1 > 3) {
      c[ldc * (j + 3) + (i + 2)] = reduce_sum(vc23) + summs[2][3];
    }
  }
  if constexpr (B0 > 3) {
    if constexpr (B1 > 0) {
      c[ldc * (j + 0) + (i + 3)] = reduce_sum(vc30) + summs[3][0];
    }
    if constexpr (B1 > 1) {
      c[ldc * (j + 1) + (i + 3)] = reduce_sum(vc31) + summs[3][1];
    }
    if constexpr (B1 > 2) {
      c[ldc * (j + 2) + (i + 3)] = reduce_sum(vc32) + summs[3][2];
    }
    if constexpr (B1 > 3) {
      c[ldc * (j + 3) + (i + 3)] = reduce_sum(vc33) + summs[3][3];
    }
  }
}

#endif // LAMM_KERNEL_Q4_1_HPP