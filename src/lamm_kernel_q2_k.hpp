#ifndef LAMM_KERNEL_Q2_K_HPP
#define LAMM_KERNEL_Q2_K_HPP

#include "lamm_common.h"
#include <cstdint>

/*
# define QK_K 256
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
} block_q2_K;

typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
*/

LA_INLINE void lamm_naive_kernel(const block_q2_K *a, const block_q8_K *b,
                                 float *c, int64_t lda, int64_t ldb,
                                 int64_t ldc, int i, int j, int K) {
  constexpr int Q = ggml_type_trait<GGML_TYPE_Q2_K>::super_block_size;
  float sumf = 0.0;
  auto *aik = a + (i * lda);
  auto *bjk = b + (j * ldb);
  for (int k = 0; k < K; k++, aik++, bjk++) {
    const uint8_t * q2 = aik->qs;
        const  int8_t * q8 = bjk->qs;
        const uint8_t * sc = aik->scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += bjk->bsums[j] * (sc[j] >> 4);
        }

        const float dall = bjk->d * GGML_FP16_TO_FP32(aik->d);
        const float dmin = bjk->d * GGML_FP16_TO_FP32(aik->dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
  }
  c[j * ldc + i] = sumf;
}

LA_INLINE void lamm_simd_kernel(const block_q2_K *a, const block_q8_K *b,
                                float *c, int64_t lda, int64_t ldb, int64_t ldc,
                                int i, int j, int K) {
  simd::vreg_t acc = {0};
  auto *aik = a + (i * lda);
  auto *bjk = b + (j * ldb);
  for (int k = 0; k < K; k++, aik++, bjk++) {
    // TODO
  }
  c[j * ldc + i] = simd::reduce_sum(acc);
}

template <int B0, int B1>
LA_INLINE void lamm_simd_block_kernel(const block_q2_K *a, const block_q8_K *b,
                                      float *c, int64_t lda, int64_t ldb,
                                      int64_t ldc, int i, int j, int K) {

  static_assert(B0 > 0 && B0 <= 4);
  static_assert(B1 > 0 && B1 <= 4);

  using namespace simd;

  const ivreg_t voffset = ivset(8);

  ivreg_t va_qs = {0};
  simd::vreg_t vad = {0};
  [[maybe_unused]] ivreg_t vb0_qs = {0}, vb1_qs = {0}, vb2_qs = {0},
                           vb3_qs = {0};
  [[maybe_unused]] simd::vreg_t vbd0, vbd1, vbd2, vbd3;
  [[maybe_unused]] simd::vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0};
  [[maybe_unused]] simd::vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0};
  [[maybe_unused]] simd::vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0};
  [[maybe_unused]] simd::vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0};

  [[maybe_unused]] const auto *ai0{a + ((i + 0) * lda)};
  [[maybe_unused]] const auto *ai1{a + ((i + 1) * lda)};
  [[maybe_unused]] const auto *ai2{a + ((i + 2) * lda)};
  [[maybe_unused]] const auto *ai3{a + ((i + 3) * lda)};
  [[maybe_unused]] const auto *bj0{b + ((j + 0) * ldb)};
  [[maybe_unused]] const auto *bj1{b + ((j + 1) * ldb)};
  [[maybe_unused]] const auto *bj2{b + ((j + 2) * ldb)};
  [[maybe_unused]] const auto *bj3{b + ((j + 3) * ldb)};

  for (int k = 0; k < K; k++) {
    // TODO
  }
}

#endif // LAMM_KERNEL_Q2_K_HPP