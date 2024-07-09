#ifndef LAMM_KERNEL_Q5_0_HPP
#define LAMM_KERNEL_Q5_0_HPP

#include "lamm_common.h"
#include <cstdint>

/*
#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
*/

LA_INLINE void lamm_naive_kernel(const block_q5_0 *a, const block_q8_0 *b,
                                 float *c, int64_t lda, int64_t ldb,
                                 int64_t ldc, int i, int j, int K) {
  constexpr int Q = ggml_type_trait<GGML_TYPE_Q5_0>::super_block_size;
  float sum = 0.0;
  for (int k = 0; k < K; k++) {
    uint32_t qh;
    memcpy(&qh, a[i].qh, sizeof(qh));
    const auto *aik = a + (i * lda + k);
    const auto *bjk = b + (j * ldb + k);
    int sumi = 0;
    for (int h = 0; h < Q / 2; h++) {
      const uint8_t xh_0 = ((qh & (1u << (h + 0 ))) >> (h + 0 )) << 4;
      const uint8_t xh_1 = ((qh & (1u << (h + 16))) >> (h + 12));
      const int32_t x0 = ((a[i].qs[j] & 0x0F) | (~xh_0 & 0xF0)); // ((a[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((a[i].qs[j] >>   4) | (~xh_1 & 0xF0));  // ((a[i].qs[j] >>   4) | xh_1) - 16;
      sumi += (x0 * b[i].qs[j]) + (x1 * b[i].qs[j + Q/2]);
    }
    sum += (GGML_FP16_TO_FP32(aik->d) * GGML_FP16_TO_FP32(bjk->d)) * sumi;
  }
  c[j * ldc + i] = sum;
}

LA_INLINE void lamm_simd_kernel(const block_q5_0 *a, const block_q8_0 *b,
                                float *c, int64_t lda, int64_t ldb, int64_t ldc,
                                int i, int j, int K) {
//   simd::vreg_t acc = {0};
//   const auto *ai = a + (i * lda);
//   const auto *bj = b + (j * ldb);
//   for (int k = 0; k < K; k++, ai++, bj++) {
//     const simd::vreg_t ad = simd::vset(GGML_FP16_TO_FP32(ai->d));
//     const simd::vreg_t bd = simd::vset(GGML_FP16_TO_FP32(bj->d));
//     const __m256 adbd = simd::mul(ad, bd);
//     simd::ivreg_t va_qs = simd::load_quants(ai);
//     simd::ivreg_t vb_qs = simd::load_quants(bj);
//     const simd::vreg_t xy = simd::mul_sum_us8_pairs_float(va_qs, vb_qs);
//     acc = simd::madd(adbd, xy, acc);
//   }
//   c[j * ldc + i] = simd::reduce_sum(acc);
}

template <int B0, int B1>
LA_INLINE void lamm_simd_block_kernel(const block_q5_0 *a, const block_q8_0 *b,
                                      float *c, int64_t lda, int64_t ldb,
                                      int64_t ldc, int i, int j, int K) {

//   static_assert(B0 > 0 && B0 <= 4);
//   static_assert(B1 > 0 && B1 <= 4);

//   using namespace simd;

//   ivreg_t va_qs = {0};
//   simd::vreg_t vad = {0};
//   [[maybe_unused]] ivreg_t vb0_qs = {0}, vb1_qs = {0}, vb2_qs = {0},
//                            vb3_qs = {0};
//   [[maybe_unused]] simd::vreg_t vbd0, vbd1, vbd2, vbd3;
//   [[maybe_unused]] simd::vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0};
//   [[maybe_unused]] simd::vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0};
//   [[maybe_unused]] simd::vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0};
//   [[maybe_unused]] simd::vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0};

//   [[maybe_unused]] const auto *ai0{a + ((i + 0) * lda)};
//   [[maybe_unused]] const auto *ai1{a + ((i + 1) * lda)};
//   [[maybe_unused]] const auto *ai2{a + ((i + 2) * lda)};
//   [[maybe_unused]] const auto *ai3{a + ((i + 3) * lda)};
//   [[maybe_unused]] const auto *bj0{b + ((j + 0) * ldb)};
//   [[maybe_unused]] const auto *bj1{b + ((j + 1) * ldb)};
//   [[maybe_unused]] const auto *bj2{b + ((j + 2) * ldb)};
//   [[maybe_unused]] const auto *bj3{b + ((j + 3) * ldb)};

//   for (int k = 0; k < K; k++) {

// #define FN(N1)                                                                 \
//   if constexpr (B1 > N1) {                                                     \
//     vb##N1##_qs = load_quants(bj##N1 + k);                                     \
//     vbd##N1 = vset(GGML_FP16_TO_FP32(bj##N1[k].d));                            \
//   }
//     LOOP(FN, 4)
// #undef FN

// #define INNER_FN(N0, N1)                                                       \
//   if constexpr (B1 > 0) {                                                      \
//     vc##N0##N1 =                                                               \
//         madd(mul(vad, vbd##N1), mul_sum_us8_pairs_float(va_qs, vb##N1##_qs),   \
//              vc##N0##N1);                                                      \
//   }
// #define OUTER_FN(N0)                                                           \
//   if constexpr (B0 > N0) {                                                     \
//     va_qs = load_quants(ai##N0 + k);                                           \
//     vad = vset(GGML_FP16_TO_FP32(ai##N0[k].d));                                \
//     LOOP_INNER(INNER_FN, N0, 4)                                                \
//   }
//     LOOP(OUTER_FN, 4)
// #undef INNER_FN
// #undef OUTER_FN
//   }

// #define INNER_FN(N0, N1)                                                       \
//   if constexpr (B1 > N1) {                                                     \
//     c[ldc * (j + N1) + (i + N0)] = reduce_sum(vc##N0##N1);                     \
//   }
// #define OUTER_FN(N0)                                                           \
//   if constexpr (B0 > N0) {                                                     \
//     LOOP_INNER(INNER_FN, N0, 4)                                                \
//   }
//   LOOP(OUTER_FN, 4)
// #undef INNER_FN
// #undef OUTER_FN
}

#endif // LAMM_KERNEL_Q5_0_HPP