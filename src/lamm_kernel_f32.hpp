#ifndef LAMM_KERNEL_F32_HPP
#define LAMM_KERNEK_F32_HPP

#include "lamm_common.h"
#include <cstdint>

LA_INLINE void lamm_naive_kernel(const float *a, const float *b, float *c,
                                 int64_t lda, int64_t ldb, int64_t ldc, int i,
                                 int j, int K) {
  float sum = 0;
  for (int k = 0; k < K; k++) {
    sum += a[i * lda + k] * b[j * ldb + k];
  }
  c[j * ldc + i] = sum;
}

LA_INLINE void lamm_simd_kernel(const float *a, const float *b, float *c,
                                int64_t lda, int64_t ldb, int64_t ldc, int i,
                                int j, int K) {
  simd::vreg_t vc = {0}, va = {0}, vb = {0};
  for (int k = 0; k < K; k += simd::kF32PerVec) {
    va = simd::load(a + i * lda + k);
    vb = simd::load(b + j * ldb + k);
    vc = simd::madd(va, vb, vc);
  }
  c[j * ldc + i] = simd::reduce_sum(vc);
}

template <int B0, int B1>
void lamm_simd_block_kernel(const float *a, const float *b, float *c,
                            int64_t lda, int64_t ldb, int64_t ldc, int i, int j,
                            int K) {

  static_assert(B0 > 0 && B0 <= 5);
  static_assert(B1 > 0 && B1 <= 5);

  using namespace simd;
  [[maybe_unused]] vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0},
                          vc04 = {0};
  [[maybe_unused]] vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0},
                          vc14 = {0};
  [[maybe_unused]] vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0},
                          vc24 = {0};
  [[maybe_unused]] vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0},
                          vc34 = {0};
  [[maybe_unused]] vreg_t vc40 = {0}, vc41 = {0}, vc42 = {0}, vc43 = {0},
                          vc44 = {0};
  [[maybe_unused]] vreg_t vb0 = {0}, vb1 = {0}, vb2 = {0}, vb3 = {0}, vb4 = {0};
  vreg_t va = {0};

  for (int l = 0; l < K; l += kF32PerVec) {

#define FN(N1)                                                                 \
  if constexpr (B1 > N1) {                                                     \
    vb##N1 = load(b + ldb * (j + N1) + l);                                     \
  }
    LOOP(FN, 5);
#undef FN

#define INNER_FN(N0, N1)                                                       \
  if constexpr (B1 > N1) {                                                     \
    vc##N0##N1 = madd(va, vb##N1, vc##N0##N1);                                 \
  }
#define OUTER_FN(N0)                                                           \
  if constexpr (B0 > N0) {                                                     \
    va = load(a + lda * (i + N0) + l);                                         \
    LOOP_INNER(INNER_FN, N0, 5);                                               \
  }
    LOOP(OUTER_FN, 5);
#undef INNER_FN
#undef OUTER_FN
  } // end for

#define INNER_FN(N1, N0)                                                       \
  if constexpr (B0 > N0) {                                                     \
    c[ldc * (j + N1) + (i + N0)] = reduce_sum(vc##N0##N1);                     \
  }
#define OUTER_FN(N1)                                                           \
  if constexpr (B1 > N1) {                                                     \
    LOOP_INNER(INNER_FN, N1, 5);                                               \
  }
  LOOP(OUTER_FN, 5)
#undef INNER_FN
#undef OUTER_FN
}

#endif // LAMM_KERNEL_F32_HPP