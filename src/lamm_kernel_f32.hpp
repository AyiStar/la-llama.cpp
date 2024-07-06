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

    if constexpr (B1 > 0) {
      vb0 = load(b + ldb * (j + 0) + l);
    }
    if constexpr (B1 > 1) {
      vb1 = load(b + ldb * (j + 1) + l);
    }
    if constexpr (B1 > 2) {
      vb2 = load(b + ldb * (j + 2) + l);
    }
    if constexpr (B1 > 3) {
      vb3 = load(b + ldb * (j + 3) + l);
    }
    if constexpr (B1 > 4) {
      vb4 = load(b + ldb * (j + 4) + l);
    }

    if constexpr (B0 > 0) {
      va = load(a + lda * (i + 0) + l);
      if constexpr (B1 > 0) {
        vc00 = madd(va, vb0, vc00);
      }
      if constexpr (B1 > 1) {
        vc01 = madd(va, vb1, vc01);
      }
      if constexpr (B1 > 2) {
        vc02 = madd(va, vb2, vc02);
      }
      if constexpr (B1 > 3) {
        vc03 = madd(va, vb3, vc03);
      }
      if constexpr (B1 > 4) {
        vc04 = madd(va, vb4, vc04);
      }
    }

    if constexpr (B0 > 1) {
      va = load(a + lda * (i + 1) + l);
      if constexpr (B1 > 0) {
        vc10 = madd(va, vb0, vc10);
      }
      if constexpr (B1 > 1) {
        vc11 = madd(va, vb1, vc11);
      }
      if constexpr (B1 > 2) {
        vc12 = madd(va, vb2, vc12);
      }
      if constexpr (B1 > 3) {
        vc13 = madd(va, vb3, vc13);
      }
      if constexpr (B1 > 4) {
        vc14 = madd(va, vb4, vc14);
      }
    }

    if constexpr (B0 > 2) {
      va = load(a + lda * (i + 2) + l);
      if constexpr (B1 > 0) {
        vc20 = madd(va, vb0, vc20);
      }
      if constexpr (B1 > 1) {
        vc21 = madd(va, vb1, vc21);
      }
      if constexpr (B1 > 2) {
        vc22 = madd(va, vb2, vc22);
      }
      if constexpr (B1 > 3) {
        vc23 = madd(va, vb3, vc23);
      }
      if constexpr (B1 > 4) {
        vc24 = madd(va, vb4, vc24);
      }
    }

    if constexpr (B0 > 3) {
      va = load(a + lda * (i + 3) + l);
      if constexpr (B1 > 0) {
        vc30 = madd(va, vb0, vc30);
      }
      if constexpr (B1 > 1) {
        vc31 = madd(va, vb1, vc31);
      }
      if constexpr (B1 > 2) {
        vc32 = madd(va, vb2, vc32);
      }
      if constexpr (B1 > 3) {
        vc33 = madd(va, vb3, vc33);
      }
      if constexpr (B1 > 4) {
        vc34 = madd(va, vb4, vc34);
      }
    }

    if constexpr (B0 > 4) {
      va = load(a + lda * (i + 4) + l);
      if constexpr (B1 > 0) {
        vc40 = madd(va, vb0, vc40);
      }
      if constexpr (B1 > 1) {
        vc41 = madd(va, vb1, vc41);
      }
      if constexpr (B1 > 2) {
        vc42 = madd(va, vb2, vc42);
      }
      if constexpr (B1 > 3) {
        vc43 = madd(va, vb3, vc43);
      }
      if constexpr (B1 > 4) {
        vc44 = madd(va, vb4, vc44);
      }
    }
  }

  if constexpr (B1 > 0) {
    if constexpr (B0 > 0) {
      c[ldc * (j + 0) + (i + 0)] = reduce_sum(vc00);
    }
    if constexpr (B0 > 1) {
      c[ldc * (j + 0) + (i + 1)] = reduce_sum(vc10);
    }
    if constexpr (B0 > 2) {
      c[ldc * (j + 0) + (i + 2)] = reduce_sum(vc20);
    }
    if constexpr (B0 > 3) {
      c[ldc * (j + 0) + (i + 3)] = reduce_sum(vc30);
    }
    if constexpr (B0 > 4) {
      c[ldc * (j + 0) + (i + 4)] = reduce_sum(vc40);
    }
  }
  if constexpr (B1 > 1) {
    if constexpr (B0 > 0) {
      c[ldc * (j + 1) + (i + 0)] = reduce_sum(vc01);
    }
    if constexpr (B0 > 1) {
      c[ldc * (j + 1) + (i + 1)] = reduce_sum(vc11);
    }
    if constexpr (B0 > 2) {
      c[ldc * (j + 1) + (i + 2)] = reduce_sum(vc21);
    }
    if constexpr (B0 > 3) {
      c[ldc * (j + 1) + (i + 3)] = reduce_sum(vc31);
    }
    if constexpr (B0 > 4) {
      c[ldc * (j + 1) + (i + 4)] = reduce_sum(vc41);
    }
  }
  if constexpr (B1 > 2) {
    if constexpr (B0 > 0) {
      c[ldc * (j + 2) + (i + 0)] = reduce_sum(vc02);
    }
    if constexpr (B0 > 1) {
      c[ldc * (j + 2) + (i + 1)] = reduce_sum(vc12);
    }
    if constexpr (B0 > 2) {
      c[ldc * (j + 2) + (i + 2)] = reduce_sum(vc22);
    }
    if constexpr (B0 > 3) {
      c[ldc * (j + 2) + (i + 3)] = reduce_sum(vc32);
    }
    if constexpr (B0 > 4) {
      c[ldc * (j + 2) + (i + 4)] = reduce_sum(vc42);
    }
  }
  if constexpr (B1 > 3) {
    if constexpr (B0 > 0) {
      c[ldc * (j + 3) + (i + 0)] = reduce_sum(vc03);
    }
    if constexpr (B0 > 1) {
      c[ldc * (j + 3) + (i + 1)] = reduce_sum(vc13);
    }
    if constexpr (B0 > 2) {
      c[ldc * (j + 3) + (i + 2)] = reduce_sum(vc23);
    }
    if constexpr (B0 > 3) {
      c[ldc * (j + 3) + (i + 3)] = reduce_sum(vc33);
    }
    if constexpr (B0 > 4) {
      c[ldc * (j + 3) + (i + 4)] = reduce_sum(vc43);
    }
  }
  if constexpr (B1 > 4) {
    if constexpr (B0 > 0) {
      c[ldc * (j + 4) + (i + 0)] = reduce_sum(vc04);
    }
    if constexpr (B0 > 1) {
      c[ldc * (j + 4) + (i + 1)] = reduce_sum(vc14);
    }
    if constexpr (B0 > 2) {
      c[ldc * (j + 4) + (i + 2)] = reduce_sum(vc24);
    }
    if constexpr (B0 > 3) {
      c[ldc * (j + 4) + (i + 3)] = reduce_sum(vc34);
    }
    if constexpr (B0 > 4) {
      c[ldc * (j + 4) + (i + 4)] = reduce_sum(vc44);
    }
  }
}

#endif // LAMM_KERNEL_F32_HPP