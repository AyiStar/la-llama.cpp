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

//// opt level control
#if defined(LAMM_OPT_LEVEL)
constexpr int kOptLevel = LAMM_OPT_LEVEL;
#else
constexpr int kOptLevel = 3;
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

// vector -> f32
LA_INLINE float reduce_sum(vreg_t x) {
  float res{0};
  float *tmp_p = (float *)&x;
  res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] +
        tmp_p[6] + tmp_p[7];
  return res;
}

// load from float*
LA_INLINE vreg_t load(const float *p) { return (vreg_t)__lasx_xvld(p, 0); }
// load from quantized block
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}
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
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  __m128i qs =
      _mm_loadu_si128((const __m128i *)(p->qs)); // load squeezed 4-bit qs
  return _mm256_and_si256( // mask higher 4 bits for each uint8
      _mm256_set1_epi8(15),
      _mm256_insertf128_si256( // copy and expand
          _mm256_castsi128_si256(qs), _mm_srli_epi16(qs, 4), 1));
}
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

namespace impl {

struct Matrix {
  void *data;
  ggml_type type;
  int row;
  int col;
  int64_t ld;
};

template <ggml_type dtype>
LA_NOINLINE void gemm(const Matrix &A, const Matrix &B, const Matrix &C,
                      int ith, int nth);

template <ggml_type dtype>
LA_INLINE void gemm_naive(const Matrix &A, const Matrix &B, const Matrix &C,
                          int ith, int nth);

template <ggml_type dtype>
LA_INLINE void gemm_simd(const Matrix &A, const Matrix &B, const Matrix &C,
                         int ith, int nth);

template <ggml_type dtype>
LA_INLINE void gemm_block_simd(const Matrix &A, const Matrix &B,
                               const Matrix &C, int ith, int nth);

// the real gemm function
template <ggml_type dtype>
void gemm(const Matrix &A, const Matrix &B, const Matrix &C, int ith, int nth) {
  if constexpr (kOptLevel == 1) {
    gemm_naive<dtype>(A, B, C, ith, nth);
  } else if constexpr (kOptLevel == 2) {
    gemm_simd<dtype>(A, B, C, ith, nth);
  } else {
    gemm_block_simd<dtype>(A, B, C, ith, nth);
  }
}

template <ggml_type dtype>
LA_INLINE void gemm_naive(const Matrix &A, const Matrix &B, const Matrix &C,
                          int ith, int nth) {
  if constexpr (kDebug) {
    if (ith == 0) {
      std::cout << "naive implementation called with (" << A.row << ", " << A.col << ", " << B.col << ")" << std::endl;
    }
  }

  int M = C.row, N = C.col, K = A.col;
  int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};
  assert(M == A.row && N == B.col && K == B.row);
  assert(nth > 0);
  // split thread-local job by M
  int job_size = M / nth;
  int job_start = ith * job_size;
  int job_end = job_start + job_size;
  if (job_end > M) {
    job_end = M;
  }

  assert(C.type == GGML_TYPE_F32);
  if constexpr (dtype == GGML_TYPE_F32) {
    assert(A.type == dtype && B.type == dtype);
    float *a = (float *)(A.data), *b = (float *)(B.data),
          *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
          sum += a[i * lda + k] * b[j * ldb + k];
        }
        c[j * ldc + i] = sum;
      }
    }
    return;
  } else if constexpr (dtype == GGML_TYPE_Q4_1) {
    assert(A.type == dtype && B.type == GGML_TYPE_Q8_1);
    constexpr int Q = QK8_1;
    assert(K % Q == 0);
    auto *a = (block_q4_1 *)(A.data);
    auto *b = (block_q8_1 *)(B.data);
    auto *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
          const auto *aik = a + (i * lda + k);
          const auto *bjk = b + (j * ldb + k);
          int sumi = 0;
          for (int h = 0; h < Q / 2; h++) {
            sumi += (aik->qs[h] & 0x0F) * (bjk->qs[h]);
            sumi += (aik->qs[h] >> 4) * (bjk->qs[h + Q / 2]);
          }
          sum +=
              (GGML_FP16_TO_FP32(aik->d) * GGML_FP16_TO_FP32(bjk->d)) * sumi +
              GGML_FP16_TO_FP32(aik->m) * GGML_FP16_TO_FP32(bjk->s);
          // printf("sumi = %d, aik->m=%.2f, bjk->s=%.2f at (i=%d, j=%d,
          // k=%d)\n", sumi, GGML_FP16_TO_FP32(aik->m),
          // GGML_FP16_TO_FP32(bjk->s), i, j, k);
        }
        c[j * ldc + i] = sum;
        if constexpr (kDebug) {
          printf("C[%d, %d] = %f\n", i, j, sum);
          // double check
          float ref_dot_ret = 0;
          ggml_vec_dot_q4_1_q8_1(K * Q, &ref_dot_ret, 0, a + (i * lda), 0, b + (j * ldb), 0, 1);
          if (std::abs(ref_dot_ret - c[j * ldc + i]) > 1e-3) {
            std::cerr << "q4_1_q8_1 vec dot error: " << c[j * ldc + i] << " != " << ref_dot_ret << std::endl;
            assert(false);
          }
        }
        
      }
    }
    return;
  }
  assert(false); // unreachable
}

template <ggml_type dtype>
LA_INLINE void gemm_simd(const Matrix &A, const Matrix &B, const Matrix &C,
                         int ith, int nth) {
  int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

  // simd implementation
  if constexpr (kDebug) {
    if (ith == 0) {
      std::cout << "SIMD implementation called" << std::endl;
    }
  }
  int M = C.row, N = C.col, K = A.col;
  assert(M == A.row && N == B.col && K == B.row);
  assert(K % simd::kF32PerVec == 0);
  assert(nth > 0);
  // split thread-local job by M
  int job_size = M / nth;
  int job_start = ith * job_size;
  int job_end = job_start + job_size;
  if (job_end > M) {
    job_end = M;
  }

  if constexpr (dtype == GGML_TYPE_F32) {
    float *a = (float *)(A.data), *b = (float *)(B.data),
          *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        simd::vreg_t vc = {0}, va = {0}, vb = {0};
        for (int k = 0; k < K; k += simd::kF32PerVec) {
          va = simd::load(a + i * lda + k);
          vb = simd::load(b + j * ldb + k);
          vc = simd::madd(va, vb, vc);
        }
        c[j * ldc + i] = simd::reduce_sum(vc);
      }
    }
  } else if constexpr (dtype == GGML_TYPE_Q4_1) {
    assert(A.type == dtype && B.type == GGML_TYPE_Q8_1);
    auto *a = (block_q4_1 *)(A.data);
    auto *b = (block_q8_1 *)(B.data);
    auto *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
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
    }
  }
}

template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const float *a, const float *b, float *c,
                                 int64_t lda, int64_t ldb, int64_t ldc, int i,
                                 int j, int k);

template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const block_q4_1 *a, const block_q8_1 *b,
                                 float *c, int64_t lda, int64_t ldb,
                                 int64_t ldc, int i, int j, int k);

template <ggml_type dtype>
LA_INLINE void gemm_block_simd(const Matrix &A, const Matrix &B,
                               const Matrix &C, int ith, int nth) {

  // block and simd implementation
  if constexpr (kDebug) {
    if (ith == 0) {
      std::cout << "Block SIMD implementation called" << std::endl;
    }
  }
  int M = C.row, N = C.col, K = A.col;
  assert(M == A.row && N == B.col && K == B.row);
  assert(nth > 0);
  // split thread-local job by M
  int job_size = M / nth;
  int job_start = ith * job_size;
  int job_end = job_start + job_size;
  if (job_end > M) {
    job_end = M;
  }

  // assert ((job_end - job_start) % kBlockSize == 0);

  // first use KxK block

  constexpr int kBlockSize = (dtype == GGML_TYPE_F32) ? 5 : 4;
  int L0 = job_end - job_start, L1 = N;
  int ii = (L0 / kBlockSize * kBlockSize) + job_start;
  int jj = (L1 / kBlockSize * kBlockSize);
  int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

  if constexpr (dtype == GGML_TYPE_F32) {
    assert((K % simd::kF32PerVec) == 0);
    float *a = (float *)(A.data);
    float *b = (float *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < ii; i += kBlockSize) {
      for (int j = 0; j < jj; j += kBlockSize) {
        gemm_block_kernel<kBlockSize, kBlockSize>(a, b, c, lda, ldb, ldc, i, j,
                                                  K);
      }
      for (int j = jj; j < N; j++) {
        gemm_block_kernel<kBlockSize, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    for (int i = ii; i < job_end; i++) {
      for (int j = 0; j < jj; j += kBlockSize) {
        gemm_block_kernel<1, kBlockSize>(a, b, c, lda, ldb, ldc, i, j, K);
      }
      for (int j = jj; j < N; j++) {
        gemm_block_kernel<1, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  } else if constexpr (dtype == GGML_TYPE_Q4_1) {
    block_q4_1 *a = (block_q4_1 *)(A.data);
    block_q8_1 *b = (block_q8_1 *)(B.data);
    float *c = (float *)(C.data);
    // TODO duplicated code
    for (int i = job_start; i < ii; i += kBlockSize) {
      for (int j = 0; j < jj; j += kBlockSize) {
        gemm_block_kernel<kBlockSize, kBlockSize>(a, b, c, lda, ldb, ldc, i, j,
                                                  K);
      }
      for (int j = jj; j < N; j++) {
        gemm_block_kernel<kBlockSize, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    for (int i = ii; i < job_end; i++) {
      for (int j = 0; j < jj; j += kBlockSize) {
        gemm_block_kernel<1, kBlockSize>(a, b, c, lda, ldb, ldc, i, j, K);
      }
      for (int j = jj; j < N; j++) {
        gemm_block_kernel<1, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  }
}

template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const float *a, const float *b, float *c,
                                 int64_t lda, int64_t ldb, int64_t ldc, int i,
                                 int j, int k) {

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

  for (int l = 0; l < k; l += kF32PerVec) {

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

template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const block_q4_1 *a, const block_q8_1 *b,
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

} // namespace impl

// check if the gemm is suitable to be accelerated
// we assume that the basic assertions have been done
bool lamm_can_mul_mat(const struct ggml_compute_params *params,
                      const struct ggml_tensor *dst) {
  if (kOptLevel == 0) {
    return false;
  }
  if (params->type != GGML_TASK_TYPE_COMPUTE) {
    return false;
  }

  auto src0 = dst->src[0];
  auto src1 = dst->src[1];

  // contiguous check
  const bool src1_cont = ggml_is_contiguous(src1);
  enum ggml_type const vec_dot_type =
      ggml_internal_get_type_traits(src0->type).vec_dot_type;
  if ((src1->type == vec_dot_type) && !src1_cont) {
    return false;
  }
  if (src1->nb[0] != ggml_type_size(src1->type)) {
    return false;
  }

  // what types do we support?
  if (dst->type != GGML_TYPE_F32) {
    return false;
  }
  static const enum ggml_type supported_types[][2] = {
      {GGML_TYPE_F32, GGML_TYPE_F32},
      {GGML_TYPE_Q4_1, GGML_TYPE_Q8_1},
  };
  const int num_supported_types =
      sizeof(supported_types) / sizeof(supported_types[0]);
  enum ggml_type type0 = src0->type, type1 = vec_dot_type;
  bool support = false;
  for (int i = 0; i < num_supported_types; i++) {
    if (type0 == supported_types[i][0] && type1 == supported_types[i][1]) {
      support = true;
      break;
    }
  }
  if (!support) {
    // std::cout << "data type not supported" << std::endl;
    return false;
  }

  return true;
}

void lamm_mul_mat(const struct ggml_compute_params *params,
                  struct ggml_tensor *dst) {

  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];
  // enum ggml_type const vec_dot_type =
  // ggml_internal_get_type_traits(src0->type).vec_dot_type; const bool
  // use_wdata = (src1->type != vec_dot_type);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t r2 = ne12 / ne02;
  const int64_t r3 = ne13 / ne03;

  enum ggml_type const vec_dot_type =
      ggml_internal_get_type_traits(src0->type).vec_dot_type;
  const size_t row_size =
      ggml_row_size(vec_dot_type, ne10) / ggml_type_size(vec_dot_type);

  impl::Matrix A, B, C;

  A.type = src0->type;
  A.row = ne01;
  A.col = ne00 / ggml_blck_size(src0->type);
  A.ld = nb01 / ggml_type_size(src0->type);

  B.type = vec_dot_type;
  B.row = ne00 / ggml_blck_size(src0->type);
  B.col = ne11;
  B.ld = (src1->type == vec_dot_type)
             ? (nb11 / ggml_type_size(src1->type))
             : (row_size / ggml_type_size(vec_dot_type));

  C.type = dst->type;
  C.row = ne01;
  C.col = ne11;
  C.ld = nb1 / ggml_type_size(dst->type);

  decltype(impl::gemm<GGML_TYPE_F32>) *gemm_func = nullptr;
  if (A.type == GGML_TYPE_F32) {
    gemm_func = impl::gemm<GGML_TYPE_F32>;
  } else if (A.type == GGML_TYPE_Q4_1) {
    gemm_func = impl::gemm<GGML_TYPE_Q4_1>;
  }

  for (int64_t i13 = 0; i13 < ne13; i13++) {
    for (int64_t i12 = 0; i12 < ne12; i12++) {
      A.data = (char *)src0->data + i12 / r2 * nb02 + i13 / r3 * nb03;
      if (src1->type == vec_dot_type) {
        B.data = (char *)src1->data + i12 * nb12 + i13 * nb13;
      } else {
        B.data = (char *)(params->wdata) +
                 (i12 * ne11 + i13 * ne12 * ne11) * row_size;
      }
      C.data = (char *)dst->data + i12 * nb2 + i13 * nb3;
      // if (A.type == GGML_TYPE_F32) {
      //   impl::gemm<GGML_TYPE_F32>(A, B, C, params->ith, params->nth);
      // } else {
      //   impl::gemm<GGML_TYPE_Q4_1>(A, B, C, params->ith, params->nth);
      // }
      gemm_func(A, B, C, params->ith, params->nth);
    }
  }
}
