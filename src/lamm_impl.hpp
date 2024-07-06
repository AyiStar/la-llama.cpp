#ifndef LAMM_IMPL_HPP
#define LAMM_IMPL_HPP

#include "lamm_common.h"
#include "lamm_kernel_f32.hpp"
#include "lamm_kernel_q4_1.hpp"
#include <cassert>

template <ggml_type GGMLType> class LAMMImpl {
public:
  using dtype = typename ggml_type_trait<GGMLType>::dtype;
  using vec_dot_dtype = typename ggml_type_trait<GGMLType>::vec_dot_dtype;

  static void matmul(const Matrix &A, const Matrix &B, const Matrix &C, int ith,
                     int nth) {
    if constexpr (kOptLevel == 1) {
      matmul_naive(A, B, C, ith, nth);
    } else if constexpr (kOptLevel == 2) {
      matmul_simd(A, B, C, ith, nth);
    } else {
      matmul_simd_block(A, B, C, ith, nth);
    }
  }

  LA_INLINE static void matmul_naive(const Matrix &A, const Matrix &B,
                                     const Matrix &C, int ith, int nth) {
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
    // TODO check B.type
    dtype *a = (dtype *)(A.data);
    vec_dot_dtype *b = (vec_dot_dtype *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        lamm_naive_kernel(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    return;
  }

  LA_INLINE static void matmul_simd(const Matrix &A, const Matrix &B,
                                    const Matrix &C, int ith, int nth) {
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

    float *a = (float *)(A.data), *b = (float *)(B.data),
          *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        lamm_simd_kernel(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  }

  LA_INLINE static void matmul_simd_block(const Matrix &A, const Matrix &B,
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
    constexpr int kBlockSize = 4;
    int L0 = job_end - job_start, L1 = N;
    int ii = (L0 / kBlockSize * kBlockSize) + job_start;
    int jj = (L1 / kBlockSize * kBlockSize);
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

    assert((K % simd::kF32PerVec) == 0);
    dtype *a = (dtype *)(A.data);
    vec_dot_dtype *b = (vec_dot_dtype *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < ii; i += kBlockSize) {
      for (int j = 0; j < jj; j += kBlockSize) {
        lamm_simd_block_kernel<kBlockSize, kBlockSize>(a, b, c, lda, ldb, ldc,
                                                       i, j, K);
      }
      for (int j = jj; j < N; j++) {
        lamm_simd_block_kernel<kBlockSize, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    for (int i = ii; i < job_end; i++) {
      for (int j = 0; j < jj; j += kBlockSize) {
        lamm_simd_block_kernel<1, kBlockSize>(a, b, c, lda, ldb, ldc, i, j, K);
      }
      for (int j = jj; j < N; j++) {
        lamm_simd_block_kernel<1, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  }
};

#endif // LAMM_IMPL_HPP