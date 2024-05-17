#pragma GCC diagnostic ignored "-Wpedantic"

#include "loongarch_matmul.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <array>
#include <cassert>
#include <iostream>

#if defined(__loongarch_asx)
#include <lasxintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif




// abstraction for loongarch_asx SIMD intrinsics
namespace simd {

    enum class SIMD_ISA {
        AVX2,
        LASX,
        NONE
    };

    #if defined(__loongarch_asx)
    constexpr SIMD_ISA kISA = SIMD_ISA::LASX; 
    #elif defined(__AVX2__)
    constexpr SIMD_ISA kISA = SIMD_ISA::AVX2;
    #else
    constexpr SIMD_ISA kISA = SIMD_ISA::NONE;
    #endif

    static_assert(kISA != SIMD_ISA::NONE);
    
    constexpr int kNumVecReg = 32;
    constexpr int kVecWidth = 256;
    constexpr int kF32PerVec = kVecWidth / 32;

    using vreg_t = __m256;

#if defined(__loongarch_asx)
    // x + y: f32
    LA_INLINE vreg_t add(vreg_t x, vreg_t y) {
        return __lasx_xvfadd_s(x, y);
    }

    // x * y + z: f32
    LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z){
        return __lasx_xvfmadd_s(x, y, z);
    }

    // x - y: f32
    LA_INLINE vreg_t sub(vreg_t x, vreg_t y) {
        return __lasx_xvfsub_s(x, y);
    }

    // x * y: f32
    LA_INLINE vreg_t mul(vreg_t x, vreg_t y) {
        return __lasx_xvfmul_s(x, y);
    }

    // vector -> f32
    LA_INLINE float reduce_sum(vreg_t x) {
        return 0;  // TODO
    }

    // load from float*
    LA_INLINE vreg_t load(const float* p) {
        return __lasx_xvld(p, 0);
    }

#elif defined(__AVX2__)

    // x + y: f32
    LA_INLINE vreg_t add(vreg_t x, vreg_t y) {
        return _mm256_add_ps(x, y);
    }

    // x * y + z: f32
    LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z){
        return _mm256_fmadd_ps(x, y, z);
    }

    // x - y: f32
    LA_INLINE vreg_t sub(vreg_t x, vreg_t y) {
        return _mm256_sub_ps(x, y);
    }

    // x * y: f32
    LA_INLINE vreg_t mul(vreg_t x, vreg_t y) {
        return _mm256_mul_ps(x, y);
    }

    // vector -> f32
    LA_INLINE float reduce_sum(__m128 x) {
        x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        x = _mm_add_ss(x, _mm_movehdup_ps(x));
        return _mm_cvtss_f32(x);
    }
    
    LA_INLINE float reduce_sum(vreg_t x) {
        return reduce_sum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
            _mm256_castps256_ps128(x)));
    }

    // load from float*
    LA_INLINE vreg_t load(const float* p) {
        return _mm256_loadu_ps(p);
    }
#endif

} // namespace simd


namespace impl {

struct Matrix {
    void* data;
    ggml_type type;
    int row;
    int col;
    int64_t ld;
};

LA_NOINLINE void gemm(const Matrix& A, const Matrix& B, const Matrix& C, int ith, int nth);
LA_INLINE void gemm_naive(const Matrix& A, const Matrix& B, const Matrix& C, int ith, int nth);
LA_INLINE void gemm_simd(const Matrix& A, const Matrix& B, const Matrix& C, int ith, int nth);

// the real gemm function
void gemm(
    const Matrix& A,
    const Matrix& B,
    const Matrix& C,
    int ith,
    int nth
) {
    gemm_simd(A, B, C, ith, nth);
}

LA_INLINE void gemm_naive(
    const Matrix& A,
    const Matrix& B,
    const Matrix& C,
    int ith,
    int nth
) {
    assert(A.type == GGML_TYPE_F32 && B.type == A.type && C.type == A.type);
    float *a = (float*)(A.data), *b = (float*)(B.data), *c = (float*)(C.data);
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

    // naive implementation
    if (ith == 0) {
        std::cout << "naive implementation called" << std::endl;
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
    for (int i = job_start; i < job_end; i++) {
        for (int j = 0; j < N; j++) {
            c[j * ldc + i] = 0;
            for (int k = 0; k < K; k++) {
                c[j * ldc + i] += a[i * lda + k] * b[j * ldb + k];
            }
        }
    }
}

LA_INLINE void gemm_simd(
    const Matrix& A,
    const Matrix& B,
    const Matrix& C,
    int ith,
    int nth
) {
    assert(A.type == GGML_TYPE_F32 && B.type == A.type && C.type == A.type);
    float *a = (float*)(A.data), *b = (float*)(B.data), *c = (float*)(C.data);
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

    // simd implementation
    if (ith == 0) {
        std::cout << "SIMD implementation called" << std::endl;
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
}

} // namespace impl

// check if the gemm is suitable to be accelerated
// we assume that the basic assertions have been done
bool lamm_can_mul_mat(
    const struct ggml_compute_params * params,
    const struct ggml_tensor* dst
) {
	if (params->type != GGML_TASK_TYPE_COMPUTE) {
		return false;
	}

    auto src0 = dst->src[0];
    auto src1 = dst->src[1];

    // contiguous
    const bool src1_cont = ggml_is_contiguous(src1);
    enum ggml_type const vec_dot_type = ggml_internal_get_type_traits(src0->type).vec_dot_type;
    const bool src1_wdata = (src1->type != vec_dot_type);
    if (!src1_cont && !src1_wdata) {
        return false;
    }

    // what types do we support?
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    static const enum ggml_type supported_types[][2] = {
        {GGML_TYPE_F32, GGML_TYPE_F32},
        // {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0},
        // {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
    };
    const int num_supported_types = sizeof(supported_types) / sizeof(supported_types[0]);
    enum ggml_type type0 = src0->type, type1 = (src1_wdata ? vec_dot_type : src1->type);
    bool support = false;
    for (int i = 0; i < num_supported_types; i++) {
        if (type0 == supported_types[i][0] && type1 == supported_types[i][1]) {
            support = true;
            break;
        }
    }
    if (!support) {
        return false;
    }

    return true;
}


void lamm_mul_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    // enum ggml_type const vec_dot_type = ggml_internal_get_type_traits(src0->type).vec_dot_type;
    // const bool use_wdata = (src1->type != vec_dot_type);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    impl::Matrix A, B, C;

    A.type = src0->type;
    A.row = ne01;
    A.col = ne00/ggml_blck_size(src0->type);
    A.ld = nb01/ggml_type_size(src0->type);
    
    B.type = src1->type;
    B.row = ne00/ggml_blck_size(src0->type);
    B.col = ne11;
    B.ld = nb11/ggml_type_size(src1->type);

    C.type = dst->type;
    C.row = ne01;
    C.col = ne11;
    C.ld = nb1/ggml_type_size(dst->type);

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            A.data = (char *)src0->data + i12/r2*nb02 + i13/r3*nb03;
            B.data = (char *)src1->data + i12*nb12 + i13*nb13;
            C.data = (char *)dst->data + i12*nb2 + i13*nb3;
            impl::gemm(A, B, C, params->ith, params->nth);
        }
    }
}

