#include "lamm_common.h"
#include "lamm_impl.hpp"

#include <array>
#include <cassert>
#include <iostream>

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
      {GGML_TYPE_F32, GGML_TYPE_F32},   {GGML_TYPE_Q2_K, GGML_TYPE_Q8_K},
      {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}, {GGML_TYPE_Q4_1, GGML_TYPE_Q8_1},
      {GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}, {GGML_TYPE_Q5_1, GGML_TYPE_Q8_1},
      {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
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
    if (kDebug) {
      std::cout << "data type not supported" << std::endl;
      assert(false);
    }
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
  const size_t row_size = ggml_row_size(vec_dot_type, ne10);

  Matrix A, B, C;

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

  using MatMulFuncPtr = void(*)(const Matrix &A, const Matrix &B, const Matrix &C, int ith, int nth);
  MatMulFuncPtr mm_func = nullptr;
  switch (A.type) {
  case GGML_TYPE_F32:
    mm_func = LAMMImpl<GGML_TYPE_F32>::matmul;
    break;
  case GGML_TYPE_Q2_K:
    mm_func = LAMMImpl<GGML_TYPE_Q2_K>::matmul;
    break;
  case GGML_TYPE_Q4_0:
    mm_func = LAMMImpl<GGML_TYPE_Q4_0>::matmul;
    break;
  case GGML_TYPE_Q4_1:
    mm_func = LAMMImpl<GGML_TYPE_Q4_1>::matmul;
    break;
  case GGML_TYPE_Q5_0:
    mm_func = LAMMImpl<GGML_TYPE_Q5_0>::matmul;
    break;
  case GGML_TYPE_Q5_1:
    mm_func = LAMMImpl<GGML_TYPE_Q5_1>::matmul;
    break;
  case GGML_TYPE_Q8_0:
    mm_func = LAMMImpl<GGML_TYPE_Q8_0>::matmul;
    break;
  default:
    assert(false); // unreachable
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
      mm_func(A, B, C, params->ith, params->nth);
    }
  }
}

int lamm_get_opt_level() { return kOptLevel; }
