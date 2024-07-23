#ifndef LAMM_GGML_TYPE_TRAIT_H
#define LAMM_GGML_TYPE_TRAIT_H

#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml.h"

template <ggml_type Type> struct ggml_type_trait {};

template <> struct ggml_type_trait<GGML_TYPE_F32> {
  typedef float dtype;
  typedef float vec_dot_dtype;
};

static_assert (QK_K == 256);

template <> struct ggml_type_trait<GGML_TYPE_Q2_K> {
  typedef block_q2_K dtype;
  typedef block_q8_K vec_dot_dtype;
  static constexpr int super_block_size = QK_K;
};

template <> struct ggml_type_trait<GGML_TYPE_Q4_0> {
  typedef block_q4_0 dtype;
  typedef block_q8_0 vec_dot_dtype;
  static constexpr int super_block_size = QK4_0;
};

template <> struct ggml_type_trait<GGML_TYPE_Q4_1> {
  typedef block_q4_1 dtype;
  typedef block_q8_1 vec_dot_dtype;
  static constexpr int super_block_size = QK4_1;
};

template <> struct ggml_type_trait<GGML_TYPE_Q8_0> {
  typedef block_q8_0 dtype;
  typedef block_q8_0 vec_dot_dtype;
  static constexpr int super_block_size = QK8_0;
};

template <> struct ggml_type_trait<GGML_TYPE_Q8_1> {
  typedef block_q8_1 dtype;
  typedef block_q8_1 vec_dot_dtype;
  static constexpr int super_block_size = QK8_1;
};

template <> struct ggml_type_trait<GGML_TYPE_Q5_0> {
  typedef block_q5_0 dtype;
  typedef block_q8_0 vec_dot_dtype;
  static constexpr int super_block_size = QK5_0;
};

template <> struct ggml_type_trait<GGML_TYPE_Q5_1> {
  typedef block_q5_1 dtype;
  typedef block_q8_1 vec_dot_dtype;
  static constexpr int super_block_size = QK5_1;
};

#endif // LAMM_GGML_TYPETRAIT