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

template <> struct ggml_type_trait<GGML_TYPE_Q4_0> {
  typedef block_q4_0 dtype;
  typedef block_q8_0 vec_dot_dtype;
};

template <> struct ggml_type_trait<GGML_TYPE_Q4_1> {
  typedef block_q4_1 dtype;
  typedef block_q8_1 vec_dot_dtype;
};

#endif // LAMM_GGML_TYPETRAIT