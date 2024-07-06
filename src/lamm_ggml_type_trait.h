#ifndef LAMM_GGML_TYPE_TRAIT_H
#define LAMM_GGML_TYPE_TRAIT_H

#include "ggml.h"

template <ggml_type Type> struct ggml_type_trait {};

template <> struct ggml_type_trait<GGML_TYPE_F32> {
  using dtype = float;
  using vec_dot_dtype = float;
};

template <> struct ggml_type_trait<GGML_TYPE_Q4_1> {
  using dtype = block_q4_1;
  using vec_dot_dtype = block_q8_1;
};

#endif // LAMM_GGML_TYPETRAIT