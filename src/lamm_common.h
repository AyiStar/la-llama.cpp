#ifndef LAMM_COMMON_H
#define LAMM_COMMON_H

#pragma GCC diagnostic ignored "-Wpedantic"

#include "lamm_ggml_type_trait.h"
#include "lamm_simd.h"
#include "loongarch_matmul.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

struct Matrix {
  void *data;
  ggml_type type;
  int row;
  int col;
  int64_t ld;
};

// opt level control
#if defined(LAMM_OPT_LEVEL)
constexpr int kOptLevel = LAMM_OPT_LEVEL;
#else
constexpr int kOptLevel = 3;
#endif

#endif // LAMM_COMMON_H