#ifndef LAMM_COMMON_H
#define LAMM_COMMON_H

#pragma GCC diagnostic ignored "-Wpedantic"

#include "lamm_ggml_type_trait.h"
#include "lamm_simd.h"
#include "loongarch_matmul.h"

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

// useful macro for code repeat
#define LOOP_1(FN) FN(0)
#define LOOP_2(FN) LOOP_1(FN) FN(1)
#define LOOP_3(FN) LOOP_2(FN) FN(2)
#define LOOP_4(FN) LOOP_3(FN) FN(3)
#define LOOP_5(FN) LOOP_4(FN) FN(4)

#define LOOP(FN, N) LOOP_##N(FN)

#define LOOP_INNER_1(FN, M0) FN(M0, 0)
#define LOOP_INNER_2(FN, M0) LOOP_INNER_1(FN, M0) FN(M0, 1)
#define LOOP_INNER_3(FN, M0) LOOP_INNER_2(FN, M0) FN(M0, 2)
#define LOOP_INNER_4(FN, M0) LOOP_INNER_3(FN, M0) FN(M0, 3)
#define LOOP_INNER_5(FN, M0) LOOP_INNER_4(FN, M0) FN(M0, 4)

#define LOOP_INNER(FN, M0, N) LOOP_INNER_##N(FN, M0)

#define LOOP_1_1_ROW(FN) FN(0, 0)
#define LOOP_1_2_ROW(FN) LOOP_1_1_ROW(FN) FN(0, 1)
#define LOOP_1_3_ROW(FN) LOOP_1_2_ROW(FN) FN(0, 2)
#define LOOP_1_4_ROW(FN) LOOP_1_3_ROW(FN) FN(0, 3)
#define LOOP_1_5_ROW(FN) LOOP_1_4_ROW(FN) FN(0, 4)
#define LOOP_2_1_ROW(FN) FN(1, 0)
#define LOOP_2_2_ROW(FN) LOOP_2_1_ROW(FN) FN(1, 1)
#define LOOP_2_3_ROW(FN) LOOP_2_2_ROW(FN) FN(1, 2)
#define LOOP_2_4_ROW(FN) LOOP_2_3_ROW(FN) FN(1, 3)
#define LOOP_2_5_ROW(FN) LOOP_2_4_ROW(FN) FN(1, 4)
#define LOOP_3_1_ROW(FN) FN(2, 0)
#define LOOP_3_2_ROW(FN) LOOP_3_1_ROW(FN) FN(2, 1)
#define LOOP_3_3_ROW(FN) LOOP_3_2_ROW(FN) FN(2, 2)
#define LOOP_3_4_ROW(FN) LOOP_3_3_ROW(FN) FN(2, 3)
#define LOOP_3_5_ROW(FN) LOOP_3_4_ROW(FN) FN(2, 4)
#define LOOP_4_1_ROW(FN) FN(3, 0)
#define LOOP_4_2_ROW(FN) LOOP_4_1_ROW(FN) FN(3, 1)
#define LOOP_4_3_ROW(FN) LOOP_4_2_ROW(FN) FN(3, 2)
#define LOOP_4_4_ROW(FN) LOOP_4_3_ROW(FN) FN(3, 3)
#define LOOP_4_5_ROW(FN) LOOP_4_4_ROW(FN) FN(3, 4)
#define LOOP_5_1_ROW(FN) FN(4, 0)
#define LOOP_5_2_ROW(FN) LOOP_5_1_ROW(FN) FN(4, 1)
#define LOOP_5_3_ROW(FN) LOOP_5_2_ROW(FN) FN(4, 2)
#define LOOP_5_4_ROW(FN) LOOP_5_3_ROW(FN) FN(4, 3)
#define LOOP_5_5_ROW(FN) LOOP_5_4_ROW(FN) FN(4, 4)

#define LOOP_1_1(FN) LOOP_1_1_ROW(FN)
#define LOOP_1_2(FN) LOOP_1_2_ROW(FN)
#define LOOP_1_3(FN) LOOP_1_3_ROW(FN)
#define LOOP_1_4(FN) LOOP_1_4_ROW(FN)
#define LOOP_1_5(FN) LOOP_1_5_ROW(FN)
#define LOOP_2_1(FN) LOOP_1_1(FN) LOOP_2_1_ROW(FN)
#define LOOP_2_2(FN) LOOP_1_2(FN) LOOP_2_2_ROW(FN)
#define LOOP_2_3(FN) LOOP_1_3(FN) LOOP_2_3_ROW(FN)
#define LOOP_2_4(FN) LOOP_1_4(FN) LOOP_2_4_ROW(FN)
#define LOOP_2_5(FN) LOOP_1_5(FN) LOOP_2_5_ROW(FN)
#define LOOP_3_1(FN) LOOP_2_1(FN) LOOP_3_1_ROW(FN)
#define LOOP_3_2(FN) LOOP_2_2(FN) LOOP_3_2_ROW(FN)
#define LOOP_3_3(FN) LOOP_2_3(FN) LOOP_3_3_ROW(FN)
#define LOOP_3_4(FN) LOOP_2_4(FN) LOOP_3_4_ROW(FN)
#define LOOP_3_5(FN) LOOP_2_5(FN) LOOP_3_5_ROW(FN)
#define LOOP_4_1(FN) LOOP_3_1(FN) LOOP_4_1_ROW(FN)
#define LOOP_4_2(FN) LOOP_3_2(FN) LOOP_4_2_ROW(FN)
#define LOOP_4_3(FN) LOOP_3_3(FN) LOOP_4_3_ROW(FN)
#define LOOP_4_4(FN) LOOP_3_4(FN) LOOP_4_4_ROW(FN)
#define LOOP_4_5(FN) LOOP_3_5(FN) LOOP_4_5_ROW(FN)
#define LOOP_5_1(FN) LOOP_4_1(FN) LOOP_5_1_ROW(FN)
#define LOOP_5_2(FN) LOOP_4_2(FN) LOOP_5_2_ROW(FN)
#define LOOP_5_3(FN) LOOP_4_3(FN) LOOP_5_3_ROW(FN)
#define LOOP_5_4(FN) LOOP_4_4(FN) LOOP_5_4_ROW(FN)
#define LOOP_5_5(FN) LOOP_4_5(FN) LOOP_5_5_ROW(FN)

#define DOUBLE_LOOP(FN, M, N) LOOP_##M##_##N(FN)

// Basic data structures
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

// debug control
#if defined(LAMM_DEBUG)
constexpr bool kDebug = true;
#else
constexpr bool kDebug = false;
#endif

#endif // LAMM_COMMON_H