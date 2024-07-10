#ifndef LAMM_SIMD_H
#define LAMM_SIMD_H

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
#include "lamm_simd_loongarch.h"
#elif defined(__AVX2__)
#include "lamm_simd_avx2.h"
#endif

#endif // LAMM_SIMD_H