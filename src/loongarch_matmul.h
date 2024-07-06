#ifndef LOONGARCH_MATMUL_H
#define LOONGARCH_MATMUL_H

#include <stdbool.h>

#ifdef _MSC_VER
#define LA_INLINE __forceinline
#define LA_NOINLINE __declspec(noinline)
#else
#define LA_INLINE inline __attribute__((always_inline))
#define LA_NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool lamm_can_mul_mat(const struct ggml_compute_params *params,
                      const struct ggml_tensor *dst);

void lamm_mul_mat(const struct ggml_compute_params *params,
                  struct ggml_tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // LOONGARCH_MATMUL_H
