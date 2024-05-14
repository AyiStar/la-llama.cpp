#ifndef LOONGARCH_MATMUL_H
#define LOONGARCH_MATMUL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool lamm_can_mul_mat(
    const struct ggml_compute_params * params,
    const struct ggml_tensor* dst
);


void lamm_mul_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);



#ifdef __cplusplus
}
#endif

#endif // LOONGARCH_MATMUL_H