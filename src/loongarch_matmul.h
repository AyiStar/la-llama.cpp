#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"


#ifdef __cplusplus
extern "C" {
#endif

bool lamm_can_mul_mat(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    const struct ggml_tensor* dst
);


void lamm_mul_mat(
    struct ggml_tensor* src0,
    struct ggml_tensor* src1,
    struct ggml_tensor* dst,
    void * wdata,
    size_t wsize
);



#ifdef __cplusplus
}
#endif