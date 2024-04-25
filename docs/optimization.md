# Optimization Procedure

## Know about llama.cpp
`llama.cpp` is an application-level program built on top of the `ggml` tensor library, mainly focusing on accelerating LLM inference on CPUs.
The two most important features to achieve acceleration in `llama.cpp` and `ggml` are **quantization** and **SIMD**.



## Check the Bottleneck

The profiling of the inference procedure shows that the "hottest call" is `ggml_vec_dot_*()` (~85\% in single thread case), which is called by `ggml_compute_forward_mul_mat()`.
`*` is `q4_0_q8_0`, `q4_1_q8_1`, etc., depending on the type of the running model. We use `Q4_0` as an instance.
As the names indicate, the functions perform vector dot-production and matrix multiplication, respectively.
