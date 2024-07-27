# LA-llama.cpp

Let's play LLM on LoongArch!


## Overview

The project aims at porting and optimizing llama.cpp, a C++ LLM inference framework, on LoongArch.
Especially, we want to tackle the following challenges:

* Potential problems when porting the code on LoongArch platform.
* Inference performance optimization via SIMD, temporarily targeting at 3A6000 platform.
* LLM evaluation on LoongArch platform.
* Interesting applications with presentation.

## Project Structure

The overall directory structure of this project is organized as follows:
- `llama.cpp-b2430/`: The original code of llama.cpp with fixed release version `b2430`. During development, we try to keep minimum change within this directory by only revising the build system (Makefile) and some conditionally compiled code (Macros to insert our work). Most of the real work are in the `src/` directory.
- `src/`: This is where we put the real optimization code, i.e., `loongarch_matmul.[cpp|h]`.
- `test/`: The benchmark code, which is altered from `llama.cpp-b2430/examples/benchmark/benchmark-matmult.cpp`. That means, the performance measure is completely comparable with the former reported results in community.
- `docs/`: The documentation generated along with the project.

## Plan

Based on the above challenges, the project can be divided into the following 4 steps:

### Setup
- Task: Build basic environments and get familiar to the codebase.
- Objective: Environment setup and self warm-up.

### Porting
- Task: Port llama.cpp to LoongArch platform.
- Objective: Compile and run llama.cpp on 3A6000.

### Optimization
- Task: Optimize the efficiency of llama.cpp on LoongArch (focus on CPU).
- Objective: Apply programming optimization techniques and document the improvements.

### Evaluation
- Task: Benchmark various LLMs of different sizes.
- Objective: Output a technical report.

### Application
- Task: Deploy usable applications with LLM on LoongArch platforms.
- Objective: Output well-written deployment documents and visual demos.

## Miscellaneous
- We develop based on release `b2430` of the [original repo](https://github.com/ggerganov/llama.cpp/releases/tag/b2430).

## Progress and TODO list

### Stage-1

#### Setup
At this step, we get familiar with the concept of cross compilation, build and 
- [x] Compile and run original llama.cpp on x86 CPU.
- [x] Cross compile llama.cpp to RISCV64 and run with QEMU on x86 CPU (refer to https://github.com/ggerganov/llama.cpp/pull/3453).
- [x] Set up cross compilation tools and QEMU environment for LoongArch.

#### Porting
- [x] Alter the makefile for LoongArch cross compilation.
- [x] Cross compile llama.cpp to LoongArch64.

#### Optimization
Thanks to [the excellent work from Loongson team](https://github.com/ggerganov/llama.cpp/pull/6454), we have a great oppotunity to learn about SIMD acceleration with LoongArch LSX/LASX vector instruction set. Part of our work are based on them.
- [x] Identify performance bottleneck in llama.cpp.
- [x] Add LSX/LASX SIMD acceleration for llama.cpp.
- [x] Add LASX GEMM acceleration for llama.cpp.

#### Benchmark
Benchmark goes along with optimization because we always want to know the exact improvement.
- [x] Measure performance improvement on Loongson 3A6000 processor.

#### Finish
Output a well-organized technical report.
- [x] Compete technical report.

### Stage-2

#### Code Refactoring
- [x] Decouple different quantization.
- [x] Abstract blocking logic.
- [x] Meta-programming the block kernel with macros.

#### Quantization Methods
- [x] Q2_K
- [x] Q4_0
- [x] Q4_1
- [ ] Q4_K
- [x] Q5_0
- [x] Q5_1
- [ ] Q5_K
- [ ] Q6_K
- [x] Q8_0

#### Benchmarks
- [x] Add arguments with different types.
- [x] Prepare more model weights.
- [x] Benchmark more models.

#### Autotune
TBD