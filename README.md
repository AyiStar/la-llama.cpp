# LA-llama.cpp

Let's play LLM on LoongArch!


## Overview

The project aims at porting and optimizing llama.cpp, a C++ LLM inference framework, on LoongArch.
Especially, we want to tackle the following challenges:

* Potential problems when porting the code on LoongArch platform.
* Inference performance optimization via SIMD, temporarily targeting at 3A6000 platform.
* LLM evaluation on LoongArch platform.
* Interesting applications with presentation.

## Plan

Based on the above challenges, the project can be divided into the following 4 stages:

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

### Setup Stage
At this stage, we get familiar with the concept of cross compilation, build and 
- [x] Compile and run original llama.cpp on x86 CPU.
- [x] Cross compile llama.cpp to RISCV64 and run with QEMU on x86 CPU (refer to https://github.com/ggerganov/llama.cpp/pull/3453).
- [x] Set up cross compilation tools and QEMU environment for LoongArch.

### Porting Stage
- [x] Alter the makefile for LoongArch cross compilation.
- [x] Cross compile llama.cpp to LoongArch64.