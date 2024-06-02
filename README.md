# Project 334 技术报告 - llama.cpp的龙芯平台移植与优化

> 项目成员：  
> 毕昊阳，中国科学技术大学  
> 胡冰玉，中国科学技术大学  
> 
> 指导教师：  
> 王皓，中国科学技术大学

## 摘要

* **项目目标**：将llama.cpp移植至龙芯处理器3A6000，并进行软硬件协同优化，加速模型的CPU推理速度，使得以Meta LLaMA为代表的流行的大语言模型能够以可接受的速度运行于龙芯平台；
* **完成情况**：本项目的规划和进展情况可见[dev.md](dev.md)。截至本阶段，较于未经优化的代码，在矩阵乘法benchmark上达到6x-35x的FLOPS加速比，在模型推理上达到3x-6x的token吞吐量加速比，并能以流畅的用户体验进行13B参数量的大语言模型推理；
* **主要创新**：定位和分析了大语言模型推理的主要性能瓶颈；针对龙芯平台进行了**SIMD**和**Cache**两个方向的计算优化；同时支持**浮点**参数和**量化**参数的运算加速；在3A6000处理器上进行了正确性和性能的标准测试。

本技术报告是对本项目的阶段性总结，也希望为后续工作及其他相关工作提供一些启发，具体包含以下章节：
1. 关于 llama.cpp 的背景介绍；
2. 针对龙芯平台的移植工作介绍；
3. 针对龙芯平台的软硬件协同优化工作介绍；
4. 项目的工程实现及成果展示；
5. 相关工作；
6. 未来工作与收获总结。


## 1. llama.cpp 背景介绍

### 1.1 什么是llama.cpp
llama.cpp是一个开源的大语言模型(Large Language Model, LLM)推理程序，支持包括Meta LLaMA等众多知名模型。所谓推理是指载入训练好的模型参数并运行模型，得到输出。LLM巨大的参数量，给推理过程的计算速度和内存占用都带来挑战。llama.cpp所解决的核心问题，就是在用户级设备，尤其是CPU上，进行高效的LLM推理。其解决该问题的主要手段包括：
1. 基于纯C/C++，无GC，面向底层，充分利用硬件特性，相比Python等语言天然具有性能优势；
2. 引入模型量化技术，显著减小内存占用，也间接提升运算性能。

针对以上两方面，我们分别进行简要介绍。

### 1.2 GGML
整个llama.cpp项目可以分为两部分：底层的张量库GGML(C语言)，和应用层的模型推理代码(C++语言)。严格来说，GGML是一个[独立的项目](https://github.com/ggerganov/ggml)，但在实际开发中，GGML被完整包含在llama.cpp项目中(工程目录下的ggml*文件)一起开发，并反馈合并给上游的原仓库。  
GGML是一个纯C语言的张量库，类似PyTorch，包含以下功能：

1. 张量构建：多维数组及相关基本操作（如拷贝、赋值等），相当于PyTorch中的tensor；
2. 算子实现：加法、矩阵乘法等张量算子，包含在各平台的优化；
3. 计算图调度：由张量和算子构成计算图，输入数据后执行真正的计算，包含多线程调度等。

和许多C语言库一样，GGML通过高效的内存管理，以及SIMD等面向平台特性的优化，来支持上层的高效推理。  
llama.cpp中的模型推理代码的本质，是利用GGML构建不同模型对应的计算图，载入模型参数，最后运行计算图。  
本项目对llama.cpp的性能优化，实际发生在GGML相关的代码中。

### 1.3 模型量化
除了C/C++带来的性能提升外，llama.cpp高度依赖于一种称为模型量化(model quantization)的推理优化技术，即通过更低精度的数据类型来存储模型参数，以牺牲参数精度为代价，显著降低参数的内存占用。例如，一个4字节的单精度浮点参数，可以存储为一个2字节半精度浮点，或一个1字节甚至4比特的整数。

llama.cpp支持多种量化方法，在项目中命名为Q2_0, Q4_0, Q4_1等。以Q4_1为例，Q代表Quantization，4代表每个参数量化为4比特整数，1代表量化方法的"版本号"。每一种量化方法有不同的数据存储方式，背后对应着推理性能和模型效果的权衡，一般来说，量化的"压缩率"越高，推理性能越高，但因精度损失，模型效果越差。

本项目在优化过程中同时考虑了普通的单精度浮点运算和量化运算。对于量化，由于精力有限，我们只考虑Q4_1这一种量化方法，该方法在社区内被广泛认为是一种性能-效果较为折衷的量化方法。具体来说，Q4_1将每32个参数打包成一个block，存储结构如下：

```C
// ggml-common.h
#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
```

 要想从`blockq4_1` 中还原一个参数，只需 `blk.m + blk.qs[i] * blk.d` 。在性能优化中，我们将充分利用该结构的性质，比如每个block中qs的32个uint8整数连续存储等等。




## 2. 针对龙芯平台的移植工作

针对龙芯平台的移植工作分为两个部分：平台无关移植与平台相关移植。下面分别进行介绍。

### 2.1 平台无关移植
llama.cpp基于C/C++，开发过程保持了良好的跨平台规范，所有针对特定平台（如x86、ARM、RISC-V等）的代码均由条件编译（基于宏）和构建系统（基于Make、CMake等工具）控制处理。举例来说，项目中随处可见以下形式的代码：
```C
#if defined(__AVX__)
... // 针对x86平台的优化
#elif defined(__ARM_NEON)
... // 针对ARM平台的优化
#else
... //平台无关的代码
```
在构建系统中，也须加入对应的编译器选项。例如原项目的Makefile中有：
```Makefile
# llama.cpp-b2430/Makefile
ifdef RISCV
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
else
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
...
```

另一方面，龙芯平台上有完整的GNU工具链。因此，直接在LoongArch架构上编译llama.cpp项目是无痛的。针对上面的代码片段，在3A6000处理器则默认会编译出 `#else` 部分的代码。

### 2.2 平台相关移植
llama.cpp中平台无关的代码对应的性能是未经优化且无法接受的。因此，开发过程不可避免地须在项目代码中加入LoongArch的平台相关代码。在本项目中，所涉及的平台相关特性为LASX扩展向量指令集（注：由于主要针对3A6000处理器，该处理器支持更高效的LASX指令，所以我们暂未考虑LSX）。

因此，我们仿照项目中相应的做法，在代码中插入条件编译以保持原有的跨平台特性：
```C
...
#elif defined(__loongarch_lasx__)
... // 针对龙芯平台的优化
...
```
对应地，在Makefile中，我们插入：
```Makefile
# llama.cpp-b2430/Makefile
ifneq ($(filter loongarch64%,$(UNAME_M)),)
	MK_CFLAGS   += -mlasx
	MK_CXXFLAGS += -mlasx
endif
```

至此，针对龙芯平台的移植工作完成。



## 3. 针对龙芯平台的软硬件协同优化
针对龙芯平台的大模型推理速度优化是本项目的重点工作，相对移植来说，占据了主要的工作量。我们的优化流程总体如下：
1. 通过profile工具，定位性能瓶颈为GEMM操作；
2. 针对GEMM优化进行调研，阅读和理解llama.cpp中相应的代码，确定从SIMD和Cache两个方向进行优化；
3. 借助龙芯的LASX向量指令集进行SIMD优化（这一步恰巧与龙芯团队的工作重合）；
4. 在SIMD优化基础上，进一步针对Cache进行优化。

下面进行详细介绍。

### 3.1 定位性能瓶颈
在有限的时间内，我们无法对llama.cpp这样的大型项目进行全面的优化。而想要以最高的效率获得性能提升，应先定位程序真正的性能瓶颈或热点代码。我们通过Linux平台的perf工具对llama.cpp的模型推理进行profile，发现90%以上的CPU计算用于位于`ggml.c` 的 `ggml_compute_forward_mul_mat()` 函数。该函数的作用是对两个张量的前两维进行矩阵乘法运算，也即所谓的GEMM。究其原因，是因为当前大语言模型均基于Transformer架构，而Transformer中绝大部分计算为Attention，后者本质就是在做矩阵乘法。总之，这对本项目来而言是利好的，我们后续只需针对 `ggml_compute_forward_mul_mat()` 函数进行性能优化即可。


### 3.2 确定优化方案
GEMM是高性能计算中的经典课题。本项目团队并非相关专业出身，谨根据有限的调研结果，确定从两个方向进行GEMM优化。

需要指出的是，这并不意味着本项目简单地规约成了一个GEMM优化，因为：
1. llama.cpp重度依赖模型量化技术，量化后的矩阵并非存储为连续的浮点数，GEMM优化必须考虑量化后参数的存储结构；
2. 需要理解项目中张量的存储逻辑、算子的实现逻辑和计算图的构造逻辑。

### 3.3 SIMD优化

Single Instruction Multiple Data (SIMD) 是现代处理器的重要并行手段，一般通过向量扩展指令集实现，如x86 SSE/AVX指令集，和本项目涉及的LASX。LASX包含32个256位向量寄存器，每个寄存器可以并行处理8个单精度浮点数（用于优化浮点矩阵乘法）或32个8位整数（用之优化量化矩阵乘法）。

为清楚说明优化过程，考虑两个相乘的矩阵$A_{M,K} \times B_{K,N}\rightarrow C_{M,N}$。在大模型中，$K$通常为模型隐向量长度，一般在512到4096之间。在llama.cpp中，矩阵可以写成如下伪代码：

```
for i = 0 to M-1:
  for j = 0 to N-1:
    C[i, j] = vector_dot_product(A[i, :], B[:, j]) // A的第i行点乘B的第j列
```

点乘操作是对两个向量中两两对应元素的乘积的累加，符合典型的SIMD模式。以256位向量寄存器加速F32浮点运算为例，我们可对`256/32=8` 个浮点数进行并行计算。

#### 3.3.1 SIMD操作的工程抽象

LASX上的向量指令繁多，但本项目中真正用到的操作极其有限，并且需要一些基本操作的组合，因此我们对其进行了必要的工程抽象，核心代码如下：

```C++
// src/loongarch_matmul.cpp
namespace simd {

constexpr int kNumVecReg = 32;
constexpr int kVecWidth = 256;
constexpr int kF32PerVec = kVecWidth / 32;

using vreg_t = __m256;  // vector register type
using ivreg_t = __m256i;  // integer vector register type

#if defined(__loongarch_asx)
...

LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return __lasx_xvfadd_s(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return __lasx_xvfmadd_s(x, y, z);
}

// x - y: f32
LA_INLINE vreg_t sub(vreg_t x, vreg_t y) { return __lasx_xvfsub_s(x, y); }

// x * y: f32
LA_INLINE vreg_t mul(vreg_t x, vreg_t y) { return __lasx_xvfmul_s(x, y); }

// Convert __m256i high part to __m128i
LA_INLINE __m128i lasx_extracti128_hi(__m256i in)
{
    __m128i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}

LA_INLINE __m128 lasx_extractf128( __m256 a, int pos)
{
    __m128 ret;
    if( pos == 0)
    {
       ret = (__m128)lasx_extracti128_lo((__m256i)a);
    } else {
       ret = (__m128)lasx_extracti128_hi((__m256i)a);
    }
    return ret;
}

// vector -> f32
LA_INLINE float reduce_sum(vreg_t x) {
  __m128 res = lasx_extractf128(x, 1);
  FloatInt tmp;
  res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
  tmp.i = __lsx_vpickve2gr_w(res, 0);
  return tmp.f;
}

// load from float*
LA_INLINE vreg_t load(const float *p) { return (vreg_t)__lasx_xvld(p, 0); }

// load from quantized block
LA_INLINE ivreg_t load_quants(const block_q4_1 *p) {
  const __m128i lo = __lsx_vld((const __m128i *)(p->qs), 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_set_q(hi, lo), 0xf);
}
LA_INLINE ivreg_t load_quants(const block_q8_1 *p) {
  return __lasx_xvld( (const __m256i *)(p->qs), 0);
}

LA_INLINE vreg_t sum_i16_pairs_float(const ivreg_t x) {
    ivreg_t v = __lasx_xvpackod_h(x, x);
    ivreg_t summed_pairs = __lasx_xvaddwev_w_h(x, v);
    return __lasx_xvffint_s_w(summed_pairs);
}

LA_INLINE ivreg_t lasx_maddubs_h(ivreg_t a, ivreg_t b)
{
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    return __lasx_xvsadd_h(tmp1, tmp2);
}

LA_INLINE vreg_t mul_sum_us8_pairs_float(const ivreg_t ax, const ivreg_t sy) {
    // Perform multiplication and create 16-bit values
    const ivreg_t dot = lasx_maddubs_h(ax, sy);
    return sum_i16_pairs_float(dot);
}

...
}  // namespace simd
```

其中部分代码借鉴了龙芯团队的[相关工作](https://github.com/ggerganov/llama.cpp/pull/6454)。巧合的是，该工作出现在团队成员正在学习LASX指令集的过程中。事实证明，龙芯团队对于LASX的运用比我们要精到得多，我们学到不少技巧的同时，也省去了大量的工作量，为后续进行更深入的优化提供可能性。在此也十分感谢张福新老师及时将相关工作进展同步于我们。

在实现中，我们还针对AVX2实现了同样的接口，因为其具有和LASX一样的256位向量寄存器，方便在其他平台同步开发测试。

#### 3.3.2 F32浮点计算的SIMD优化

利用上述工程抽象，我们针对浮点的GEMM进行了SIMD优化，核心代码如下：

```C++
// src/loongarch_matmul.cpp
for (int i = job_start; i < job_end; i++) {
  for (int j = 0; j < N; j++) {
    simd::vreg_t vc = {0}, va = {0}, vb = {0};
    for (int k = 0; k < K; k += simd::kF32PerVec) {
      va = simd::load(a + i * lda + k);
      vb = simd::load(b + j * ldb + k);
      vc = simd::madd(va, vb, vc);
    }
    c[j * ldc + i] = simd::reduce_sum(vc);
  }
}
```

#### 3.3.3  Q4_1量化计算的SIMD优化

类似地，我们针对Q4_1的GEMM进行了SIMD优化，核心代码如下：

```C++
// src/loongarch_matmul.cpp
for (int i = job_start; i < job_end; i++) {
  for (int j = 0; j < N; j++) {
    float summs = 0;
    simd::vreg_t acc = {0};
    const auto *ai = a + (i * lda);
    const auto *bj = b + (j * ldb);
    for (int k = 0; k < K; k++, ai++, bj++) {
      summs += GGML_FP16_TO_FP32(ai->m) * GGML_FP16_TO_FP32(bj->s);
      const simd::vreg_t ad = simd::vset(GGML_FP16_TO_FP32(ai->d));
      const simd::vreg_t bd = simd::vset(GGML_FP16_TO_FP32(bj->d));
      const __m256 adbd = simd::mul( ad, bd );
      simd::ivreg_t va_qs = simd::load_quants(ai);
      simd::ivreg_t vb_qs = simd::load_quants(bj);
      const simd::vreg_t xy = simd::mul_sum_us8_pairs_float(va_qs, vb_qs);
      acc = simd::madd(adbd, xy, acc);
    }
    c[j * ldc + i] = simd::reduce_sum(acc) + summs;
  }
}
```

### 3.4 Cache优化

SIMD仅仅利用了处理器中的向量计算单元，而影响GEMM性能的另一大因素则是访存。根据手册，龙芯3A6000处理器拥有每核64KB L1缓存和256KB L2缓存。合理利用缓存，即在相同计算量下，提升缓存命中，降低内存访问次数，可以显著提升性能。在llama.cpp原有代码以及前述SIMD优化代码中，矩阵乘法的计算没有充分利用缓存。

注意，llama.cpp中已经对矩阵A的内存分布做了优化，此处矩阵A实质上已经做了转置。进一步地，考虑这个过程中的缓存行为。根据处理器缓存大小可以估算，缓存大于单个隐向量而小于整个矩阵。考虑最外层$i$变量循环，当$i=0$时，内层循环使得$A$的第0行与$B$的每一列一一点乘，这个过程中，$A$的行向量除第一次点乘外一直在缓存中，而$B$的列向量则在遍历过程中不断装入缓存，最终因缓存无法容纳所有列向量，而使得前面的列向量被替换。如此，当$i=1$的时候，$B$的所有列向量将须重新一一装入缓存。也就是说，真正被有效利用的缓存只有$A$的行向量所对应的部分，远小于处理器全部缓存大小。

因此，我们考虑通过分块处理来提高缓存命中率，一次性计算出$C$矩阵中的一个$B0\times B1$块，其本质是循环展开。为讨论简单起见，假设$M$和$N$分别整除$B0$和$B1$，伪代码如下：
```
for i = 0 to M with step B0：
  for j = 0 to N with step B1:
    for ib = 0 to B0-1:
      for jb = 0 to B1-1:
        C[i+ib,j+jb] = vector_dot_product(A[i+ib, :], B[:, j+jb])
```
当最内的两层循环涉及到的行/列向量可以合理容纳进缓存时，缓存的利用率可以大大提升。另一方面，SIMD和Cache的优化本身是正交关系，应结合起来达到更好的优化效果。我们注意到，通过循环展开时合理的计算排布，不仅可以使数据尽可能留在缓存内，也能够使数据尽可能留在向量寄存器内。而分块大小$B0\times B1$的设计，则同时与缓存大小和向量寄存器的数量相关。
在工程实现中，为尝试不同大小的分块带来的优化效果，同时方便处理$M,N$不整除$B0,B1$时的剩余情况，我们采用基于C++函数模板和`if constexpr`特性给予静态化的参数实现。

#### 3.3.1 F32浮点计算的Cache优化

针对单精度（F32）浮点计算的核心优化代码如下：
```C++
template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const float *a, const float *b, float *c, int64_t lda,
                                 int64_t ldb, int64_t ldc, int i, int j,
                                 int k) {

  static_assert(B0 > 0 && B0 <= 5);
  static_assert(B1 > 0 && B1 <= 5);
  
  using namespace simd;

  [[maybe_unused]] vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0}, vc04 = {0};
  [[maybe_unused]] vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0}, vc14 = {0};
  [[maybe_unused]] vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0}, vc24 = {0};
  [[maybe_unused]] vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0}, vc34 = {0};
  [[maybe_unused]] vreg_t vc40 = {0}, vc41 = {0}, vc42 = {0}, vc43 = {0}, vc44 = {0};
  [[maybe_unused]] vreg_t vb0 = {0}, vb1 = {0}, vb2 = {0}, vb3 = {0}, vb4 = {0};
  vreg_t va = {0};

  for (int l = 0; l < k; l += kF32PerVec) {

    if constexpr (B1 > 0) {vb0 = load(b + ldb * (j + 0) + l);}
    if constexpr (B1 > 1) {vb1 = load(b + ldb * (j + 1) + l);}
    if constexpr (B1 > 2) {vb2 = load(b + ldb * (j + 2) + l);}
    if constexpr (B1 > 3) {vb3 = load(b + ldb * (j + 3) + l);}
    if constexpr (B1 > 4) {vb4 = load(b + ldb * (j + 4) + l);}

    if constexpr (B0 > 0) {
      va = load(a + lda * (i + 0) + l);
      if constexpr (B1 > 0) {vc00 = madd(va, vb0, vc00);}
      if constexpr (B1 > 1) {vc01 = madd(va, vb1, vc01);}
      if constexpr (B1 > 2) {vc02 = madd(va, vb2, vc02);}
      if constexpr (B1 > 3) {vc03 = madd(va, vb3, vc03);}
      if constexpr (B1 > 4) {vc04 = madd(va, vb4, vc04);}
    }

    if constexpr (B0 > 1) {va = load(a + lda * (i + 1) + l);
      if constexpr (B1 > 0) {vc10 = madd(va, vb0, vc10);}
      if constexpr (B1 > 1) {vc11 = madd(va, vb1, vc11);}
      if constexpr (B1 > 2) {vc12 = madd(va, vb2, vc12);}
      if constexpr (B1 > 3) {vc13 = madd(va, vb3, vc13);}
      if constexpr (B1 > 4) {vc14 = madd(va, vb4, vc14);}
    }

    if constexpr (B0 > 2) {
      va = load(a + lda * (i + 2) + l);
      if constexpr (B1 > 0) {vc20 = madd(va, vb0, vc20);}
      if constexpr (B1 > 1) {vc21 = madd(va, vb1, vc21);}
      if constexpr (B1 > 2) {vc22 = madd(va, vb2, vc22);}
      if constexpr (B1 > 3) {vc23 = madd(va, vb3, vc23);}
      if constexpr (B1 > 4) {vc24 = madd(va, vb4, vc24);}
    }

    if constexpr (B0 > 3) {
      va = load(a + lda * (i + 3) + l);
      if constexpr (B1 > 0) {vc30 = madd(va, vb0, vc30);}
      if constexpr (B1 > 1) {vc31 = madd(va, vb1, vc31);}
      if constexpr (B1 > 2) {vc32 = madd(va, vb2, vc32);}
      if constexpr (B1 > 3) {vc33 = madd(va, vb3, vc33);}
      if constexpr (B1 > 4) {vc34 = madd(va, vb4, vc34);}
    }

    if constexpr (B0 > 4) {
      va = load(a + lda * (i + 4) + l);
      if constexpr (B1 > 0) {vc40 = madd(va, vb0, vc40);}
      if constexpr (B1 > 1) {vc41 = madd(va, vb1, vc41);}
      if constexpr (B1 > 2) {vc42 = madd(va, vb2, vc42);}
      if constexpr (B1 > 3) {vc43 = madd(va, vb3, vc43);}
      if constexpr (B1 > 4) {vc44 = madd(va, vb4, vc44);}
    }
  }

  if constexpr (B1 > 0) {
    if constexpr (B0 > 0) {c[ldc * (j + 0) + (i + 0)] = reduce_sum(vc00);}
    if constexpr (B0 > 1) {c[ldc * (j + 0) + (i + 1)] = reduce_sum(vc10);}
    if constexpr (B0 > 2) {c[ldc * (j + 0) + (i + 2)] = reduce_sum(vc20);}
    if constexpr (B0 > 3) {c[ldc * (j + 0) + (i + 3)] = reduce_sum(vc30);}
    if constexpr (B0 > 4) {c[ldc * (j + 0) + (i + 4)] = reduce_sum(vc40);}
  }

  if constexpr (B1 > 1) {
    if constexpr (B0 > 0) {c[ldc * (j + 1) + (i + 0)] = reduce_sum(vc01);}
    if constexpr (B0 > 1) {c[ldc * (j + 1) + (i + 1)] = reduce_sum(vc11);}
    if constexpr (B0 > 2) {c[ldc * (j + 1) + (i + 2)] = reduce_sum(vc21);}
    if constexpr (B0 > 3) {c[ldc * (j + 1) + (i + 3)] = reduce_sum(vc31);}
    if constexpr (B0 > 4) {c[ldc * (j + 1) + (i + 4)] = reduce_sum(vc41);}
  }

  if constexpr (B1 > 2) {
    if constexpr (B0 > 0) {c[ldc * (j + 2) + (i + 0)] = reduce_sum(vc02);}
    if constexpr (B0 > 1) {c[ldc * (j + 2) + (i + 1)] = reduce_sum(vc12);}
    if constexpr (B0 > 2) {c[ldc * (j + 2) + (i + 2)] = reduce_sum(vc22);}
    if constexpr (B0 > 3) {c[ldc * (j + 2) + (i + 3)] = reduce_sum(vc32);}
    if constexpr (B0 > 4) {c[ldc * (j + 2) + (i + 4)] = reduce_sum(vc42);}
  }

  if constexpr (B1 > 3) {
    if constexpr (B0 > 0) {c[ldc * (j + 3) + (i + 0)] = reduce_sum(vc03);}
    if constexpr (B0 > 1) {c[ldc * (j + 3) + (i + 1)] = reduce_sum(vc13);}
    if constexpr (B0 > 2) {c[ldc * (j + 3) + (i + 2)] = reduce_sum(vc23);}
    if constexpr (B0 > 3) {c[ldc * (j + 3) + (i + 3)] = reduce_sum(vc33);}
    if constexpr (B0 > 4) {c[ldc * (j + 3) + (i + 4)] = reduce_sum(vc43);}
  }
  
  if constexpr (B1 > 4) {
    if constexpr (B0 > 0) {c[ldc * (j + 4) + (i + 0)] = reduce_sum(vc04);}
    if constexpr (B0 > 1) {c[ldc * (j + 4) + (i + 1)] = reduce_sum(vc14);}
    if constexpr (B0 > 2) {c[ldc * (j + 4) + (i + 2)] = reduce_sum(vc24);}
    if constexpr (B0 > 3) {c[ldc * (j + 4) + (i + 3)] = reduce_sum(vc34);}
    if constexpr (B0 > 4) {c[ldc * (j + 4) + (i + 4)] = reduce_sum(vc44);}
  }
}
```
以上代码可以实现`5x5`以内的分块，因为我们认为更大的分块会需要过多的向量寄存器。


#### 3.3.2 Q4_1量化计算的Cache优化

针对Q4_1量化计算的主要逻辑与F32优化类似，须额外考虑量化的存储和计算逻辑，具体如下：
1. 对于`Q4_1`量化类型的两个相乘矩阵$A,B$，llama.cpp（严格来说是GGML）在计算图的预处理阶段阶段会将矩阵$B$转化成`Q8_1`量化类型；
2. 对于任何量化计算，算子的输出结果仍是F32类型。

因此，本质上我们是在做 $Q4\_1\times Q8\_1\rightarrow F32$的计算优化，核心优化代码如下：

```C++
template <int B0, int B1>
LA_INLINE void gemm_block_kernel(const block_q4_1 *a, const block_q8_1 *b, float *c, int64_t lda,
                                 int64_t ldb, int64_t ldc, int i, int j,
                                 int K) {

  static_assert(B0 > 0 && B0 <= 4);
  static_assert(B1 > 0 && B1 <= 4);

  using namespace simd;

  ivreg_t va_qs = {0};
  simd::vreg_t vad = {0};
  [[maybe_unused]] ivreg_t vb0_qs = {0}, vb1_qs = {0}, vb2_qs = {0}, vb3_qs = {0};
  [[maybe_unused]] simd::vreg_t vbd0, vbd1, vbd2, vbd3;
  [[maybe_unused]] simd::vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0};
  [[maybe_unused]] simd::vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0};
  [[maybe_unused]] simd::vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0};
  [[maybe_unused]] simd::vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0};

  float summs[B0][B1] = {0};
  [[maybe_unused]] const auto *ai0{a + ((i + 0) * lda)};
  [[maybe_unused]] const auto *ai1{a + ((i + 1) * lda)};
  [[maybe_unused]] const auto *ai2{a + ((i + 2) * lda)};
  [[maybe_unused]] const auto *ai3{a + ((i + 3) * lda)};
  [[maybe_unused]] const auto *bj0{b + ((j + 0) * ldb)};
  [[maybe_unused]] const auto *bj1{b + ((j + 1) * ldb)};
  [[maybe_unused]] const auto *bj2{b + ((j + 2) * ldb)};
  [[maybe_unused]] const auto *bj3{b + ((j + 3) * ldb)};
  

  for (int k = 0; k < K; k++) {

    if constexpr (B1 > 0) {
      vb0_qs = load_quants(bj0 + k);
      vbd0 = vset(GGML_FP16_TO_FP32(bj0[k].d));
    }
    if constexpr (B1 > 1) {
      vb1_qs = load_quants(bj1 + k);
      vbd1 = vset(GGML_FP16_TO_FP32(bj1[k].d));
    }
    if constexpr (B1 > 2) {
      vb2_qs = load_quants(bj2 + k);
      vbd2 = vset(GGML_FP16_TO_FP32(bj2[k].d));
    }
    if constexpr (B1 > 3) {
      vb3_qs = load_quants(bj3 + k);
      vbd3 = vset(GGML_FP16_TO_FP32(bj3[k].d));
    }

    if constexpr (B0 > 0) {
      va_qs = load_quants(ai0 + k);
      vad = vset(GGML_FP16_TO_FP32(ai0[k].d));
      if constexpr (B1 > 0) {
        summs[0][0] += GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc00 = madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs) , vc00);
      }
      if constexpr (B1 > 1) {
        summs[0][1] += GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc01 = madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs) , vc01);
      }
      if constexpr (B1 > 2) {
        summs[0][2] += GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc02 = madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs) , vc02);
      }
      if constexpr (B1 > 3) {
        summs[0][3] += GGML_FP16_TO_FP32(ai0[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc03 = madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs) , vc03);
      }
    }

    if constexpr (B0 > 1) {
      va_qs = load_quants(ai1 + k);
      vad = vset(GGML_FP16_TO_FP32(ai1[k].d));
      if constexpr (B1 > 0) {
        summs[1][0] += GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc10 = madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs) , vc10);
      }
      if constexpr (B1 > 1) {
        summs[1][1] += GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc11 = madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs) , vc11);
      }
      if constexpr (B1 > 2) {
        summs[1][2] += GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc12 = madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs) , vc12);
      }
      if constexpr (B1 > 3) {
        summs[1][3] += GGML_FP16_TO_FP32(ai1[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc13 = madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs) , vc13);
      }
    }

    if constexpr (B0 > 2) {
      va_qs = load_quants(ai2 + k);
      vad = vset(GGML_FP16_TO_FP32(ai2[k].d));
      if constexpr (B1 > 0) {
        summs[2][0] += GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc20 = madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs) , vc20);
      }
      if constexpr (B1 > 1) {
        summs[2][1] += GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc21 = madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs) , vc21);
      }
      if constexpr (B1 > 2) {
        summs[2][2] += GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc22 = madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs) , vc22);
      }
      if constexpr (B1 > 3) {
        summs[2][3] += GGML_FP16_TO_FP32(ai2[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc23 = madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs) , vc23);
      }
    }

    if constexpr (B0 > 3) {
      va_qs = load_quants(ai3 + k);
      vad = vset(GGML_FP16_TO_FP32(ai3[k].d));
      if constexpr (B1 > 0) {
        summs[3][0] += GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj0[k].s);
        vc30 = madd(mul(vad, vbd0), mul_sum_us8_pairs_float(va_qs, vb0_qs) , vc30);
      }
      if constexpr (B1 > 1) {
        summs[3][1] += GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj1[k].s);
        vc31 = madd(mul(vad, vbd1), mul_sum_us8_pairs_float(va_qs, vb1_qs) , vc31);
      }
      if constexpr (B1 > 2) {
        summs[3][2] += GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj2[k].s);
        vc32 = madd(mul(vad, vbd2), mul_sum_us8_pairs_float(va_qs, vb2_qs) , vc32);
      }
      if constexpr (B1 > 3) {
        summs[3][3] += GGML_FP16_TO_FP32(ai3[k].m) * GGML_FP16_TO_FP32(bj3[k].s);
        vc33 = madd(mul(vad, vbd3), mul_sum_us8_pairs_float(va_qs, vb3_qs) , vc33);
      }
    }
    
  }

  if constexpr (B0 > 0) {
    if constexpr (B1 > 0) { c[ldc * (j + 0) + (i + 0)] = reduce_sum(vc00) + summs[0][0]; }
    if constexpr (B1 > 1) { c[ldc * (j + 1) + (i + 0)] = reduce_sum(vc01) + summs[0][1]; }
    if constexpr (B1 > 2) { c[ldc * (j + 2) + (i + 0)] = reduce_sum(vc02) + summs[0][2]; }
    if constexpr (B1 > 3) { c[ldc * (j + 3) + (i + 0)] = reduce_sum(vc03) + summs[0][3]; }
  }
  if constexpr (B0 > 1) {
    if constexpr (B1 > 0) { c[ldc * (j + 0) + (i + 1)] = reduce_sum(vc10) + summs[1][0]; }
    if constexpr (B1 > 1) { c[ldc * (j + 1) + (i + 1)] = reduce_sum(vc11) + summs[1][1]; }
    if constexpr (B1 > 2) { c[ldc * (j + 2) + (i + 1)] = reduce_sum(vc12) + summs[1][2]; }
    if constexpr (B1 > 3) { c[ldc * (j + 3) + (i + 1)] = reduce_sum(vc13) + summs[1][3]; }
  }
  if constexpr (B0 > 2) {
    if constexpr (B1 > 0) { c[ldc * (j + 0) + (i + 2)] = reduce_sum(vc20) + summs[2][0]; }
    if constexpr (B1 > 1) { c[ldc * (j + 1) + (i + 2)] = reduce_sum(vc21) + summs[2][1]; }
    if constexpr (B1 > 2) { c[ldc * (j + 2) + (i + 2)] = reduce_sum(vc22) + summs[2][2]; }
    if constexpr (B1 > 3) { c[ldc * (j + 3) + (i + 2)] = reduce_sum(vc23) + summs[2][3]; }
  }
  if constexpr (B0 > 3) {
    if constexpr (B1 > 0) { c[ldc * (j + 0) + (i + 3)] = reduce_sum(vc30) + summs[3][0]; }
    if constexpr (B1 > 1) { c[ldc * (j + 1) + (i + 3)] = reduce_sum(vc31) + summs[3][1]; }
    if constexpr (B1 > 2) { c[ldc * (j + 2) + (i + 3)] = reduce_sum(vc32) + summs[3][2]; }
    if constexpr (B1 > 3) { c[ldc * (j + 3) + (i + 3)] = reduce_sum(vc33) + summs[3][3]; }
  }
}
```
我们实现了最多`4x4`的分块优化，因为相比F32，量化计算的向量寄存器需求更高，需要更多的中间操作进行反量化（dequantize），即从量化后的整数恢复成浮点数。


## 4. 工程实现及成果展示

### 4.1 工程目录结构
本项目的总体目录结构如下所示：
- `llama.cpp-b2430/`：Tag为 `b2430` 的 llama.cpp 的原始代码，是本项目开始时的最新release版本。
- `src/`：这是我们放置自己的优化代码的地方，即 `loongarch_matmul.[cpp|h]`。
- `test/`：Benchmark测试代码，修改自 `llama.cpp-b2430/examples/benchmark/benchmark-matmult.cpp`。这意味着性能测量与社区先前报告的结果完全可比。
- `model_weights/`：存放模型参数文件。由于参数文件过大，我们没有直接将文件上传至代码仓库，而是在此目录下提供了下载文件的Python脚本。

### 4.2 工程实现概览
在开发过程中，我们尽量保持plug-in的原则，在原项目目录（`llama.cpp-b2430/`）内只对构建系统（Makefile）和一些包含条件编译的代码（用于插入我们的工作）进行必要的更改，大部分真正的开发工作都在 `src/` 目录中进行，其中声明的两个函数 `lamm_can_mul_mat()` 和 `lamm_mul_mat()` 被插入至 `llama.cpp-b2430/ggml.c` 中的GEMM执行调度函数 `ggml_compute_forward_mul_mat()` 来达到优化的目的。  
此外，我们在编译过程中加入 `LAMM_OPT_LEVEL` 宏来控制优化水平(LAMM表示LoongArch Matrix Multiplication)，便于测试比较：
- `LAMM_OPT_LEVEL=1`: 性能等于直接移植llama.cpp，不做任何平台优化，可见 `src/loongarch_matmul.cpp` 中的 `gemm_naive()`；
- `LAMM_OPT_LEVEL=2`: SIMD优化代码，可见`src/loongarch_matmul.cpp` 中的 `gemm_simd()`；
- `LAMM_OPT_LEVEL=3`: SIMD+Cache优化代码，可见 `src/loongarch_matmul.cpp` 中的 `gemm_block_simd()`.

### 4.3 编译测试
本项目在根目录提供了 `Makefile` 来完成编译(会递归调用 `llama.cpp-b2430/Makefile` )，包含两个target：
1. `benchmark`: 默认target，会在 `test/` 下编译出可执行文件 `la-benchmark-matmult`，用于测试F32和Q4_1矩阵乘法的FLOPS；
2. `main`：会在 `test/` 下编译出可执行文件 `main`，用于测试模型推理速度。

更具体地，要测试矩阵乘法性能，在项目根目录下运行以下指令：
```bash
make clean && make benchmark LAMM_OPT_LEVEL=[1|2|3]
./test/la-benchmark-matmult
```
要测试模型推理性能，须先下载模型文件，我们在 `model_weights/` 目录下提供了一个Python脚本，会自动从Huggingface下载Meta-Llama-2 的7B和13B模型（依赖`huggingface_hub`库），但注意，LLaMA的下载须申请授权，并获得相应的Token：
```bash
cd model_weights/
pip install huggingface_hub
HF_TOKEN=[YOUR_TOKEN] python llama_weights_download.py
```
然后，用llama.cpp提供的程序将下载的模型文件转成F32/Q4_1格式的GGUF文件：
```bash
# prerequisite
make clean && make src/loongarch_matmul.o LAMM_OPT_LEVEL=0
cd llama.cpp-b2430/
make quantize -j8 LLAMA_LOONGARCH=1
pip install -r requirements/requirements-convert.txt
# convert & quantize
python convert.py ../model_weights/Meta-Llama-2-7b --outfile ../model_weights/Meta-Llama-2-7B.F32.gguf --outtype f32
./quantize ../model_weights/Meta-Llama-2-7B.F32.gguf ../model_weights/Meta-Llama-2-7B.Q4_1.gguf Q4_1
```
最后，编译 `main` 并运行相应的GGUF文件进行推理：
```bash
make clean && make main LAMM_OPT_LEVEL=[0|1|2|3]
./test/main -m model_weights/Meta-Llama-2-[7B|13B].[F32|Q4_1].gguf -t 4 -n 512 -p "Building a website can be done in 10 simple steps:\nStep 1:"
```

### 4.4 测试结果
我们分别对矩阵乘法和模型推理两个任务进行基准测试。  
矩阵乘法的基准代码在 `test/la-benchmark-matmult.cpp` ，其修改自 llama.cpp 原项目中的 `examples/benchmark/benchmark-matmult.cpp` ，没有做实验设定上的修改，因此测试结果可直接与社区报告的结果进行比较。  
模型推理则直接用 llama.cpp 项目中的 `examples/main/main.cpp` 进行推理。  

对矩阵乘法任务，分别用F32和Q4_1两种数据格式进行测试，以gFLOPS(giga fLoating point operations per second)作为衡量指标；
对模型推理任务，使用 `Meta-LLaMA-2-7B` 和 `Meta-LLaMA-2-13B` 两种模型进行推理，以模型在prompt evaluation和text generation两阶段的token吞吐量作为衡量指标。在F32格式下，最小的7B参数Meta LLaMA 2模型也无法装进16G内存，因此，我们只进行Q4_1格式的量化推理（这也是llama.cpp项目中模型量化技术的重要性体现）。

对每个任务，都进行如下三组对比：
1. 直接移植：无任何龙芯平台特定优化，等价于 `LAMM_OPT_LEVEL=1` 的编译结果;
2. SIMD优化：包含SIMD优化的结果，等价于 `LAMM_OPT_LEVEL=2` 的编译结果；
3. SIMD+Cache优化：包含SIMD+Cache优化结果，等价于 `LAMM_OPT_LEVEL=3` 的编译结果。

对每个任务，分别测试单线程(t=1)和多线程(t=2/4)下的正确性及性能。

#### 4.4.1 矩阵乘法测试结果

| Matrix Multiplication Peformence (gFLOPS)            | F32 (t=1) | F32 (t=2) | F32 (t=4) | Q4_1 (t=1) | Q4_1 (t=2) | Q4_1 (t=4) |
| ------------------------ | ---------- | ---------- | ---------- | ----------- | ----------- | ----------- |
| 直接移植性能(LAMM_OPT_LEVEL=1)      | 1.67       | 3.34       | 6.67       | 4.91        | 9.77        | 18.96       |
| SIMD优化(LAMM_OPT_LEVEL=2)         | 12.89      | 24.71      | 44.11      | 25.98       | 51.39       | 88.84       |
| SIMD+Cache优化(LAMM_OPT_LEVEL=3)   | **59.34**  | **85.66**  | **128.46** | **39.45**   | **77.00**   | **123.32**  |

实验结果表明，本团队所作优化，在llama.cpp中矩阵乘法计算上可实现6x-35x的加速。


#### 4.4.2 模型推理测试结果

| Meta-LLaMA-2-7B Inference (Tokens/Sec) | Q4_1 prompt evaluation (t=1)| Q4_1 text generation (t=1)| Q4_1 prompt evaluation (t=4)| Q4_1 text generation (t=4)|
| ------------------------  | --------------------- | ------------------- | ---------------------- | -------------------- |
| 直接移植性能(LAMM_OPT_LEVEL=1)                   | 0.37               | 0.36                | 1.44                  | 1.37                 |
| SIMD优化(LAMM_OPT_LEVEL=2)                   | 1.48                  | 1.29                |  6.17                 | 3.30                  |
| SIMD+Cache优化(LAMM_OPT_LEVEL=3)             | **2.25**               | **1.54**            |  **8.56**            | **3.87**              |


| Meta-LLaMA2-13B Inference (Tokens/Sec) | Q4_1 prompt evaluation (t=1)| Q4_1 text generation (t=1)| Q4_1 prompt evaluation (t=4)| Q4_1 text generation (t=4)|
| ------------------------  | --------------------- | ------------------- | ---------------------- | -------------------- |
| 直接移植(LAMM_OPT_LEVEL=1)                   | 0.19                  | 0.19                | 0.74                   | 0.71                 |
| SIMD优化(LAMM_OPT_LEVEL=2)                   | 0.77                  | 0.69                | 2.99                   | 2.02                 |
| SIMD+Cache优化(LAMM_OPT_LEVEL=3)             |  **1.23**             | **0.82**            | **4.79**               | **2.30**             |

实验结果表明，本团队所作优化，在模型推理的吞吐量上可实现3x-6x的加速，其中prompt evaluation阶段的加速效果比text generation阶段更为明显。这是因为，相对来说，前者比后者更计算密集，后者更受制于内存访问。因此，对于直接移植未经优化的代码，prompt evaluation和text generation的推理性能是差不多的，而优化过的代码在text generation在达到瓶颈。访存优化也是下一阶段我们的重点优化目标。


## 5. 相关工作
llama.cpp是一个关注度很高且社区推动力很强的优秀开源项目。因此，与本项目同期的也有不少相关的优化工作，感谢张福新老师对我们的指点，让我们多学知识，少走弯路。

### 5.1 龙芯团队的CPU优化
龙芯内部团队针对矩阵乘法中的点积操作做了平台优化且向社区提交了[PR](https://github.com/ggerganov/llama.cpp/pull/6454)。该优化主要做了SIMD指令支持，我们项目中的SIMD工程抽象代码向他们多有借鉴，在此致谢。  
在其基础上，我们针对矩阵乘法整体做了Cache优化，实现更深入的优化加速，最终比较如下：

| Matrix Multiplication Peformence (gFLOPS)            | F32 (t=1) | F32 (t=2) | F32 (t=4) | Q4_1 (t=1) | Q4_1 (t=2) | Q4_1 (t=4) |
| ------------------------ | ---------- | ---------- | ---------- | ----------- | ----------- | ----------- |
| 龙芯团队PR         | 12.89      | 24.71      | 44.11      | 23.34       | 46.17       | 87.84       |
| 本项目优化(LAMM_OPT_LEVEL=3)   | **59.34**  | **85.66**  | **128.46** | **39.45**   | **77.00**   | **123.32**  |

### 5.2 Mozilla llamafile团队的优化
[llamafile](https://github.com/mozilla-Ocho/llamafile)是Mozilla公司支持的另一个针对模型推理的开源项目，团队中的开发者将部分CPU优化算子贡献到了llama.cpp并提交了[PR](https://github.com/ggerganov/llama.cpp/pull/6414)。其优化思路与我们类似，也是从SIMD加速和Cache优化两个方向。与本项目的主要区别在于，其主要针对Intel/ARM平台进行优化，本项目主要针对龙芯。另外，其实现了Q4_0量化方法的优化，本项目实现了Q4_1。 

## 6. 未来工作与收获总结
由于比赛时间和成员精力有限，本阶段所完成的工作距离理想目标还甚有欠缺，无论比赛是否继续，希望能够在未来补足，具体包括：
1. 对模型推理的进一步优化，例如Cache优化中分块参数（块形状）和分块策略的调优；
2. 对所有量化方式的优化的全面支持（目前只考虑了Q4_1）；
3. 对大量模型在龙芯平台的可用性的全面评测（目前只测评了Meta LLaMA 2）；
4. 针对模型推理过程的text generation阶段做进一步优化（目前prompt evaluation阶段效果更显著）；
5. 将代码整理成PR并入社区。

本次比赛对我们来说是一次宝贵的经历，让我们有机会真正接触一项开源项目并进行工程实操。
这其中包括不少挑战，例如需要理解并改进一个2M LOC量级的实际工程项目，需要快速理解和掌握一个新的指令集架构，需要对较为陌生的性能优化领域展开调研，等等。在克服这些挑战的过程中也收获了很多，一言蔽之是增进了系统能力，无论是阅读代码、查找资料、还是阅读手册，我们在这个过程中开始领悟如何在一个复杂的系统中开展工作。

感谢比赛方的张福新老师、殷时友老师、高燕萍老师、韩冰老师的耐心沟通和指导。

感谢指导教师中国科大王皓老师的鼎力支持和指导。