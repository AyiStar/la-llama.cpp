#ifndef LAMM_KERNEL_Q2_K_HPP
#define LAMM_KERNEL_Q2_K_HPP

#include "lamm_common.h"
#include <cstdint>

/*
# define QK_K 256
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
} block_q2_K;

typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
*/

LA_INLINE void lamm_naive_kernel(const block_q2_K *a, const block_q8_K *b,
                                 float *c, int64_t lda, int64_t ldb,
                                 int64_t ldc, int i, int j, int K) {
  constexpr int Q = ggml_type_trait<GGML_TYPE_Q2_K>::super_block_size;
  static_assert(Q == 256);
  float sumf = 0.0;
  auto *aik = a + (i * lda);
  auto *bjk = b + (j * ldb);
  for (int k = 0; k < K; k++, aik++, bjk++) {
    const uint8_t *q2 = aik->qs;
    const int8_t *q8 = bjk->qs;
    const uint8_t *sc = aik->scales;

    int summs = 0;
    for (int j = 0; j < Q / 16; ++j) {
      summs += bjk->bsums[j] * (sc[j] >> 4);
    }

    const float dall = bjk->d * GGML_FP16_TO_FP32(aik->d);
    const float dmin = bjk->d * GGML_FP16_TO_FP32(aik->dmin);

    int isum = 0;
    int is = 0;
    int d;
    for (int k = 0; k < 2; ++k) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        d = sc[is++] & 0xF;
        int isuml = 0;
        for (int l = 0; l < 16; ++l) {
          isuml += q8[l] * ((q2[l] >> shift) & 3);
        }
        isum += d * isuml;
        d = sc[is++] & 0xF;
        isuml = 0;
        for (int l = 16; l < 32; ++l) {
          isuml += q8[l] * ((q2[l] >> shift) & 3);
        }
        isum += d * isuml;
        shift += 2;
        q8 += 32;
      }
      q2 += 32;
    }
    sumf += dall * isum - dmin * summs;
  }
  c[j * ldc + i] = sumf;
}

static LA_INLINE simd::ivreg_t get_scale_shuffle_q3k(int i) {
  static const uint8_t k_shuffle[128] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };
  return simd::load((const char *)((const simd::ivreg_t *)k_shuffle + i));
}

LA_INLINE void lamm_simd_kernel(const block_q2_K *a, const block_q8_K *b,
                                float *c, int64_t lda, int64_t ldb, int64_t ldc,
                                int i, int j, int K) {
  using namespace simd;

  auto *aik = a + (i * lda);
  auto *bjk = b + (j * ldb);

  const ivreg_t m3 = ivset(3);
  const hivreg_t m4 = hivset(0xF);
  vreg_t acc = {0};
  for (int k = 0; k < K; k++, aik++, bjk++) {

    // for b
    const char *q8 = reinterpret_cast<const char *>(bjk->qs);
    const ivreg_t q8_0[2] = {load(q8 + 0), load(q8 + 128)};
    const ivreg_t q8_1[2] = {load(q8 + 32), load(q8 + 160)};
    const ivreg_t q8_2[2] = {load(q8 + 64), load(q8 + 192)};
    const ivreg_t q8_3[2] = {load(q8 + 96), load(q8 + 224)};
    const ivreg_t bsum = load(reinterpret_cast<const char *>(bjk->bsums));
    const float dmin = -bjk->d * GGML_FP16_TO_FP32(aik->dmin);

    // for a
    const char *q2 = reinterpret_cast<const char *>(aik->qs);
    const hivreg_t mins_and_scales =
        loadh(reinterpret_cast<const char *>(aik->scales));
    const hivreg_t scales8 = _and(mins_and_scales, m4);
    const hivreg_t mins8 = _and(logic_shift_right(mins_and_scales, 4), m4);
    const ivreg_t mins = extend(mins8);
    const ivreg_t all_scales = extend(scales8);
    const hivreg_t l_scales = trunc(all_scales, 0);
    const hivreg_t h_scales = trunc(all_scales, 1);
    const ivreg_t scales[2] = {concat(l_scales, l_scales),
                               concat(h_scales, h_scales)};

    ivreg_t sumi = {0};
    const float d = bjk->d * GGML_FP16_TO_FP32(aik->d);

    for (int l = 0; l < 2; ++l) {

      // for a
      const ivreg_t q2bits = load(q2);
      q2 += 32;
      const ivreg_t q2_0 = _and(q2bits, m3);
      const ivreg_t q2_1 = _and(logic_shift_right(q2bits, 2), m3);
      const ivreg_t q2_2 = _and(logic_shift_right(q2bits, 4), m3);
      const ivreg_t q2_3 = _and(logic_shift_right(q2bits, 6), m3);

      // for a and b
      ivreg_t p0 = mul_ubs(q2_0, q8_0[l]);
      ivreg_t p1 = mul_ubs(q2_1, q8_1[l]);
      ivreg_t p2 = mul_ubs(q2_2, q8_2[l]);
      ivreg_t p3 = mul_ubs(q2_3, q8_3[l]);

      p0 = mul(shuffle(scales[l], get_scale_shuffle_q3k(0)), p0);
      p1 = mul(shuffle(scales[l], get_scale_shuffle_q3k(1)), p1);
      p2 = mul(shuffle(scales[l], get_scale_shuffle_q3k(2)), p2);
      p3 = mul(shuffle(scales[l], get_scale_shuffle_q3k(3)), p3);

      p0 = add(p0, p1);
      p2 = add(p2, p3);

      sumi = add(sumi, add(p0, p2));
    }

    const ivreg_t prod = mul(mins, bsum);
    acc = madd(vset(dmin), to_float(prod), acc);
    acc = madd(vset(d), to_float(sumi), acc);
  }
  c[j * ldc + i] = reduce_sum(acc);
}

template <int B0, int B1>
LA_INLINE void lamm_simd_block_kernel(const block_q2_K *a, const block_q8_K *b,
                                      float *c, int64_t lda, int64_t ldb,
                                      int64_t ldc, int i, int j, int K) {

  static_assert(B0 > 0 && B0 <= 4);
  static_assert(B1 > 0 && B1 <= 4);
  // std::cout << "Q2_K SIMD block called with B0=" << B0 << ", B1=" << B1 <<
  // std::endl;
  using namespace simd;

  const ivreg_t m3 = ivset(3);
  const hivreg_t m4 = hivset(0xF);

  [[maybe_unused]] ivreg_t q8_00[2] = {0}, q8_01[2] = {0}, q8_02[2] = {0},
                           q8_03[2] = {0};
  [[maybe_unused]] ivreg_t q8_10[2] = {0}, q8_11[2] = {0}, q8_12[2] = {0},
                           q8_13[2] = {0};
  [[maybe_unused]] ivreg_t q8_20[2] = {0}, q8_21[2] = {0}, q8_22[2] = {0},
                           q8_23[2] = {0};
  [[maybe_unused]] ivreg_t q8_30[2] = {0}, q8_31[2] = {0}, q8_32[2] = {0},
                           q8_33[2] = {0};

  [[maybe_unused]] ivreg_t bsum0 = {0}, bsum1 = {0}, bsum2 = {0}, bsum3 = {0};
  [[maybe_unused]] vreg_t bd0 = {0}, bd1 = {0}, bd2 = {0}, bd3 = {0};

  [[maybe_unused]] vreg_t acc00 = {0}, acc01 = {0}, acc02 = {0}, acc03 = {0};
  [[maybe_unused]] vreg_t acc10 = {0}, acc11 = {0}, acc12 = {0}, acc13 = {0};
  [[maybe_unused]] vreg_t acc20 = {0}, acc21 = {0}, acc22 = {0}, acc23 = {0};
  [[maybe_unused]] vreg_t acc30 = {0}, acc31 = {0}, acc32 = {0}, acc33 = {0};

  [[maybe_unused]] const auto *ai0{a + ((i + 0) * lda)};
  [[maybe_unused]] const auto *ai1{a + ((i + 1) * lda)};
  [[maybe_unused]] const auto *ai2{a + ((i + 2) * lda)};
  [[maybe_unused]] const auto *ai3{a + ((i + 3) * lda)};
  [[maybe_unused]] const auto *bj0{b + ((j + 0) * ldb)};
  [[maybe_unused]] const auto *bj1{b + ((j + 1) * ldb)};
  [[maybe_unused]] const auto *bj2{b + ((j + 2) * ldb)};
  [[maybe_unused]] const auto *bj3{b + ((j + 3) * ldb)};

  for (int k = 0; k < K; k++) {
#define FN(N1)                                                                 \
  if constexpr (B1 > N1) {                                                     \
    const char *q8 = reinterpret_cast<const char *>(bj##N1->qs);               \
    q8_0##N1[0] = load(q8 + 0);                                                \
    q8_1##N1[0] = load(q8 + 32);                                               \
    q8_2##N1[0] = load(q8 + 64);                                               \
    q8_3##N1[0] = load(q8 + 96);                                               \
    q8_0##N1[1] = load(q8 + 128);                                              \
    q8_1##N1[1] = load(q8 + 160);                                              \
    q8_2##N1[1] = load(q8 + 192);                                              \
    q8_3##N1[1] = load(q8 + 224);                                              \
    bsum##N1 = load(reinterpret_cast<const char *>(bj##N1->bsums));            \
    bd##N1 = vset(bj##N1->d);                                                  \
  }
    LOOP(FN, 4)
#undef FN

#define INNER_FN(N0, N1)                                                       \
  if constexpr (B1 > N1) {                                                     \
    ivreg_t p0 = mul_ubs(q2_00, q8_0##N1[0]);                                  \
    ivreg_t p1 = mul_ubs(q2_10, q8_1##N1[0]);                                  \
    ivreg_t p2 = mul_ubs(q2_20, q8_2##N1[0]);                                  \
    ivreg_t p3 = mul_ubs(q2_30, q8_3##N1[0]);                                  \
    p0 = mul(shuffle(scales[0], get_scale_shuffle_q3k(0)), p0);                \
    p1 = mul(shuffle(scales[0], get_scale_shuffle_q3k(1)), p1);                \
    p2 = mul(shuffle(scales[0], get_scale_shuffle_q3k(2)), p2);                \
    p3 = mul(shuffle(scales[0], get_scale_shuffle_q3k(3)), p3);                \
    p0 = add(p0, p1);                                                          \
    p2 = add(p2, p3);                                                          \
    ivreg_t sumi = add(p0, p2);                                                \
                                                                               \
    p0 = mul_ubs(q2_01, q8_0##N1[1]);                                          \
    p1 = mul_ubs(q2_11, q8_1##N1[1]);                                          \
    p2 = mul_ubs(q2_21, q8_2##N1[1]);                                          \
    p3 = mul_ubs(q2_31, q8_3##N1[1]);                                          \
    p0 = mul(shuffle(scales[1], get_scale_shuffle_q3k(0)), p0);                \
    p1 = mul(shuffle(scales[1], get_scale_shuffle_q3k(1)), p1);                \
    p2 = mul(shuffle(scales[1], get_scale_shuffle_q3k(2)), p2);                \
    p3 = mul(shuffle(scales[1], get_scale_shuffle_q3k(3)), p3);                \
    p0 = add(p0, p1);                                                          \
    p2 = add(p2, p3);                                                          \
    sumi = add(sumi, add(p0, p2));                                             \
                                                                               \
    acc##N0##N1 = madd(mul(ad, bd##N1), to_float(sumi), acc##N0##N1);          \
    const ivreg_t prod = mul(mins, bsum##N1);                                  \
    acc##N0##N1 = madd(mul(admin, bd##N1), to_float(prod), acc##N0##N1);       \
  }
#define OUTER_FN(N0)                                                           \
  if constexpr (B0 > N0) {                                                     \
    const char *q2 = reinterpret_cast<const char *>(ai##N0->qs);               \
    const hivreg_t mins_and_scales =                                           \
        loadh(reinterpret_cast<const char *>(ai##N0->scales));                 \
    const hivreg_t scales8 = _and(mins_and_scales, m4);                        \
    const hivreg_t mins8 = _and(logic_shift_right(mins_and_scales, 4), m4);    \
    const ivreg_t mins = extend(mins8);                                        \
    const ivreg_t all_scales = extend(scales8);                                \
    const hivreg_t l_scales = trunc(all_scales, 0);                            \
    const hivreg_t h_scales = trunc(all_scales, 1);                            \
    const ivreg_t scales[2] = {concat(l_scales, l_scales),                     \
                               concat(h_scales, h_scales)};                    \
                                                                               \
    ivreg_t q2bits = load(q2);                                                 \
    const ivreg_t q2_00 = _and(q2bits, m3);                                    \
    const ivreg_t q2_10 = _and(logic_shift_right(q2bits, 2), m3);              \
    const ivreg_t q2_20 = _and(logic_shift_right(q2bits, 4), m3);              \
    const ivreg_t q2_30 = _and(logic_shift_right(q2bits, 6), m3);              \
                                                                               \
    q2bits = load(q2 + 32);                                                    \
    const ivreg_t q2_01 = _and(q2bits, m3);                                    \
    const ivreg_t q2_11 = _and(logic_shift_right(q2bits, 2), m3);              \
    const ivreg_t q2_21 = _and(logic_shift_right(q2bits, 4), m3);              \
    const ivreg_t q2_31 = _and(logic_shift_right(q2bits, 6), m3);              \
                                                                               \
    const vreg_t ad = vset(GGML_FP16_TO_FP32(ai##N0->d));                      \
    const vreg_t admin = vset(-GGML_FP16_TO_FP32(ai##N0->dmin));               \
    LOOP_INNER(INNER_FN, N0, 4)                                                \
  }
    LOOP(OUTER_FN, 4)
#undef INNER_FN
#undef OUTER_FN

#define FN(N)                                                                  \
  if constexpr (B0 > N) {                                                      \
    ai##N++;                                                                   \
  }                                                                            \
  if constexpr (B1 > N) {                                                      \
    bj##N++;                                                                   \
  }
    LOOP(FN, 4)
#undef FN
  } // loop `k`

#define INNER_FN(N0, N1)                                                       \
  if constexpr (B1 > N1) {                                                     \
    c[ldc * (j + N1) + (i + N0)] = reduce_sum(acc##N0##N1);                    \
  }
#define OUTER_FN(N0)                                                           \
  if constexpr (B0 > N0) {                                                     \
    LOOP_INNER(INNER_FN, N0, 4)                                                \
  }
  LOOP(OUTER_FN, 4)
#undef INNER_FN
#undef OUTER_FN
}

#endif // LAMM_KERNEL_Q2_K_HPP