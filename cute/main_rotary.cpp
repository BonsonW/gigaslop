// AOT / C-ABI test for the fused INT8 GEMM + rotary kernel.
//
// This exercises the export_to_c path (the .o + generated .h), which the Python
// JIT test (test_gemm_i8_rotary_vs_openfish.py) never touches. It reproduces the
// exact C ABI slorado uses: build the 7 tensor descriptors, call the wrapper,
// then check the result against a host reference (dequant GEMM + rotate-half).
//
// Build/run:  cute/build_rotary_test.sh
//
// Matches the exported kernel gemm_i8_rotary_N1536_K512_H8D64R64S1024:
//   nhead=8 head_dim=64 rotary_dim=64 seqlen=1024  ->  N=1536, K=512.
// scale/sin/cos shapes are baked at export, so M is pinned to the exported
// m_size (256) — the scale tensor descriptor carries no dynamic shape.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "artifacts/gemm_i8_rotary_N1536_K512_H8D64R64S1024.h"

#define PFX gemm_i8_rotary_N1536_K512_H8D64R64S1024

// token-paste helpers so we can use the (long) exported prefix ergonomically
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define T(suffix) CAT(PFX, suffix)

// ---- problem dims (must match the exported kernel) --------------------------
static constexpr int kNHead = 8;
static constexpr int kHeadDim = 64;
static constexpr int kRotaryDim = 64;
static constexpr int kSeqLen = 1024;
static constexpr int kRotaryHalf = kRotaryDim / 2;   // 32

static constexpr int M = 256;                         // == exported m_size
static constexpr int K = 512;
static constexpr int N = 3 * kNHead * kHeadDim;       // 1536
static constexpr int L = 1;

static constexpr int kQCols = kNHead * kHeadDim;      // [0, kQCols)      -> Q
static constexpr int kKCols = 2 * kNHead * kHeadDim;  // [kQCols, kKCols) -> K
                                                      // [kKCols, N)      -> V

static void fill_float_matrix(std::vector<float> &m, int rows, int cols, int seed) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float x = 0.021f * r + 0.037f * c + 0.11f * seed;
            m[r * cols + c] = 1.7f * std::sin(x) + 0.8f * std::cos(1.9f * x);
        }
}

static void quantize_per_row(const std::vector<float> &in, int rows, int cols,
                             std::vector<int8_t> &out, std::vector<float> &dequant) {
    constexpr float kQMax = 127.0f, kEps = 1e-8f;
    for (int r = 0; r < rows; ++r) {
        float amax = kEps;
        for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(in[r * cols + c]));
        float qs = kQMax / amax;
        dequant[r] = 1.0f / qs;
        for (int c = 0; c < cols; ++c) {
            int q = static_cast<int>(std::round(in[r * cols + c] * qs));
            out[r * cols + c] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
        }
    }
}

int main() {
    // ---- host inputs --------------------------------------------------------
    std::vector<float> h_a_f(M * K), h_b_f(N * K);
    std::vector<int8_t> h_a(M * K), h_b(N * K);
    std::vector<float> h_scale_a(M), h_scale_b(N);
    std::vector<__half> h_c(M * N, __float2half(0.0f));

    fill_float_matrix(h_a_f, M, K, 1);
    fill_float_matrix(h_b_f, N, K, 7);
    quantize_per_row(h_a_f, M, K, h_a, h_scale_a);
    quantize_per_row(h_b_f, N, K, h_b, h_scale_b);

    // Canonical RoPE sin/cos [seqlen, rotary_half]  (matches the JIT test).
    // ROTARY_IDENTITY=1 fills a constant table (sin=0, cos=1) so rotary is a
    // no-op: Q/K must then equal the plain GEMM. A constant table masks any
    // (seq,rot) indexing error, so PASS there + FAIL below pins the bug to
    // sin/cos indexing rather than the argument wiring.
    const bool identity = std::getenv("ROTARY_IDENTITY") != nullptr;
    std::vector<float> h_sin(kSeqLen * kRotaryHalf), h_cos(kSeqLen * kRotaryHalf);
    for (int s = 0; s < kSeqLen; ++s)
        for (int i = 0; i < kRotaryHalf; ++i) {
            float inv_freq = std::pow(10000.0f, -(2.0f * i) / kHeadDim);
            float ang = s * inv_freq;
            h_sin[s * kRotaryHalf + i] = identity ? 0.0f : std::sin(ang);
            h_cos[s * kRotaryHalf + i] = identity ? 1.0f : std::cos(ang);
        }

    // ---- device buffers -----------------------------------------------------
    int8_t *d_a = nullptr, *d_b = nullptr;
    __half *d_c = nullptr;
    float *d_scale_a = nullptr, *d_scale_b = nullptr, *d_sin = nullptr, *d_cos = nullptr;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(int8_t)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(int8_t)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(__half)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_scale_a, h_scale_a.size() * sizeof(float)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_scale_b, h_scale_b.size() * sizeof(float)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_sin, h_sin.size() * sizeof(float)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_cos, h_cos.size() * sizeof(float)));

    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_scale_a, h_scale_a.data(), h_scale_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_scale_b, h_scale_b.data(), h_scale_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(__half)));

    // ---- load module + build descriptors ------------------------------------
    T(_Kernel_Module_t) module{};
    T(_Kernel_Module_Load)(&module);

    // A,B,C are (rows, contraction, L) K-major: strides = (contraction, 1, rows*contraction)
    T(_Tensor_mA_t) mA{};
    mA.data = d_a;
    mA.dynamic_shapes[0] = M; mA.dynamic_shapes[1] = K; mA.dynamic_shapes[2] = L;
    mA.dynamic_strides[0] = K; mA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

    T(_Tensor_mB_t) mB{};
    mB.data = d_b;
    mB.dynamic_shapes[0] = N; mB.dynamic_shapes[1] = K; mB.dynamic_shapes[2] = L;
    mB.dynamic_strides[0] = K; mB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

    T(_Tensor_mC_t) mC{};
    mC.data = d_c;
    mC.dynamic_shapes[0] = M; mC.dynamic_shapes[1] = N; mC.dynamic_shapes[2] = L;
    mC.dynamic_strides[0] = N; mC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

    T(_Tensor_mScaleA_t) mScaleA{d_scale_a};
    T(_Tensor_mScaleB_t) mScaleB{d_scale_b};
    T(_Tensor_mSin_t) mSin{d_sin};
    T(_Tensor_mCos_t) mCos{d_cos};

    int32_t ret = CAT(cute_dsl_, T(_wrapper))(&module, &mA, &mB, &mC, &mScaleA, &mScaleB, &mSin, &mCos);
    if (ret != 0) std::printf("kernel returned error code: %d\n", ret);
    CUTE_DSL_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    // ---- host reference: dequant GEMM, then rotate-half on Q/K --------------
    std::vector<float> ref(M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += (static_cast<float>(h_a[r * K + k]) * h_scale_a[r]) *
                       (static_cast<float>(h_b[c * K + k]) * h_scale_b[c]);
            ref[r * N + c] = acc;
        }
    for (int r = 0; r < M; ++r) {
        int seq = r % kSeqLen;
        for (int head0 = 0; head0 < kKCols; head0 += kHeadDim) {   // Q chunk, then K chunk
            for (int i = 0; i < kRotaryHalf; ++i) {
                float cs = h_cos[seq * kRotaryHalf + i], sn = h_sin[seq * kRotaryHalf + i];
                float x0 = ref[r * N + head0 + i];
                float x1 = ref[r * N + head0 + kRotaryHalf + i];
                ref[r * N + head0 + i] = x0 * cs - x1 * sn;
                ref[r * N + head0 + kRotaryHalf + i] = x0 * sn + x1 * cs;
            }
        }
    }

    // ---- compare (in fp16, per region) --------------------------------------
    struct Region { const char *name; int lo, hi; } regions[] = {
        {"Q (rotary)     ", 0, kQCols},
        {"K (rotary)     ", kQCols, kKCols},
        {"V (passthrough)", kKCols, N},
    };
    // fp16 output at magnitude ~hundreds carries ~0.5 abs rounding, so gate on a
    // magnitude-relative tolerance: fail only when BOTH abs and rel are exceeded.
    constexpr float kAbsTol = 0.05f, kRelTol = 0.01f;
    float global_max = 0.0f;
    int fail = 0;
    std::printf("gemm_i8_rotary AOT test  M=%d N=%d K=%d  (ret=%d)\n", M, N, K, ret);
    for (auto &rg : regions) {
        float rmax = 0.0f, rmax_rel = 0.0f;
        double rsum = 0.0;
        long cnt = 0, rfail = 0;
        for (int r = 0; r < M; ++r)
            for (int c = rg.lo; c < rg.hi; ++c) {
                float got = __half2float(h_c[r * N + c]);
                float exp = __half2float(__float2half(ref[r * N + c]));
                float e = std::fabs(got - exp);
                float rel = e / std::max(std::fabs(exp), 1e-6f);
                rmax = std::max(rmax, e);
                rmax_rel = std::max(rmax_rel, rel);
                rsum += e;
                ++cnt;
                if (e > kAbsTol && rel > kRelTol) ++rfail;
            }
        float rmean = static_cast<float>(rsum / cnt);
        global_max = std::max(global_max, rmax);
        if (rfail > 0) fail = 1;
        std::printf("  %s: max_abs=%.6f  max_rel=%.6f  mean_abs=%.8f  bad=%ld/%ld\n",
                    rg.name, rmax, rmax_rel, rmean, rfail, cnt);
    }
    std::printf("top-left 4x4 of Q:\n");
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) std::printf("%9.4f", __half2float(h_c[r * N + c]));
        std::printf("\n");
    }
    std::printf(fail ? "\nFAIL (rotary output does not match reference)\n"
                     : "\nPASS (max abs err %.4f)\n", global_max);

    T(_Kernel_Module_Unload)(&module);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_scale_a); cudaFree(d_scale_b); cudaFree(d_sin); cudaFree(d_cos);
    return fail;
}
