#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "artifacts/gemm_i8_standard.h"

// pull in the generated object file symbols
extern void _mlir_gemm_i8_std_cuda_init(void **);
extern void _mlir_gemm_i8_std_cuda_load_to_device(void **);
extern void _mlir_gemm_i8_std__mlir_ciface_cutlass___call___ampere_gemm_i8_quant_rmemTensorOpGemmI8_object_at__Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv8i64div81i64div8_FakeTensorFlo(void **args, int32_t num_args);

static void fill_float_matrix(std::vector<float> &matrix, int rows, int cols, int seed) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float x = 0.021f * static_cast<float>(row) + 0.037f * static_cast<float>(col) + 0.11f * static_cast<float>(seed);
            matrix[row * cols + col] = 1.7f * std::sin(x) + 0.8f * std::cos(1.9f * x);
        }
    }
}

static void quantize_per_row(
    const std::vector<float> &input,
    int rows,
    int cols,
    std::vector<int8_t> &output,
    std::vector<float> &dequant_scale
) {
    constexpr float kQuantMax = 127.0f;
    constexpr float kEps = 1e-8f;

    for (int row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float v = std::fabs(input[row * cols + col]);
            max_abs = std::max(max_abs, v);
        }

        max_abs = std::max(max_abs, kEps);
        float quant_scale = kQuantMax / max_abs;
        dequant_scale[row] = 1.0f / quant_scale;

        for (int col = 0; col < cols; ++col) {
            float scaled = input[row * cols + col] * quant_scale;
            float rounded = std::round(scaled);
            int q = static_cast<int>(rounded);
            q = std::max(-127, std::min(127, q));
            output[row * cols + col] = static_cast<int8_t>(q);
        }
    }
}

int main() {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int L = 1;

    std::vector<float> h_a_float(M * K);
    std::vector<float> h_b_float(N * K);
    std::vector<int8_t> h_a(M * K);
    std::vector<int8_t> h_b(N * K);
    std::vector<float> h_scale_a(M);
    std::vector<float> h_scale_b(N);
    std::vector<__half> h_c(M * N);

    fill_float_matrix(h_a_float, M, K, 1);
    fill_float_matrix(h_b_float, N, K, 7);
    quantize_per_row(h_a_float, M, K, h_a, h_scale_a);
    quantize_per_row(h_b_float, N, K, h_b, h_scale_b);

    int8_t *d_a = nullptr;
    int8_t *d_b = nullptr;
    __half *d_c = nullptr;
    float *d_scale_a = nullptr;
    float *d_scale_b = nullptr;

    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(int8_t)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(int8_t)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(__half)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_scale_a, h_scale_a.size() * sizeof(float)));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMalloc(&d_scale_b, h_scale_b.size() * sizeof(float)));

    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_scale_a, h_scale_a.data(), h_scale_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(d_scale_b, h_scale_b.data(), h_scale_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(__half)));

    gemm_i8_std_Kernel_Module_t module{};
    gemm_i8_std_Kernel_Module_Load(&module);

    gemm_i8_std_Tensor_mA_t mA{};
    mA.data = d_a;
    mA.dynamic_shapes[0] = M;
    mA.dynamic_shapes[1] = K;
    mA.dynamic_shapes[2] = L;
    mA.dynamic_strides[0] = K;
    mA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

    gemm_i8_std_Tensor_mB_t mB{};
    mB.data = d_b;
    mB.dynamic_shapes[0] = N;
    mB.dynamic_shapes[1] = K;
    mB.dynamic_shapes[2] = L;
    mB.dynamic_strides[0] = K;
    mB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

    gemm_i8_std_Tensor_mC_t mC{};
    mC.data = d_c;
    mC.dynamic_shapes[0] = M;
    mC.dynamic_shapes[1] = N;
    mC.dynamic_shapes[2] = L;
    mC.dynamic_strides[0] = N;
    mC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

    gemm_i8_std_Tensor_mScaleA_t mScaleA{d_scale_a};
    gemm_i8_std_Tensor_mScaleB_t mScaleB{d_scale_b};

    int32_t ret = cute_dsl_gemm_i8_std_wrapper(&module, &mA, &mB, &mC, &mScaleA, &mScaleB);
    if (ret != 0) {
        std::printf("kernel returned error code: %d\n", ret);
    }

    CUTE_DSL_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUTE_DSL_CUDA_ERROR_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float max_abs_err = 0.0f;
    double sum_abs_err = 0.0;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float ref_fp32 = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_dq = static_cast<float>(h_a[row * K + k]) * h_scale_a[row];
                float b_dq = static_cast<float>(h_b[col * K + k]) * h_scale_b[col];
                ref_fp32 += a_dq * b_dq;
            }

            float got = __half2float(h_c[row * N + col]);
            float ref = __half2float(__float2half(ref_fp32));
            float err = std::fabs(got - ref);
            sum_abs_err += static_cast<double>(err);
            if (err > 0.5f) {
                ++mismatches;
            }
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }
    }
    float mean_abs_err = static_cast<float>(sum_abs_err / static_cast<double>(M * N));

    std::printf("gemm_i8_standard %dx%dx%d completed\n", M, N, K);
    std::printf("max abs error: %.4f\n", max_abs_err);
    std::printf("mean abs error: %.4f\n", mean_abs_err);
    std::printf("mismatches: %d\n", mismatches);
    std::printf("scale A range: [%.6f, %.6f]\n", *std::min_element(h_scale_a.begin(), h_scale_a.end()), *std::max_element(h_scale_a.begin(), h_scale_a.end()));
    std::printf("scale B range: [%.6f, %.6f]\n", *std::min_element(h_scale_b.begin(), h_scale_b.end()), *std::max_element(h_scale_b.begin(), h_scale_b.end()));
    std::printf("top-left 4x4 output:\n");
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::printf("%8.1f", __half2float(h_c[row * N + col]));
        }
        std::printf("\n");
    }

    gemm_i8_std_Kernel_Module_Unload(&module);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_scale_a);
    cudaFree(d_scale_b);

    return mismatches == 0 ? 0 : 1;
}