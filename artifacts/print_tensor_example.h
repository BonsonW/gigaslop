
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} print_tensor_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_print_tensor_cuda_init(void **);
void _mlir_print_tensor_cuda_load_to_device(void **);
static inline void print_tensor_Kernel_Module_Load(print_tensor_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_print_tensor_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_print_tensor_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void print_tensor_Kernel_Module_Unload(print_tensor_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} print_tensor_Tensor_a_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_print_tensor__mlir_ciface_cutlass_print_tensor_FakeTensorFloat32div11_CUstream0x0(void **args, int32_t num_args);

static inline int32_t cute_dsl_print_tensor_wrapper(print_tensor_Kernel_Module_t *module, print_tensor_Tensor_a_t *a, cudaStream_t stream) {
    int32_t ret;
    void *args[3] = {
        a, &stream,
        &ret
    };
    _mlir_print_tensor__mlir_ciface_cutlass_print_tensor_FakeTensorFloat32div11_CUstream0x0(args, 3);
    return ret;
}
