#include <cuda_runtime.h>
#include <cstdio>
#include "print_tensor_example.h"   // generated header

// pull in the generated object file symbols
extern void _mlir_print_tensor_cuda_init(void **);
extern void _mlir_print_tensor_cuda_load_to_device(void **);
extern void _mlir_print_tensor__mlir_ciface_cutlass_print_tensor_FakeTensorOrderedFloat32i32div10_CUstream0x0(void **args, int32_t num_args);

int main() {
    // 1. load the cubin into the driver
    print_tensor_Kernel_Module_t mod;
    print_tensor_Kernel_Module_Load(&mod);

    // 2. allocate a small float tensor on device
    const int N = 16;
    float *d_a = nullptr;
    cudaMalloc(&d_a, N * sizeof(float));

    // fill with 0..15 on host, copy to device
    float h_a[N];
    for (int i = 0; i < N; i++) h_a[i] = (float)i;
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. build the tensor descriptor the generated wrapper expects
    print_tensor_Tensor_a_t a;
    a.data            = d_a;
    a.dynamic_shapes[0] = N;   // matches the sym_int() dimension from compile

    // 4. call the wrapper
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int32_t ret = cute_dsl_print_tensor_wrapper(&mod, &a, stream);
    if (ret != 0)
        printf("kernel returned error code: %d\n", ret);

    cudaStreamSynchronize(stream);

    // 5. cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    print_tensor_Kernel_Module_Unload(&mod);

    return 0;
}