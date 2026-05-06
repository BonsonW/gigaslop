import sys
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from ampere_gemm_f16 import TensorOpGemm

# Simple test params - use larger size to match tile boundaries
batch_size = 512
timestep = 1024
out_features = 4096
in_features = 512

M = batch_size * timestep
K = in_features
N = out_features

# Create float input tensors
A_float = torch.randn((batch_size, timestep, in_features), dtype=torch.float16, device='cuda').reshape(batch_size * timestep, in_features)
B_float = torch.randn((out_features, in_features), dtype=torch.float16, device='cuda').reshape(out_features, in_features)
print(f"A_float shape: {A_float.shape}, B_float shape: {B_float.shape}")

# Create output buffer
C = torch.zeros((batch_size * timestep, out_features, 1), dtype=torch.float16, device='cuda')

print(f"C shape: {C.shape}")

def as_marked_cute_tensor(tensor, leading_dim=1):
    return (
        from_dlpack(tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=1,
            stride_order=(2, 0, 1),
            divisibility=(128 // cutlass.Float16.width),
        )
    )

# Convert to CUTE tensors
mA = as_marked_cute_tensor(A_float.reshape(batch_size * timestep, in_features, 1))
mB = as_marked_cute_tensor(B_float.reshape(out_features, in_features, 1))
mC = as_marked_cute_tensor(C)

# Setup and compile kernel
tensor_op_gemm = TensorOpGemm(
    cutlass.Float16,
    cutlass.Float16,
    cutlass.Float32,
    atom_layout_mnk=(2, 2, 1),
)

print('=== Compiling ampere_gemm kernel ===')
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)

print('=== Running GEMM === ')
compiled_gemm(mA, mB, mC)

# # Float reference: A_float @ B_float^T
# A_ref = A_float.reshape(batch_size * timestep, in_features)  # (M, K)
# B_ref = B_float.reshape(out_features, in_features).t()  # (K, N)
# C_ref = torch.matmul(A_ref, B_ref)  # (M, N)

# print(f"\nC shape: {C.shape}")
# print(f"C_ref shape: {C_ref.shape}")
# print(f"\nC[:3, :3]:\n{C[:3, :3, 0]}")
# print(f"C_ref[:3, :3]:\n{C_ref[:3, :3]}")

# # Compare
# C_ref_match = C_ref[:, :out_features]
# if torch.allclose(C[:, :, 0], C_ref_match, atol=2.0, rtol=1e-1):
#     print("\n✓ Results match!")
# else:
#     print("\n✗ Results differ")
#     print(f"Max diff: {(C[:, :, 0] - C_ref_match).abs().max()}")
#     print(f"Mean diff: {(C[:, :, 0] - C_ref_match).abs().mean()}")


# Benchmark
print("\n=== Benchmarking GEMM kernel ===")
def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    a_bytes     = M * K * (cutlass.Float16.width // 8)
    b_bytes     = N * K * (cutlass.Float16.width // 8)
    c_bytes     = M * N * 2   # float16 = 2 bytes
    total_bytes = a_bytes + b_bytes + c_bytes
    total_ops   = 2 * M * N * K

    avg_time_s          = avg_time_us * 1e-6
    achieved_bw_gbs     = (total_bytes / avg_time_s) / 1e9
    tops                = (total_ops   / avg_time_s) / 1e12

    peak_tops    = 624.0    # A100 INT8 tensor core peak
    peak_bw_gbs  = 2000.0   # A100 HBM peak

    print(f"Performance Metrics:")
    print(f"  Matrix shape:         M={M}, N={N}, K={K}")
    print(f"  Execution time:       {avg_time_us:.2f} us")
    print(f"  Compute:              {tops:.3f} FLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} FLOPS peak)")
    print(f"  Bandwidth:            {achieved_bw_gbs:.1f} GB/s  ({achieved_bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  Arithmetic intensity: {total_ops/total_bytes:.1f} FLOP/byte")
benchmark(compiled_gemm, mA, mB, mC)