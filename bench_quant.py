import sys
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm_i8_quant import TensorOpGemmI8

def quantize_tensor(t, dim=-1):
    """Quantize float tensor to int8 with optional per-tensor scaling.

    Uses symmetric int8 range [-127, 127]. Handles zero ranges safely by
    clamping the FP range to a small epsilon.
    """
    levels = 256
    quant_max = (levels // 2) - 1  # 127

    fp_range = t.abs().amax(dim=dim)
    fp_range = fp_range.clamp_min(1e-8)
    quant_scale = quant_max / fp_range  # 127 / fp_range

    t_int8 = (t * quant_scale.unsqueeze(dim)).round().clamp(-quant_max, quant_max).to(torch.int8)
    dequant_scale = quant_scale.to(torch.float32).reciprocal()

    return t_int8, dequant_scale

# Simple test params - use larger size to match tile boundaries
batch_size = 512
timestep = 1024
out_features = 512
in_features = 512

# Create float input tensors
A_float = torch.randn((batch_size, timestep, in_features), dtype=torch.float16, device='cuda').reshape(batch_size * timestep, in_features)
B_float = torch.randn((out_features, in_features), dtype=torch.float16, device='cuda').reshape(out_features, in_features)
print(f"A_float shape: {A_float.shape}, B_float shape: {B_float.shape}")

# Quantize to int8
A_int8, A_scale = quantize_tensor(A_float, dim=-1)
B_int8, B_scale = quantize_tensor(B_float, dim=-1)

A_int8 = A_int8.reshape(batch_size * timestep, in_features, 1)
B_int8 = B_int8.reshape(out_features, in_features, 1)

A_scale = A_scale.reshape(A_scale.size(0), 1)
B_scale = B_scale.reshape(B_scale.size(0), 1)

# print(f"A_scale: {A_scale}, B_scale: {B_scale}")
print(f"A_int8 shape: {A_int8.shape}, B_int8 shape: {B_int8.shape}")
print(f"A_scale shape: {A_scale.shape}, B_scale shape: {B_scale.shape}")

# Create output buffer
C = torch.zeros((batch_size * timestep, out_features, 1), dtype=torch.float16, device='cuda')

print(f"C shape: {C.shape}")

# Convert to CUTE tensors
mA = from_dlpack(A_int8, assumed_align=16)
mB = from_dlpack(B_int8, assumed_align=16)
mC = from_dlpack(C, assumed_align=16)

mAScale = from_dlpack(A_scale, assumed_align=16)
mBScale = from_dlpack(B_scale, assumed_align=16)

# Setup and compile kernel
tensor_op_gemm = TensorOpGemmI8(
    cutlass.Int8,
    cutlass.Int8,
    cutlass.Float16,
    cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    use_k32=False,
    bm=128,
)

print('=== Compiling ampere_gemm kernel ===')
# compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC, mAScale, mBScale)

print('=== Running GEMM === ')
compiled_gemm(mA, mB, mC, mAScale, mBScale)

# Dequantize output: C_int32 * A_scale * B_scale
C_dequantized = C

# Float reference: A_float @ B_float^T
A_ref = A_float.reshape(batch_size * timestep, in_features)  # (M, K)
B_ref = B_float.reshape(out_features, in_features).t()  # (K, N)
C_ref = torch.matmul(A_ref, B_ref)  # (M, N)

print(f"\nC_dequantized shape: {C_dequantized.shape}")
print(f"C_ref shape: {C_ref.shape}")
print(f"\nC_dequantized[:3, :3]:\n{C_dequantized[:3, :3, 0]}")
print(f"C_ref[:3, :3]:\n{C_ref[:3, :3]}")

# Compare
C_ref_match = C_ref[:, :out_features]
if torch.allclose(C_dequantized[:, :, 0], C_ref_match, atol=2.0, rtol=1e-1):
    print("\n✓ Results match!")
else:
    print("\n✗ Results differ")
    print(f"Max diff: {(C_dequantized[:, :, 0] - C_ref_match).abs().max()}")
    print(f"Mean diff: {(C_dequantized[:, :, 0] - C_ref_match).abs().mean()}")


# Benchmark
print("\n=== Benchmarking GEMM kernel ===")
num_elements = sum([A_float.numel(), B_float.numel(), C.numel()])
def benchmark_quant(callable, a_, b_, c_, a_scale_, b_scale_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_, a_scale_, b_scale_),
        warmup_iterations=5,
        iterations=100,
    )

    # Calculate metrics
    # ----------------
    dtype = a_.element_type

    # Calculate total bytes transferred:
    # - 2 reads (A and B) + 1 write (C)
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s
    gtops = num_elements / (avg_time_us * 1000)  # GTOPS

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")
    print(f"GTOPS: {gtops:.2f}")

benchmark_quant(compiled_gemm, mA, mB, mC, mAScale, mBScale)