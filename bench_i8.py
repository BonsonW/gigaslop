import sys
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm_i8 import TensorOpGemmI8

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

batch_size = 1024
timestep = 512
in_features = 512
out_features = 512

M = batch_size * timestep
K = in_features
N = out_features
L = 1

a_major = "k"
b_major = "k"
c_major = "n"
a_dtype = cutlass.Int8
b_dtype = cutlass.Int8
c_dtype = cutlass.Int32
acc_dtype = cutlass.Int32
use_k32 = True
atom_layout_mnk = (2, 2, 1)

# Auto-select bm (same logic as run())
if M <= 16:
    bm = 16
elif M <= 32:
    bm = 32
elif M <= 64:
    bm = 64
elif M <= 256:
    bm = 128
else:
    bm = 128

bN = 128
bK = 64
M_pad = ((M + bm - 1) // bm) * bm
N_pad = ((N + bN - 1) // bN) * bN
K_pad = ((K + bK - 1) // bK) * bK

def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_dtype = cutlass_torch.dtype(dtype)
    if dtype.signed:
        torch_tensor = torch.randint(-2, 3, shape, dtype=torch_dtype)
    else:
        torch_tensor = torch.randint(0, 5, shape, dtype=torch_dtype)
    torch_tensor = torch_tensor.permute(permute_order).cuda()
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
        .mark_compact_shape_dynamic(
            mode=(1 if not is_mode0_major else 0),
            stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
            divisibility=(128 // dtype.width),
        )
    )
    return cute_tensor, torch_tensor

mA, a_torch = create_and_permute_tensor(L, M_pad, K_pad, a_major == "m", a_dtype)
mB, b_torch = create_and_permute_tensor(L, N_pad, K_pad, b_major == "n", b_dtype)
mC, c_torch = create_and_permute_tensor(L, M_pad, N_pad, c_major == "m", c_dtype)

# Zero padding
if M_pad > M:
    a_torch[M:, :, :] = 0
if K_pad > K:
    a_torch[:, K:, :] = 0
    b_torch[:, K:, :] = 0
if N_pad > N:
    b_torch[N:, :, :] = 0

# Fill with your actual data instead of random
# a_torch[:M, :K, 0] = your_real_A
# b_torch[:N, :K, 0] = your_real_B

tensor_op_gemm = TensorOpGemmI8(
    a_dtype, b_dtype, c_dtype, acc_dtype,
    atom_layout_mnk, use_k32, bm,
)

print('=== Compiling ampere_gemm kernel ===')
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)
compiled_gemm(mA, mB, mC)

# Benchmark
# print("\n=== Benchmarking GEMM kernel ===")
# def benchmark(callable, a_, b_, c_):
#     avg_time_us = cute.testing.benchmark(
#         callable,
#         kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
#         warmup_iterations=5,
#         iterations=100,
#     )

#     # Bytes transferred: A (read) + B (read) + C (write)
#     # Use actual unpadded shapes for meaningful bandwidth numbers
#     a_bytes = M * K * (cutlass.Int8.width // 8)
#     b_bytes = N * K * (cutlass.Int8.width // 8)
#     c_bytes = M * N * (cutlass.Int32.width // 8)
#     total_bytes = a_bytes + b_bytes + c_bytes

#     # Compute ops: each output element is a dot product of length K
#     # INT8 GEMM: 2 ops per MAC (multiply + accumulate), M*N*K MACs total
#     total_ops = 2 * M * N * K

#     # Metrics
#     avg_time_s = avg_time_us * 1e-6
#     achieved_bandwidth_gbs = (total_bytes / avg_time_s) / 1e9
#     tops = (total_ops / avg_time_s) / 1e12

#     # Theoretical peaks for A100 (adjust for your GPU)
#     # A100 INT8 tensor core peak: 624 TOPS
#     # A100 memory bandwidth: 2 TB/s
#     peak_tops = 624.0
#     peak_bw_gbs = 2000.0
#     tops_efficiency = (tops / peak_tops) * 100
#     bw_efficiency = (achieved_bandwidth_gbs / peak_bw_gbs) * 100

#     print(f"Performance Metrics:")
#     print(f"  Matrix shape:       M={M}, N={N}, K={K}")
#     print(f"  Execution time:     {avg_time_us:.2f} us")
#     print(f"  Compute:            {tops:.3f} TOPS  ({tops_efficiency:.1f}% of {peak_tops} TOPS peak)")
#     print(f"  Bandwidth:          {achieved_bandwidth_gbs:.1f} GB/s  ({bw_efficiency:.1f}% of {peak_bw_gbs} GB/s peak)")
#     print(f"  Arithmetic intensity: {total_ops / total_bytes:.1f} FLOP/byte")

# benchmark(compiled_gemm, mA, mB, mC)