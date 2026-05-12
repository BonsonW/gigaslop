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

# ── problem size ────────────────────────────────────────────────────────────
batch_size   = 512
timestep     = 1024
in_features  = 512
out_features = 512

M = batch_size * timestep
K = in_features
N = out_features
L = 1

# ── your torch tensors (whatever you have) ──────────────────────────────────
A_float = torch.randn(M, K, dtype=torch.float32)
B_float = torch.randn(N, K, dtype=torch.float32)

# quantise to int8 (replace with your real quantisation logic)
A_scale = A_float.abs().max(dim=1, keepdim=True).values / 127.0   # (M, 1)
B_scale = B_float.abs().max(dim=1, keepdim=True).values / 127.0   # (N, 1)
A_int8  = (A_float / A_scale).round().clamp(-128, 127).to(torch.int8)
B_int8  = (B_float / B_scale).round().clamp(-128, 127).to(torch.int8)

print(f"A_float  {A_float.shape}  B_float  {B_float.shape}")
print(f"A_int8   {A_int8.shape}   B_int8   {B_int8.shape}")
print(f"A_scale  {A_scale.shape}  B_scale  {B_scale.shape}")

# ── kernel config ────────────────────────────────────────────────────────────
a_dtype       = cutlass.Int8
b_dtype       = cutlass.Int8
c_dtype       = cutlass.Float16
acc_dtype     = cutlass.Int32
use_k32       = True
atom_layout_mnk = (2, 2, 1)

if   M <= 16:  bm = 16
elif M <= 32:  bm = 32
elif M <= 64:  bm = 64
else:          bm = 128

bN, bK = 128, 64
M_pad = ((M + bm - 1) // bm) * bm
N_pad = ((N + bN - 1) // bN) * bN
K_pad = ((K + bK - 1) // bK) * bK

# ── pack torch tensors into the layout CuteDSL expects ──────────────────────
# Rule: create (L, mode0, mode1) → permute(1,2,0) → (mode0, mode1, L)
# After permute + contiguous: strides = (mode1, 1, mode0*mode1)
# leading_dim=1 is always the contiguous dim (K for A/B, N for C)

def pack_input(t_2d, rows, cols, row_pad, col_pad, torch_dtype):
    """
    t_2d : (rows, cols) torch tensor
    returns (row_pad, col_pad, L) cute-compatible tensor + raw torch tensor
    """
    buf = torch.zeros(L, row_pad, col_pad, dtype=torch_dtype, device='cuda')
    buf[0, :rows, :cols] = t_2d.to(torch_dtype).cuda()
    t = buf.permute(1, 2, 0).contiguous()   # (row_pad, col_pad, L)
    return t

def pack_output(rows, cols, torch_dtype):
    buf = torch.zeros(L, rows, cols, dtype=torch_dtype, device='cuda')
    return buf.permute(1, 2, 0).contiguous()   # (rows, cols, L)

def wrap_AB(t, dtype):
    """K-major input: leading_dim=1 (K is contiguous)"""
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(
            mode=0,
            stride_order=(2, 0, 1),
            divisibility=(128 // dtype.width),
        )
    )

def wrap_C(t, dtype):
    """N-major output: leading_dim=1 (N is contiguous)"""
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(
            mode=0,
            stride_order=(2, 0, 1),
            divisibility=(128 // dtype.width),
        )
    )

a_torch = pack_input(A_int8, M, K, M_pad, K_pad, torch.int8)
b_torch = pack_input(B_int8, N, K, N_pad, K_pad, torch.int8)
c_torch = pack_output(M_pad, N_pad, cutlass_torch.dtype(c_dtype))

mA = wrap_AB(a_torch, cutlass.Int8)
mB = wrap_AB(b_torch, cutlass.Int8)
mC = wrap_C (c_torch, c_dtype)

# scales: (M_pad, L) and (N_pad, L), float32
scale_a_torch = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda')
scale_b_torch = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda')
scale_a_torch[:M, 0] = A_scale[:, 0].cuda()
scale_b_torch[:N, 0] = B_scale[:, 0].cuda()
mScaleA = from_dlpack(scale_a_torch.contiguous(), assumed_align=4)
mScaleB = from_dlpack(scale_b_torch.contiguous(), assumed_align=4)

print(f"a_torch strides: {a_torch.stride()}")   # (K_pad, 1, M_pad*K_pad)
print(f"b_torch strides: {b_torch.stride()}")   # (K_pad, 1, N_pad*K_pad)
print(f"c_torch strides: {c_torch.stride()}")   # (N_pad, 1, M_pad*N_pad)

# ── compile ──────────────────────────────────────────────────────────────────
tensor_op_gemm = TensorOpGemmI8(
    a_dtype, b_dtype, c_dtype, acc_dtype,
    atom_layout_mnk, use_k32, bm,
)

print("\n=== Compiling ===")
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC, mScaleA, mScaleB)
compiled_gemm(mA, mB, mC, mScaleA, mScaleB)

# result lives in c_torch[:M, :N, 0]
result = c_torch[:M, :N, 0]
print(f"Output shape: {result.shape}, dtype: {result.dtype}")

# ── optional: verify against torch reference ─────────────────────────────────
with torch.inference_mode():
    ref = (A_float.cuda() @ B_float.cuda().T).to(torch.float16)
    kernel_out = result.cpu().float()
    ref_out    = ref.cpu().float()
    print(f"Max abs error vs fp32 ref: {(kernel_out - ref_out).abs().max():.4f}")

# ── benchmark ────────────────────────────────────────────────────────────────
def benchmark(callable, a_, b_, c_, sa_, sb_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_, sa_, sb_),
        warmup_iterations=5,
        iterations=100,
    )

    a_bytes      = M * K * (cutlass.Int8.width   // 8)
    b_bytes      = N * K * (cutlass.Int8.width   // 8)
    c_bytes      = M * N * (cutlass_torch.dtype(c_dtype).itemsize)
    total_bytes  = a_bytes + b_bytes + c_bytes
    total_ops    = 2 * M * N * K

    avg_time_s   = avg_time_us * 1e-6
    bw_gbs       = (total_bytes / avg_time_s) / 1e9
    tops         = (total_ops   / avg_time_s) / 1e12

    peak_tops    = 624.0    # A100 INT8
    peak_bw_gbs  = 2000.0   # A100 HBM

    print(f"\nPerformance Metrics:")
    print(f"  Shape:              M={M} N={N} K={K}")
    print(f"  Time:               {avg_time_us:.2f} us")
    print(f"  Compute:            {tops:.3f} TOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth:          {bw_gbs:.1f} GB/s  ({bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  Arithmetic intensity: {total_ops/total_bytes:.1f} FLOP/byte")

print("\n=== Benchmarking ===")
benchmark(compiled_gemm, mA, mB, mC, mScaleA, mScaleB)