import sys
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm_i8_quant import TensorOpGemmI8

verify_correctness = False

def quantize_tensor(t, dim=-1):
    """Quantize float tensor to int8 with per-channel scaling.

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

# ── problem size ─────────────────────────────────────────────────────────────
batch_size   = 512
timestep     = 1024
in_features  = 512
out_features = 4096

M = batch_size * timestep
K = in_features
N = out_features
L = 1

# ── quantize float inputs ────────────────────────────────────────────────────
# A: (M, K) quantized per-row  (dim=-1 → scale per row, shape (M,))
# B: (N, K) quantized per-row  (dim=-1 → scale per output channel, shape (N,))
A_float = torch.randn(M, K, dtype=torch.float32)
B_float = torch.randn(N, K, dtype=torch.float32)

A_int8, A_dequant_scale = quantize_tensor(A_float, dim=-1)  # scale shape: (M,)
B_int8, B_dequant_scale = quantize_tensor(B_float, dim=-1)  # scale shape: (N,)

print(f"A_float {A_float.shape}  A_int8 {A_int8.shape}  A_scale {A_dequant_scale.shape}")
print(f"B_float {B_float.shape}  B_int8 {B_int8.shape}  B_scale {B_dequant_scale.shape}")

# ── kernel config ─────────────────────────────────────────────────────────────
a_major       = "k"
b_major       = "k"
c_major       = "n"
a_dtype       = cutlass.Int8
b_dtype       = cutlass.Int8
c_dtype       = cutlass.Float16   # dequant output → fp16
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

# ── tensor layout helpers ─────────────────────────────────────────────────────
def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_dtype = cutlass_torch.dtype(dtype)
    torch_tensor = torch.randint(-2, 3, shape, dtype=torch_dtype)
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

# ── allocate and fill tensors ─────────────────────────────────────────────────
mA, a_torch = create_and_permute_tensor(L, M_pad, K_pad, a_major == "m", a_dtype)
mB, b_torch = create_and_permute_tensor(L, N_pad, K_pad, b_major == "n", b_dtype)
mC, c_torch = create_and_permute_tensor(L, M_pad, N_pad, c_major == "m", c_dtype)

# Copy quantized data into the padded CuteDSL tensors
# a_torch / b_torch shape after permute: (mode0, mode1, L) = (M_pad, K_pad, 1)
a_torch[:M, :K, 0] = A_int8.cuda()
b_torch[:N, :K, 0] = B_int8.cuda()

# Zero padding regions
if M_pad > M:
    a_torch[M:, :, :] = 0
if K_pad > K:
    a_torch[:, K:, :] = 0
    b_torch[:, K:, :] = 0
if N_pad > N:
    b_torch[N:, :, :] = 0

# ── scale tensors: shape (rows, L), contiguous, float32 ──────────────────────
# Kernel expects mScaleA (M_pad, L) and mScaleB (N_pad, L)
scale_a_torch = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda')
scale_b_torch = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda')
scale_a_torch[:M, 0] = A_dequant_scale.cuda()
scale_b_torch[:N, 0] = B_dequant_scale.cuda()

mScaleA = from_dlpack(scale_a_torch.contiguous(), assumed_align=16)
mScaleB = from_dlpack(scale_b_torch.contiguous(), assumed_align=16)

# ── compile ───────────────────────────────────────────────────────────────────
tensor_op_gemm = TensorOpGemmI8(
    a_dtype, b_dtype, c_dtype, acc_dtype,
    atom_layout_mnk, use_k32, bm,
)

print('\n=== Compiling ampere_gemm kernel ===')
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC, mScaleA, mScaleB)

# ── optional correctness check ────────────────────────────────────────────────
if verify_correctness:
    print('\n=== Verifying correctness ===')
    compiled_gemm(mA, mB, mC, mScaleA, mScaleB)

    with torch.inference_mode():
        # Reference: dequantized fp32 matmul
        # A_dequant[m, k] = A_int8[m, k] * A_scale[m]
        # B_dequant[n, k] = B_int8[n, k] * B_scale[n]
        # C_ref[m, n]     = sum_k A_dequant[m,k] * B_dequant[n,k]
        #                 = (A_int8 @ B_int8.T)[m,n] * A_scale[m] * B_scale[n]
        A_dq = A_int8.float() * A_dequant_scale.unsqueeze(-1)   # (M, K)
        B_dq = B_int8.float() * B_dequant_scale.unsqueeze(-1)   # (N, K)
        ref  = (A_dq @ B_dq.T).to(torch.float16).cuda()         # (M, N)

        kernel_out = c_torch[:M, :N, 0]                          # (M, N) fp16
        max_err    = (kernel_out - ref).abs().max().item()
        mean_err   = (kernel_out - ref).abs().mean().item()
        print(f"  Max  abs error vs fp32 ref: {max_err:.4f}")
        print(f"  Mean abs error vs fp32 ref: {mean_err:.4f}")
        
# ── benchmark ─────────────────────────────────────────────────────────────────
print("\n=== Benchmarking GEMM kernel ===")
def benchmark(callable, a_, b_, c_, sa_, sb_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_, sa_, sb_),
        warmup_iterations=5,
        iterations=100,
    )

    a_bytes     = M * K * (cutlass.Int8.width // 8)
    b_bytes     = N * K * (cutlass.Int8.width // 8)
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
    print(f"  Compute:              {tops:.3f} TOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth:            {achieved_bw_gbs:.1f} GB/s  ({achieved_bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  Arithmetic intensity: {total_ops/total_bytes:.1f} FLOP/byte")

benchmark(compiled_gemm, mA, mB, mC, mScaleA, mScaleB)
# compiled_gemm(mA, mB, mC, mScaleA, mScaleB)