import sys, os
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(__file__))
from ampere_dual_gemm_i8_silu import (
    TensorOpDualGemmI8Silu, dual_gemm_silu_ref, create_and_permute_tensor,
)

verify_correctness = True

def quantize_tensor(t, dim=-1):
    quant_max = 127
    fp_range = t.abs().amax(dim=dim).clamp_min(1e-8)
    quant_scale = quant_max / fp_range
    t_int8 = (t * quant_scale.unsqueeze(dim)).round().clamp(-quant_max, quant_max).to(torch.int8)
    return t_int8, quant_scale.to(torch.float32).reciprocal()

# ── problem size (matches fly/bench_fp8_dual_gemm_silu.py) ────────────────────
batch_size, sequence_len = 128, 1024
in_features, out_features = 4096, 2048   # K, N
M = batch_size * sequence_len
K = in_features
N = out_features
L = 1

torch.manual_seed(42)
A_f32      = torch.randn(M, K) * 0.1
B_gate_f32 = torch.randn(N, K) * 0.1   # (N,K) weight, row = output channel
B_up_f32   = torch.randn(N, K) * 0.1

A_int8,      scale_a      = quantize_tensor(A_f32, dim=-1)       # (M,)
B_gate_int8, scale_b_gate = quantize_tensor(B_gate_f32, dim=-1)  # (N,)
B_up_int8,   scale_b_up   = quantize_tensor(B_up_f32, dim=-1)    # (N,)

# ── kernel config ─────────────────────────────────────────────────────────────
a_dtype = b_dtype = cutlass.Int8
c_dtype = cutlass.Float16
acc_dtype = cutlass.Int32
atom_layout_mnk = (2, 2, 1)
num_stages = 3
use_k32 = True
bm, bN = 128, 64   # bN small: dual accumulator (gate+up) doubles register pressure
bK = 64

M_pad = ((M + bm - 1) // bm) * bm
N_pad = ((N + bN - 1) // bN) * bN
K_pad = ((K + bK - 1) // bK) * bK

# ── tensors: A (M,K) K-major, B (N,K) K-major, C (M,N) N-major ────────────────
mA, a_torch = create_and_permute_tensor(L, M_pad, K_pad, False, a_dtype)
mBg, bg_torch = create_and_permute_tensor(L, N_pad, K_pad, False, b_dtype)
mBu, bu_torch = create_and_permute_tensor(L, N_pad, K_pad, False, b_dtype)
mC, c_torch = create_and_permute_tensor(L, M_pad, N_pad, False, c_dtype)

a_torch[:M, :K, 0] = A_int8.cuda()
bg_torch[:N, :K, 0] = B_gate_int8.cuda()
bu_torch[:N, :K, 0] = B_up_int8.cuda()
for t, r, c in [(a_torch, M, K), (bg_torch, N, K), (bu_torch, N, K)]:
    if t.shape[0] > r: t[r:, :, :] = 0
    if t.shape[1] > c: t[:, c:, :] = 0

scale_a_t  = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); scale_a_t[:M, 0] = scale_a.cuda()
scale_bg_t = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scale_bg_t[:N, 0] = scale_b_gate.cuda()
scale_bu_t = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scale_bu_t[:N, 0] = scale_b_up.cuda()
mScaleA  = from_dlpack(scale_a_t, assumed_align=16)
mScaleBg = from_dlpack(scale_bg_t, assumed_align=16)
mScaleBu = from_dlpack(scale_bu_t, assumed_align=16)

# ── compile ───────────────────────────────────────────────────────────────────
gemm = TensorOpDualGemmI8Silu(a_dtype, b_dtype, c_dtype, acc_dtype,
                              atom_layout_mnk, use_k32, bm, bn=bN, num_stages=num_stages)
print(f"M={M} K={K} N={N}  tile={bm}x{bN}x{bK}  atom={atom_layout_mnk}")
print("=== Compiling dual GEMM + silu ===")
compiled = cute.compile(gemm, mA, mBg, mBu, mC, mScaleA, mScaleBg, mScaleBu)

# ── correctness ────────────────────────────────────────────────────────────────
if verify_correctness:
    print("=== Verifying correctness ===")
    compiled(mA, mBg, mBu, mC, mScaleA, mScaleBg, mScaleBu)
    torch.cuda.synchronize()
    with torch.inference_mode():
        ref = dual_gemm_silu_ref(
            A_int8.cuda(), B_gate_int8.cuda(), B_up_int8.cuda(),
            scale_a.cuda(), scale_b_gate.cuda(), scale_b_up.cuda(),
        ).to(torch.float16)
        out = c_torch[:M, :N, 0]
        err = (out.float() - ref.float()).abs()
        print(f"  Max abs error:  {err.max().item():.4f}")
        print(f"  Mean abs error: {err.mean().item():.6f}")

# ── benchmark ─────────────────────────────────────────────────────────────────
print("=== Benchmarking ===")
t = cute.testing.benchmark(
    compiled,
    kernel_arguments=cute.testing.JitArguments(mA, mBg, mBu, mC, mScaleA, mScaleBg, mScaleBu),
    warmup_iterations=10, iterations=100,
)
total_ops = 2 * (2 * M * N * K)   # two GEMMs
tops = total_ops / (t * 1e-6) / 1e12
print(f"  Time: {t:.2f} us   {tops:.1f} TOPS  ({tops/624*100:.1f}% of 624 peak)")
