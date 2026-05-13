"""Verify rdna_fp8_preshuffle_gemm against a naive PyTorch reference.

Uses the FlyDSL JIT directly (no C++ required).  The naive reference
computes the scaled matmul in full f32 precision, so errors come purely
from fp8 quantisation noise, not from the kernel itself.

Run:
    python verify_fp8_gemm.py
"""
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fly.rdna_fp8_preshuffle_gemm import (
    compile_fp8_gemm,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
    preshuffle_b_fp8,
)

M, N, K = 32, 8192, 6144


def naive_fp8_gemm_ref(A_fp8, B_kn_fp8, scale_a, scale_b):
    """Reference matmul in f32 then cast to bf16.

    C[m,n] = sum_k (A[m,k] * scale_a[m] * B[k,n] * scale_b[n])

    A_fp8  : [M, K]  torch.float8_e4m3fn on CUDA
    B_kn_fp8: [K, N] torch.float8_e4m3fn on CUDA  (original, NOT preshuffled)
    scale_a : [M]    torch.float32
    scale_b : [N]    torch.float32
    """
    A_f32 = A_fp8.float()
    B_f32 = B_kn_fp8.float()
    C_f32 = (A_f32 * scale_a[:, None]) @ (B_f32 * scale_b[None, :])
    return C_f32.to(torch.bfloat16)


def run():
    torch.manual_seed(42)
    print(f"Running fp8 GEMM verification: M={M}, N={N}, K={K}")

    # ── Generate random fp32 inputs and quantize to fp8 ──────────────────────
    A_f32 = torch.randn(M, K, device="cuda") * 0.1
    B_f32 = torch.randn(K, N, device="cuda") * 0.1

    A_fp8, scale_a = fp8_quantize_per_token(A_f32)       # [M,K], [M]
    B_fp8, scale_b = fp8_quantize_per_channel(B_f32)     # [K,N], [N]

    # ── Preshuffle B for the kernel ───────────────────────────────────────────
    B_shuf = preshuffle_b_fp8(B_fp8)   # [N//16, K//16, 2, 16, 8]

    # ── Run FlyDSL kernel ─────────────────────────────────────────────────────
    C_gpu = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()
    launcher = compile_fp8_gemm(M=M, N=N, K=K)
    launcher(C_gpu, A_fp8, B_shuf, scale_a, scale_b, stream)
    torch.cuda.synchronize()

    # ── Naive reference ───────────────────────────────────────────────────────
    C_ref = naive_fp8_gemm_ref(A_fp8, B_fp8, scale_a, scale_b)

    # ── Compare ───────────────────────────────────────────────────────────────
    C_gpu_f32 = C_gpu.float()
    C_ref_f32 = C_ref.float()

    abs_err = (C_gpu_f32 - C_ref_f32).abs()
    max_abs  = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    # Relative error against reference magnitude
    max_rel  = (abs_err / (C_ref_f32.abs() + 1e-6)).max().item()

    print(f"  Max  absolute error : {max_abs:.4f}")
    print(f"  Mean absolute error : {mean_abs:.6f}")
    print(f"  Max  relative error : {max_rel:.4f}")

    # fp8 quantisation truncates to ~3 mantissa bits.  Over K=6144 terms the
    # worst-case accumulation error is bounded by ~K * fp8_ulp ≈ 6144 * 0.03
    # but in practice random inputs cancel and the error is much smaller.
    # A threshold of 2.0 (in bf16 units) is conservative.
    threshold = 2.0
    passed = max_abs < threshold
    print(f"  Threshold {threshold}: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
