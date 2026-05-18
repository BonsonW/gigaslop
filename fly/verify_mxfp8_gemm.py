"""Verify rdna_mxfp8_preshuffle_gemm against a naive PyTorch reference.

Run:
    python fly/verify_mxfp8_gemm.py
"""
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fly.rdna_mxfp8_preshuffle_gemm import (
    compile_mxfp8_gemm,
    mxfp8_quantize_a,
    fp8_quantize_per_channel,
    preshuffle_b_fp8,
)

CONFIGS = [
    (32,  8192, 6144),
    (256, 2048,  512),
]


def naive_mxfp8_gemm_ref(A_fp8, A_scale_e8m0, B_kn_fp8, scale_b, block_size: int = 32):
    """Reference: dequantize A with E8M0 scales, dequantize B with per-channel scale."""
    M, K = A_fp8.shape
    K_blocks = K // block_size
    A_f32 = A_fp8.float().reshape(M, K_blocks, block_size)
    deq_a = (A_scale_e8m0.int() << 23).view(torch.float32)     # [M, K//32]
    A_dq  = (A_f32 * deq_a.unsqueeze(-1)).reshape(M, K)
    B_dq  = B_kn_fp8.float() * scale_b[None, :]
    return (A_dq @ B_dq).to(torch.float16)


def run_one(M, N, K):
    print(f"\n[M={M}, N={N}, K={K}]")
    torch.manual_seed(42)

    A_f32 = torch.randn(M, K, device="cuda") * 0.1
    B_f32 = torch.randn(K, N, device="cuda") * 0.1

    A_fp8, A_scale = mxfp8_quantize_a(A_f32)            # fp8 [M,K], uint8 [M, K//32]
    B_fp8, scale_b = fp8_quantize_per_channel(B_f32)    # fp8 [K,N], f32 [N]
    B_shuf = preshuffle_b_fp8(B_fp8)                    # [N//16, K//16, 2, 16, 8]

    C_gpu = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    stream = torch.cuda.current_stream()
    launcher = compile_mxfp8_gemm(M=M, N=N, K=K)
    launcher(C_gpu, A_fp8, A_scale, B_shuf, scale_b, stream, M)
    torch.cuda.synchronize()

    C_ref = naive_mxfp8_gemm_ref(A_fp8, A_scale, B_fp8, scale_b)

    abs_err  = (C_gpu.float() - C_ref.float()).abs()
    max_abs  = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    max_rel  = (abs_err / (C_ref.float().abs() + 1e-6)).max().item()

    print(f"  Max  absolute error : {max_abs:.4f}")
    print(f"  Mean absolute error : {mean_abs:.6f}")
    print(f"  Max  relative error : {max_rel:.4f}")

    threshold = 2.0
    passed = max_abs < threshold
    print(f"  Threshold {threshold}: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = [run_one(M, N, K) for M, N, K in CONFIGS]
    sys.exit(0 if all(results) else 1)
