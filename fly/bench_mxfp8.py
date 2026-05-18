import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_mxfp8_preshuffle_gemm import (
    compile_mxfp8_gemm,
    mxfp8_quantize_a,
    fp8_quantize_per_channel,
    preshuffle_b_fp8,
)

verify_correctness = True

# ── problem size ─────────────────────────────────────────────────────────────
batch_size   = 128
sequence_len = 1024
in_features  = 512
out_features = 4096

M = batch_size * sequence_len
K = in_features
N = out_features

# ── generate fp32 inputs and quantize ────────────────────────────────────────
torch.manual_seed(42)
A_3d  = torch.randn(batch_size, sequence_len, in_features, device="cuda") * 0.1
B_f32 = torch.randn(K, N, device="cuda") * 0.1

A_f32 = A_3d.reshape(M, K)

A_fp8, A_scale = mxfp8_quantize_a(A_f32)         # fp8 [M,K], uint8 [M, K//32]
B_fp8, scale_b = fp8_quantize_per_channel(B_f32)  # fp8 [K,N], f32 [N]

print(f"A_3d {A_3d.shape} -> A_f32 {A_f32.shape}  A_fp8 {A_fp8.shape}  A_scale {A_scale.shape}")
print(f"B_f32 {B_f32.shape}  B_fp8 {B_fp8.shape}  scale_b {scale_b.shape}")

B_shuf = preshuffle_b_fp8(B_fp8)
print(f"B_shuf {B_shuf.shape}")

C_gpu  = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

launcher = compile_mxfp8_gemm(M=M, N=N, K=K)

# ── optional correctness check ────────────────────────────────────────────────
if verify_correctness:
    print("\n=== Verifying correctness ===")
    launcher(C_gpu, A_fp8, A_scale, B_shuf, scale_b, stream, M)
    torch.cuda.synchronize()

    A_dq  = A_fp8.float().reshape(M, K // 32, 32)
    deq_a = (A_scale.int() << 23).view(torch.float32)
    A_dq  = (A_dq * deq_a.unsqueeze(-1)).reshape(M, K)
    B_dq  = B_fp8.float() * scale_b[None, :]
    C_ref = (A_dq @ B_dq).to(torch.float16)

    abs_err = (C_gpu.float() - C_ref.float()).abs()
    print(f"  Max  abs error: {abs_err.max().item():.4f}")
    print(f"  Mean abs error: {abs_err.mean().item():.6f}")

# ── benchmark ─────────────────────────────────────────────────────────────────
print("\n=== Benchmarking mxfp8 GEMM kernel ===")


def benchmark(launcher_, warmup=10, iterations=200):
    for _ in range(warmup):
        launcher_(C_gpu, A_fp8, A_scale, B_shuf, scale_b, stream, M)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    start_ev.record(stream)
    for _ in range(iterations):
        launcher_(C_gpu, A_fp8, A_scale, B_shuf, scale_b, stream, M)
    end_ev.record(stream)
    torch.cuda.synchronize()

    avg_time_us = start_ev.elapsed_time(end_ev) * 1e3 / iterations

    a_bytes     = M * K * 1           # fp8 = 1 byte
    a_sc_bytes  = M * (K // 32) * 1   # uint8 E8M0 scales
    b_bytes     = K * N * 1
    c_bytes     = M * N * 2           # fp16 = 2 bytes
    total_bytes = a_bytes + a_sc_bytes + b_bytes + c_bytes
    total_ops   = 2 * M * N * K

    avg_time_s      = avg_time_us * 1e-6
    achieved_bw_gbs = (total_bytes / avg_time_s) / 1e9
    tops            = (total_ops   / avg_time_s) / 1e12

    peak_tops   = 383.0
    peak_bw_gbs = 640.0

    print(f"Performance Metrics:")
    print(f"  Problem:              batch={batch_size}, seq={sequence_len}, in={in_features}, out={out_features}")
    print(f"  Matrix shape:         M={M} (batch*seq), N={N}, K={K}")
    print(f"  Execution time:       {avg_time_us:.2f} us")
    print(f"  Compute:              {tops:.3f} TFLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth:            {achieved_bw_gbs:.1f} GB/s  ({achieved_bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  Arithmetic intensity: {total_ops/total_bytes:.1f} FLOP/byte")


benchmark(launcher)
