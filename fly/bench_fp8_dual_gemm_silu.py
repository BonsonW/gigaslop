import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_dual_gemm_silu import (
    compile_fp8_dual_gemm_silu,
    preshuffle_b_fp8,
)
from rdna_fp8_preshuffle_gemm import (
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
)

verify_correctness = True

# ── problem size ─────────────────────────────────────────────────────────────
batch_size   = 128
sequence_len = 1024
in_features  = 4096   # K: model hidden dim
out_features = 2048   # N: inter dim (each of gate and up projections)

M = batch_size * sequence_len
K = in_features
N = out_features

# ── generate fp32 inputs and quantize to fp8 ─────────────────────────────────
torch.manual_seed(42)
A_f32      = torch.randn(M, K, device="cuda") * 0.1
B_gate_f32 = torch.randn(K, N, device="cuda") * 0.1
B_up_f32   = torch.randn(K, N, device="cuda") * 0.1

A_fp8,      scale_a      = fp8_quantize_per_token(A_f32)
B_gate_fp8, scale_b_gate = fp8_quantize_per_channel(B_gate_f32)
B_up_fp8,   scale_b_up   = fp8_quantize_per_channel(B_up_f32)

print(f"A_f32 {A_f32.shape}  A_fp8 {A_fp8.shape}  scale_a {scale_a.shape}")
print(f"B_gate_f32 {B_gate_f32.shape}  B_gate_fp8 {B_gate_fp8.shape}  scale_b_gate {scale_b_gate.shape}")
print(f"B_up_f32   {B_up_f32.shape  }  B_up_fp8   {B_up_fp8.shape  }  scale_b_up   {scale_b_up.shape  }")

# ── preshuffle B matrices ─────────────────────────────────────────────────────
B_gate_shuf = preshuffle_b_fp8(B_gate_fp8)
B_up_shuf   = preshuffle_b_fp8(B_up_fp8)
print(f"B_gate_shuf {B_gate_shuf.shape}  B_up_shuf {B_up_shuf.shape}")

# ── allocate output ───────────────────────────────────────────────────────────
C_gpu  = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

# ── compile ───────────────────────────────────────────────────────────────────
launcher = compile_fp8_dual_gemm_silu(M=M, N=N, K=K)

# ── optional correctness check ────────────────────────────────────────────────
if verify_correctness:
    print("\n=== Verifying correctness ===")
    launcher(C_gpu, A_fp8, B_gate_shuf, B_up_shuf,
             scale_a, scale_b_gate, scale_b_up, stream, M)
    torch.cuda.synchronize()

    A_dq      = A_fp8.float()      * scale_a[:, None]
    B_gate_dq = B_gate_fp8.float() * scale_b_gate[None, :]
    B_up_dq   = B_up_fp8.float()   * scale_b_up[None, :]

    gate = A_dq @ B_gate_dq
    up   = A_dq @ B_up_dq
    C_ref = (gate * torch.sigmoid(gate) * up).to(torch.float16)

    abs_err = (C_gpu.float() - C_ref.float()).abs()
    print(f"  Max  abs error: {abs_err.max().item():.4f}")
    print(f"  Mean abs error: {abs_err.mean().item():.6f}")

# ── benchmark ─────────────────────────────────────────────────────────────────
print("\n=== Benchmarking fp8 dual GEMM + silu_mul kernel ===")

def benchmark(launcher_, warmup=10, iterations=200):
    for _ in range(warmup):
        launcher_(C_gpu, A_fp8, B_gate_shuf, B_up_shuf,
                  scale_a, scale_b_gate, scale_b_up, stream, M)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    start_ev.record(stream)
    for _ in range(iterations):
        launcher_(C_gpu, A_fp8, B_gate_shuf, B_up_shuf,
                  scale_a, scale_b_gate, scale_b_up, stream, M)
    end_ev.record(stream)
    torch.cuda.synchronize()

    avg_time_us = start_ev.elapsed_time(end_ev) * 1e3 / iterations

    a_bytes     = M * K * 1          # fp8 = 1 byte
    b_gate_bytes = K * N * 1
    b_up_bytes   = K * N * 1
    c_bytes     = M * N * 2          # fp16 = 2 bytes
    total_bytes = a_bytes + b_gate_bytes + b_up_bytes + c_bytes
    total_ops   = 2 * 2 * M * N * K  # two GEMMs

    avg_time_s      = avg_time_us * 1e-6
    achieved_bw_gbs = (total_bytes / avg_time_s) / 1e9
    tops            = (total_ops   / avg_time_s) / 1e12

    # RDNA4 AI PRO R9700 peak numbers
    peak_tops   = 383.0    # FP8 matrix (E4M3/E5M2) peak (TFLOPS)
    peak_bw_gbs = 640.0    # GDDR6 peak bandwidth (GB/s)

    print(f"Performance Metrics:")
    print(f"  Problem:              batch={batch_size}, seq={sequence_len}, in={in_features}, out={out_features}")
    print(f"  Matrix shape:         M={M} (batch*seq), N={N}, K={K}")
    print(f"  Execution time:       {avg_time_us:.2f} us")
    print(f"  Compute:              {tops:.3f} TFLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth:            {achieved_bw_gbs:.1f} GB/s  ({achieved_bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  Arithmetic intensity: {total_ops/total_bytes:.1f} FLOP/byte")

benchmark(launcher)
