import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from rdna_fp8_preshuffle_gemm import (
    compile_fp8_gemm,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
    preshuffle_b_fp8,
)

# ── problem size ─────────────────────────────────────────────────────────────
batch_size   = 128
sequence_len = 1024
in_features  = 512
out_features = 4096

M = batch_size * sequence_len
K = in_features
N = out_features

# ── generate fp32 inputs and quantize to fp8 ─────────────────────────────────
torch.manual_seed(42)
A_3d  = torch.randn(batch_size, sequence_len, in_features, device="cuda") * 0.1
B_f32 = torch.randn(K, N, device="cuda") * 0.1

A_f32 = A_3d.reshape(M, K)

A_fp8, scale_a = fp8_quantize_per_token(A_f32)
B_fp8, scale_b = fp8_quantize_per_channel(B_f32)

B_shuf = preshuffle_b_fp8(B_fp8)

# fp16 dequantised versions for the torch baseline
A_f16 = (A_fp8.float() * scale_a[:, None]).to(torch.float16)
B_f16 = (B_fp8.float() * scale_b[None, :]).to(torch.float16)

C_gpu  = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

print(f"Shape: M={M} (batch={batch_size}×seq={sequence_len}), N={N}, K={K}")

# ── compile variants ──────────────────────────────────────────────────────────
print("\nCompiling kernels...")
configs = [
    ("baseline  (tile_m=32, tile_n=256, k_unroll=1)", dict(tile_m=32, tile_n=256, k_unroll=1)),
    ("k_unroll=2 (tile_m=32, tile_n=256, k_unroll=2)", dict(tile_m=32, tile_n=256, k_unroll=2)),
    ("tile_m=64  (tile_m=64, tile_n=256, k_unroll=1)", dict(tile_m=64, tile_n=256, k_unroll=1)),
    ("both       (tile_m=64, tile_n=256, k_unroll=2)", dict(tile_m=64, tile_n=256, k_unroll=2)),
]
launchers = [(label, compile_fp8_gemm(M=M, N=N, K=K, **kw)) for label, kw in configs]

# ── correctness check on baseline ────────────────────────────────────────────
print("\n=== Verifying correctness ===")
_, launcher_base = launchers[0]
launcher_base(C_gpu, A_fp8, B_shuf, scale_a, scale_b, stream, M)
torch.cuda.synchronize()

A_f32_dq = A_fp8.float() * scale_a[:, None]
B_f32_dq = B_fp8.float() * scale_b[None, :]
C_ref = (A_f32_dq @ B_f32_dq).to(torch.float16)
abs_err = (C_gpu.float() - C_ref.float()).abs()
print(f"  Max  abs error: {abs_err.max().item():.4f}")
print(f"  Mean abs error: {abs_err.mean().item():.6f}")

# ── benchmark helpers ─────────────────────────────────────────────────────────
a_bytes     = M * K * 1
b_bytes     = K * N * 1
c_bytes     = M * N * 2
total_bytes = a_bytes + b_bytes + c_bytes
total_ops   = 2 * M * N * K
peak_tops   = 383.0
peak_bw_gbs = 640.0


def _print_stats(avg_us, label):
    avg_s    = avg_us * 1e-6
    bw_gbs   = (total_bytes / avg_s) / 1e9
    tops     = (total_ops   / avg_s) / 1e12
    print(f"\n{label}")
    print(f"  Time:      {avg_us:.2f} us")
    print(f"  Compute:   {tops:.3f} TFLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth: {bw_gbs:.1f} GB/s  ({bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  AI:        {total_ops/total_bytes:.1f} FLOP/byte")
    return avg_us


def benchmark_kernel(launcher_, label, warmup=2, iterations=20):
    def _run():
        launcher_(C_gpu, A_fp8, B_shuf, scale_a, scale_b, stream, M)

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record(stream)
    for _ in range(iterations):
        _run()
    end_ev.record(stream)
    torch.cuda.synchronize()

    return _print_stats(start_ev.elapsed_time(end_ev) * 1e3 / iterations, label)


def benchmark_torch(label, warmup=2, iterations=20):
    def _run():
        return A_f16 @ B_f16

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iterations):
        _run()
    end_ev.record()
    torch.cuda.synchronize()

    return _print_stats(start_ev.elapsed_time(end_ev) * 1e3 / iterations, label)


# ── run benchmarks ────────────────────────────────────────────────────────────
print("\n=== Benchmarking ===")
t_torch = benchmark_torch("PyTorch baseline  (fp16 matmul)")
times = [(label, benchmark_kernel(launcher_, f"FlyDSL fp8  {label}"))
         for label, launcher_ in launchers]

print("\n=== Summary ===")
print(f"  PyTorch fp16:  {t_torch:.2f} us  (1.000x)")
for label, t in times:
    print(f"  {label}:  {t:.2f} us  ({t_torch/t:.3f}x)")
