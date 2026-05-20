import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from rdna_fp8_dual_gemm_silu_opt import compile_fp8_dual_gemm_silu_opt, preshuffle_b_fp8
from rdna_fp8_preshuffle_gemm import fp8_quantize_per_token, fp8_quantize_per_channel

# ── problem size (matches bench_fp8_dual_gemm_silu.py) ───────────────────────
batch_size   = 128
sequence_len = 1024
in_features  = 512
out_features = 2048

M = batch_size * sequence_len
K = in_features
N = out_features

# ── inputs ────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
A_f32      = torch.randn(M, K, device="cuda") * 0.1
B_gate_f32 = torch.randn(K, N, device="cuda") * 0.1
B_up_f32   = torch.randn(K, N, device="cuda") * 0.1

A_fp8,      scale_a      = fp8_quantize_per_token(A_f32)
B_gate_fp8, scale_b_gate = fp8_quantize_per_channel(B_gate_f32)
B_up_fp8,   scale_b_up   = fp8_quantize_per_channel(B_up_f32)

B_gate_shuf = preshuffle_b_fp8(B_gate_fp8)
B_up_shuf   = preshuffle_b_fp8(B_up_fp8)

# fp16 versions for the torch baseline (dequantised)
A_f16      = (A_fp8.float()      * scale_a[:, None]).to(torch.float16)
B_gate_f16 = (B_gate_fp8.float() * scale_b_gate[None, :]).to(torch.float16)
B_up_f16   = (B_up_fp8.float()   * scale_b_up[None, :]).to(torch.float16)

C_gpu  = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

print(f"Shape: M={M} (batch={batch_size}×seq={sequence_len}), N={N}, K={K}")

# ── compile FlyDSL kernel ─────────────────────────────────────────────────────
print("\nCompiling optimised kernel...")
launcher_opt = compile_fp8_dual_gemm_silu_opt(M=M, N=N, K=K)


# ── benchmark helpers ─────────────────────────────────────────────────────────
def benchmark_kernel(launcher_, label, warmup=1, iterations=20):
    def _run():
        launcher_(C_gpu, A_fp8, B_gate_shuf, B_up_shuf,
                  scale_a, scale_b_gate, scale_b_up, stream, M)

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


def benchmark_torch(label, warmup=1, iterations=20):
    def _run():
        gate = A_f16 @ B_gate_f16
        out  = F.silu(gate) * (A_f16 @ B_up_f16)
        return out

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


def _print_stats(avg_us, label):
    total_bytes = M*K*2 + 2*K*N*2 + 2*M*N*2  # fp16 throughout for torch baseline sizing
    total_ops   = 2 * 2 * M * N * K
    avg_s       = avg_us * 1e-6
    bw_gbs      = (total_bytes / avg_s) / 1e9
    tops        = (total_ops   / avg_s) / 1e12

    peak_tops   = 383.0
    peak_bw_gbs = 640.0

    print(f"\n{label}")
    print(f"  Time:      {avg_us:.2f} us")
    print(f"  Compute:   {tops:.3f} TFLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth: {bw_gbs:.1f} GB/s  ({bw_gbs/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    print(f"  AI:        {total_ops/total_bytes:.1f} FLOP/byte")
    return avg_us


print("\n=== Benchmarking ===")
t_base = benchmark_torch("PyTorch baseline  (fp16 matmul + silu)")
t_opt  = benchmark_kernel(launcher_opt, "FlyDSL fp8 dual GEMM + silu  (interleaved WMMA, k_unroll=2)")

print(f"\nSpeedup: {t_base/t_opt:.3f}x  ({(t_base-t_opt)/t_base*100:+.1f}%)")
