import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from rdna_fp8_gemm_rotary import (
    compile_fp8_gemm_rotary,
    preshuffle_b_fp8,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
)
from rdna_fp8_preshuffle_gemm import compile_fp8_gemm

# ── problem size ──────────────────────────────────────────────────────────────
batch_size   = 128
sequence_len = 1024
in_features  = 512
out_features = 1536   # N = 3 * nhead * head_dim

nhead      = 8
head_dim   = 64
rotary_dim = 64   # full head_dim — matches OpenFish (rotary_half=32, all 64 cols rotated)
seqlen     = sequence_len

M = batch_size * sequence_len
K = in_features
N = out_features
rotary_half = rotary_dim // 2

# ── inputs ────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
A_f32 = torch.randn(M, K, device="cuda") * 0.1
B_f32 = torch.randn(K, N, device="cuda") * 0.1

A_fp8, scale_a = fp8_quantize_per_token(A_f32)
B_fp8, scale_b = fp8_quantize_per_channel(B_f32)
B_shuf = preshuffle_b_fp8(B_fp8)

# fp16 dequantised for torch baseline
A_f16 = (A_fp8.float() * scale_a[:, None]).to(torch.float16)
B_f16 = (B_fp8.float() * scale_b[None, :]).to(torch.float16)

# sin/cos buffers [seqlen, rotary_half]
theta   = torch.arange(seqlen, device="cuda").float().unsqueeze(1) * 0.01
rot     = torch.arange(rotary_half, device="cuda").float().unsqueeze(0)
sin_buf = torch.sin(theta + rot).contiguous()
cos_buf = torch.cos(theta + rot).contiguous()

C_gpu  = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

print(f"Shape: M={M} (batch={batch_size}×seq={sequence_len}), N={N}, K={K}")
print(f"Rotary: nhead={nhead}, head_dim={head_dim}, rotary_dim={rotary_dim}")

# ── compile kernels ───────────────────────────────────────────────────────────
print("\nCompiling kernels...")
launcher_fused   = compile_fp8_gemm_rotary(
    M=M, N=N, K=K,
    nhead=nhead, head_dim=head_dim,
    rotary_dim=rotary_dim,
)
launcher_gemm    = compile_fp8_gemm(M=M, N=N, K=K)

# ── correctness check ─────────────────────────────────────────────────────────
print("\n=== Verifying correctness ===")
launcher_fused(C_gpu, A_fp8, B_shuf, scale_a, scale_b, sin_buf, cos_buf, stream, M, seqlen)
torch.cuda.synchronize()

A_dq      = A_fp8.float() * scale_a[:, None]
B_dq      = B_fp8.float() * scale_b[None, :]
C_ref_f32 = (A_dq @ B_dq).clone()

rows    = torch.arange(M, device="cuda")
cos_row = cos_buf[rows % seqlen]
sin_row = sin_buf[rows % seqlen]
for chunk_start in (0, nhead * head_dim):
    for h in range(nhead):
        h0 = chunk_start + h * head_dim
        x0 = C_ref_f32[:, h0 : h0 + rotary_half].clone()
        x1 = C_ref_f32[:, h0 + rotary_half : h0 + rotary_dim].clone()
        C_ref_f32[:, h0 : h0 + rotary_half]              = x0 * cos_row - x1 * sin_row
        C_ref_f32[:, h0 + rotary_half : h0 + rotary_dim] = x0 * sin_row + x1 * cos_row

C_ref   = C_ref_f32.to(torch.float16)
abs_err = (C_gpu.float() - C_ref.float()).abs()
print(f"  Max  abs error: {abs_err.max().item():.4f}")
print(f"  Mean abs error: {abs_err.mean().item():.6f}")

# ── benchmark helpers ─────────────────────────────────────────────────────────
# Memory traffic: fp8 GEMM + rotary (fused or separate)
a_bytes     = M * K * 1          # fp8
b_bytes     = K * N * 1          # fp8
c_bytes     = M * N * 2          # fp16
sc_bytes    = seqlen * rotary_half * 4 * 2  # sin + cos (f32)
total_bytes = a_bytes + b_bytes + c_bytes + sc_bytes
total_ops   = 2 * M * N * K      # GEMM FLOPs (rotary adds negligible ~2*M*N*rotary_dim/N)
peak_tops   = 383.0
peak_bw_gbs = 640.0


def _stats(avg_us, label):
    avg_s  = avg_us * 1e-6
    bw     = (total_bytes / avg_s) / 1e9
    tops   = (total_ops   / avg_s) / 1e12
    print(f"\n{label}")
    print(f"  Time:      {avg_us:.2f} us")
    print(f"  Compute:   {tops:.3f} TFLOPS  ({tops/peak_tops*100:.1f}% of {peak_tops} TOPS peak)")
    print(f"  Bandwidth: {bw:.1f} GB/s  ({bw/peak_bw_gbs*100:.1f}% of {peak_bw_gbs} GB/s peak)")
    return avg_us


def bench(fn, label, warmup=2, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        fn()
    end.record(stream)
    torch.cuda.synchronize()
    return _stats(start.elapsed_time(end) * 1e3 / iters, label)


# ── torch reference rotary (for timing the separate approach) ─────────────────
def _torch_rotary_inplace(C):
    rows    = torch.arange(M, device="cuda")
    cos_row = cos_buf[rows % seqlen]
    sin_row = sin_buf[rows % seqlen]
    for chunk_start in (0, nhead * head_dim):
        for h in range(nhead):
            h0 = chunk_start + h * head_dim
            x0 = C[:, h0 : h0 + rotary_half].clone()
            x1 = C[:, h0 + rotary_half : h0 + rotary_dim].clone()
            C[:, h0 : h0 + rotary_half]              = x0 * cos_row - x1 * sin_row
            C[:, h0 + rotary_half : h0 + rotary_dim] = x0 * sin_row + x1 * cos_row


# ── run benchmarks ────────────────────────────────────────────────────────────
print("\n=== Benchmarking ===")

t_torch = bench(
    lambda: _torch_rotary_inplace(A_f16 @ B_f16),
    "PyTorch fp16 GEMM + rotary (separate)",
)

t_unfused = bench(
    lambda: (
        launcher_gemm(C_gpu, A_fp8, B_shuf, scale_a, scale_b, stream, M),
        _torch_rotary_inplace(C_gpu),
    ),
    "FlyDSL fp8 GEMM + PyTorch rotary (separate)",
)

t_fused = bench(
    lambda: launcher_fused(C_gpu, A_fp8, B_shuf, scale_a, scale_b, sin_buf, cos_buf, stream, M, seqlen),
    "FlyDSL fp8 GEMM + rotary (fused)",
)

print("\n=== Summary ===")
print(f"  PyTorch fp16 GEMM + separate rotary:  {t_torch:.2f} us  (1.000x)")
print(f"  FlyDSL fp8  GEMM + separate rotary:   {t_unfused:.2f} us  ({t_torch/t_unfused:.3f}x)")
print(f"  FlyDSL fp8  GEMM + fused rotary:      {t_fused:.2f} us  ({t_torch/t_fused:.3f}x)")
print(f"\n  Fusion gain over unfused fp8:          {t_unfused/t_fused:.3f}x  ({(t_unfused-t_fused)/t_unfused*100:+.1f}%)")
