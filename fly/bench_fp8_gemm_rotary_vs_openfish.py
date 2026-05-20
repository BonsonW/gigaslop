"""Compare fused fp8 GEMM + rotary kernel output against the OpenFish reference.

OpenFish rotary_emb_hip (nn_kernel_hip.h) signature / semantics:
    rotary_half  = 32  (= head_dim // 2, passed as the "rotary_half" param)
    x  shape: [batch, seqlen, nheads, head_dim]  (or [M, N] after QKV projection)
    x0 = head cols [0,          rotary_half)
    x1 = head cols [rotary_half, head_dim)
    cos/sin: [seqlen, rotary_half] f32, indexed by (seq, rot)
    output:
        x0_out[k] = x0[k]*cos[k] - x1[k]*sin[k]
        x1_out[k] = x0[k]*sin[k] + x1[k]*cos[k]

This script:
  1. Builds random fp8 inputs
  2. Runs the fused kernel (compile_fp8_gemm_rotary with rotary_dim=64)
  3. Computes a PyTorch reference that exactly replicates the OpenFish formula
  4. Reports per-element absolute error stats and prints any large mismatches
"""

import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_gemm_rotary import (
    compile_fp8_gemm_rotary,
    preshuffle_b_fp8,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
)

# ── problem dims ───────────────────────────────────────────────────────────────
batch_size   = 4      # small so errors are easy to inspect
sequence_len = 16
in_features  = 512
out_features = 1536   # N = 3 * nhead * head_dim

nhead      = 8
head_dim   = 64
# rotary_dim == head_dim means OpenFish rotates all 64 cols per head:
#   rotary_half = head_dim // 2 = 32
#   x0 = cols [0..31], x1 = cols [32..63]
rotary_dim  = 64
rotary_half = rotary_dim // 2   # 32

M      = batch_size * sequence_len
K      = in_features
N      = out_features
seqlen = sequence_len

print(f"Problem: M={M} (batch={batch_size}×seq={sequence_len}), N={N}, K={K}")
print(f"Rotary:  nhead={nhead}, head_dim={head_dim}, rotary_dim={rotary_dim}, "
      f"rotary_half={rotary_half}")
print(f"sin/cos buffer: [{seqlen}, {rotary_half}]  (matches TxModel RotaryEmbeddingImpl)\n")

# ── inputs ─────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
A_f32 = torch.randn(M, K, device="cuda") * 0.1
B_f32 = torch.randn(K, N, device="cuda") * 0.1

A_fp8, scale_a = fp8_quantize_per_token(A_f32)
B_fp8, scale_b = fp8_quantize_per_channel(B_f32)
B_shuf = preshuffle_b_fp8(B_fp8)

# Canonical sin/cos matching RotaryEmbeddingImpl frequencies:
#   freq[i] = 1 / (10000^(2i/head_dim))  for i in [0, rotary_half)
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_half, device="cuda").float() / head_dim * 2))
t        = torch.arange(seqlen, device="cuda").float()
freqs    = torch.outer(t, inv_freq)   # [seqlen, rotary_half]
sin_buf  = freqs.sin().contiguous()
cos_buf  = freqs.cos().contiguous()

print(f"sin_buf shape: {sin_buf.shape}, cos_buf shape: {cos_buf.shape}")

# ── compile fused kernel ───────────────────────────────────────────────────────
print("Compiling fused kernel...")
launcher = compile_fp8_gemm_rotary(
    M=M, N=N, K=K,
    nhead=nhead, head_dim=head_dim,
    rotary_dim=rotary_dim,
)

# ── run fused kernel ───────────────────────────────────────────────────────────
C_fused = torch.zeros(M, N, dtype=torch.float16, device="cuda")
stream  = torch.cuda.current_stream()
launcher(C_fused, A_fp8, B_shuf, scale_a, scale_b, sin_buf, cos_buf, stream, M, seqlen)
torch.cuda.synchronize()

# ── OpenFish reference (PyTorch) ──────────────────────────────────────────────
# Step 1: dequantised GEMM (same fp8 inputs → same rounding)
A_dq   = A_fp8.float() * scale_a[:, None]
B_dq   = B_fp8.float() * scale_b[None, :]
C_ref  = (A_dq @ B_dq)   # float32

# Step 2: Apply rotary exactly as OpenFish rotary_emb_hip does.
#   For each row m: seq = m % seqlen
#   For Q chunk (cols 0..nhead*head_dim-1) and K chunk (cols nhead*head_dim..2*nhead*head_dim-1):
#     For each head h in [0, nhead):
#       head_start = chunk_start + h * head_dim
#       x0 = C_ref[m, head_start       : head_start + rotary_half]
#       x1 = C_ref[m, head_start+rh    : head_start + head_dim   ]
#       cos_vec = cos_buf[seq, :]   shape [rotary_half]
#       sin_vec = sin_buf[seq, :]   shape [rotary_half]
#       C_ref[m, head_start       : head_start+rh] = x0*cos - x1*sin
#       C_ref[m, head_start+rh    : head_start+hd] = x0*sin + x1*cos
rows    = torch.arange(M, device="cuda")
seq_idx = rows % seqlen
cos_row = cos_buf[seq_idx]   # [M, rotary_half]
sin_row = sin_buf[seq_idx]   # [M, rotary_half]

for chunk_start in (0, nhead * head_dim):
    for h in range(nhead):
        h0 = chunk_start + h * head_dim
        x0 = C_ref[:, h0            : h0 + rotary_half].clone()
        x1 = C_ref[:, h0+rotary_half: h0 + head_dim   ].clone()
        C_ref[:, h0            : h0 + rotary_half] = x0 * cos_row - x1 * sin_row
        C_ref[:, h0+rotary_half: h0 + head_dim   ] = x0 * sin_row + x1 * cos_row
# V cols [2*nhead*head_dim, N) — no change

C_ref_fp16 = C_ref.to(torch.float16)

# ── compare ────────────────────────────────────────────────────────────────────
print("\n=== Comparison: fused kernel vs OpenFish reference ===")
abs_err = (C_fused.float() - C_ref_fp16.float()).abs()
rel_err = abs_err / (C_ref_fp16.float().abs().clamp(min=1e-6))

print(f"  Max  abs error:  {abs_err.max().item():.6f}")
print(f"  Mean abs error:  {abs_err.mean().item():.8f}")
print(f"  Median abs err:  {abs_err.median().item():.8f}")
print(f"  Max  rel error:  {rel_err.max().item():.6f}")

# ── per-region breakdown ──────────────────────────────────────────────────────
print("\n=== Per-region error breakdown ===")
q_end  = nhead * head_dim
k_end  = 2 * nhead * head_dim
# Q rotary x0
for region_name, col_sl in [
    ("Q x0 (rotary first-half )",  slice(0,     q_end,       None)),
    ("K x0 (rotary first-half )",  slice(q_end, k_end,       None)),
    ("V   (no rotary)          ",  slice(k_end, N,           None)),
]:
    e = abs_err[:, col_sl]
    print(f"  {region_name}: max={e.max():.6f}  mean={e.mean():.8f}")

# ── check for large individual errors ────────────────────────────────────────
threshold = 0.05
bad = (abs_err > threshold).nonzero(as_tuple=False)
if bad.numel() == 0:
    print(f"\nAll elements within {threshold} abs tolerance. PASS")
else:
    print(f"\n{bad.shape[0]} elements exceed {threshold} abs tolerance:")
    for i in range(min(20, bad.shape[0])):
        r, c = bad[i, 0].item(), bad[i, 1].item()
        chunk = "Q" if c < q_end else ("K" if c < k_end else "V")
        head  = (c % (nhead * head_dim)) // head_dim
        pos   = c % head_dim
        seq   = r % seqlen
        print(f"  row={r:5d} col={c:4d} ({chunk} h={head} pos={pos:2d} seq={seq:3d})"
              f"  fused={C_fused[r,c].item():+.4f}  ref={C_ref_fp16[r,c].item():+.4f}"
              f"  err={abs_err[r,c].item():.4f}")
