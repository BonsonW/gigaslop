"""Compare the fused INT8 GEMM + rotary kernel against the OpenFish reference.

CUDA INT8 analogue of fly/bench_fp8_gemm_rotary_vs_openfish.py.

OpenFish rotary_emb_hip (nn_kernel_hip.h) semantics:
    rotary_half = head_dim // 2
    x0 = head cols [0,           rotary_half)
    x1 = head cols [rotary_half, head_dim)
    out0[k] = x0[k]*cos[k] - x1[k]*sin[k]
    out1[k] = x0[k]*sin[k] + x1[k]*cos[k]
    cos/sin: [seqlen, rotary_half] f32, indexed by (seq = row % seqlen, rot)
    Applied to Q and K chunks; V written unchanged.

This script:
  1. Builds random INT8 inputs (per-token A, per-channel B) — same values the kernel dequantizes.
  2. Runs the fused kernel (TensorOpGemmI8Rotary).
  3. Computes a PyTorch reference that exactly replicates the OpenFish formula, with
     canonical RoPE frequencies (matching TxModel RotaryEmbeddingImpl).
  4. Reports per-element / per-region abs+rel error and prints any large mismatches.

Exit code 0 on PASS (all within tolerance), 1 otherwise.
"""
import os
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampere_gemm_i8_rotary import TensorOpGemmI8Rotary, create_and_permute_tensor


def quantize_tensor(t, dim=-1):
    qm = 127
    fr = t.abs().amax(dim=dim).clamp_min(1e-8)
    qs = qm / fr
    ti = (t * qs.unsqueeze(dim)).round().clamp(-qm, qm).to(torch.int8)
    return ti, qs.to(torch.float32).reciprocal()   # int8, dequant scale


# ── problem dims (small, so mismatches are easy to inspect) ───────────────────
batch_size, sequence_len = 4, 16
in_features = 512
nhead, head_dim, rotary_dim = 8, 64, 64
out_features = 3 * nhead * head_dim   # 1536

M = batch_size * sequence_len         # 64
K = in_features
N = out_features
seqlen = sequence_len
rotary_half = rotary_dim // 2         # 32
L = 1

print(f"Problem: M={M} (batch={batch_size}x seq={sequence_len}), N={N}, K={K}")
print(f"Rotary:  nhead={nhead}, head_dim={head_dim}, rotary_dim={rotary_dim}, rotary_half={rotary_half}")
print(f"sin/cos buffer: [{seqlen}, {rotary_half}]  (canonical RoPE freqs)\n")

# ── inputs ─────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
A_f32 = torch.randn(M, K) * 0.1
B_f32 = torch.randn(N, K) * 0.1        # (N,K) nn.Linear weight
A_int8, scale_a = quantize_tensor(A_f32, dim=-1)   # (M,)
B_int8, scale_b = quantize_tensor(B_f32, dim=-1)   # (N,)

# Canonical RoPE sin/cos: inv_freq[i] = 1 / 10000^(2i/head_dim)
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_half).float() / head_dim * 2))
t = torch.arange(seqlen).float()
freqs = torch.outer(t, inv_freq)       # [seqlen, rotary_half]
sin_buf = freqs.sin().contiguous().cuda()
cos_buf = freqs.cos().contiguous().cuda()

# ── kernel config (bN multiple of head_dim) ───────────────────────────────────
a_dtype = b_dtype = cutlass.Int8
c_dtype = cutlass.Float16
acc_dtype = cutlass.Int32
atom_layout_mnk = (2, 2, 1)
num_stages = 3
use_k32 = True
bm, bN, bK = 128, 128, 64

M_pad = ((M + bm - 1) // bm) * bm
N_pad = ((N + bN - 1) // bN) * bN
K_pad = ((K + bK - 1) // bK) * bK

mA, a_torch = create_and_permute_tensor(L, M_pad, K_pad, False, a_dtype)
mB, b_torch = create_and_permute_tensor(L, N_pad, K_pad, False, b_dtype)
mC, c_torch = create_and_permute_tensor(L, M_pad, N_pad, False, c_dtype)
a_torch[:M, :K, 0] = A_int8.cuda()
b_torch[:N, :K, 0] = B_int8.cuda()
for tt, r, c in [(a_torch, M, K), (b_torch, N, K)]:
    if tt.shape[0] > r: tt[r:, :, :] = 0
    if tt.shape[1] > c: tt[:, c:, :] = 0

scale_a_t = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); scale_a_t[:M, 0] = scale_a.cuda()
scale_b_t = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scale_b_t[:N, 0] = scale_b.cuda()
mScaleA = from_dlpack(scale_a_t, assumed_align=16)
mScaleB = from_dlpack(scale_b_t, assumed_align=16)
mSin = from_dlpack(sin_buf, assumed_align=16)
mCos = from_dlpack(cos_buf, assumed_align=16)

gemm = TensorOpGemmI8Rotary(
    a_dtype, b_dtype, c_dtype, acc_dtype, atom_layout_mnk, use_k32, bm,
    bn=bN, num_stages=num_stages,
    nhead=nhead, head_dim=head_dim, rotary_dim=rotary_dim, seqlen=seqlen,
)
print("Compiling fused kernel...")
compiled = cute.compile(gemm, mA, mB, mC, mScaleA, mScaleB, mSin, mCos)
compiled(mA, mB, mC, mScaleA, mScaleB, mSin, mCos)
torch.cuda.synchronize()
C_fused = c_torch[:M, :N, 0]

# ── OpenFish reference (PyTorch) ──────────────────────────────────────────────
with torch.inference_mode():
    A_dq = A_int8.float().cuda() * scale_a.cuda()[:, None]
    B_dq = B_int8.float().cuda() * scale_b.cuda()[:, None]
    C_ref = A_dq @ B_dq.T                              # (M, N) float32

    rows = torch.arange(M, device="cuda")
    seq_idx = rows % seqlen
    cos_row = cos_buf[seq_idx]                         # [M, rotary_half]
    sin_row = sin_buf[seq_idx]

    for chunk_start in (0, nhead * head_dim):          # Q, then K
        for h in range(nhead):
            h0 = chunk_start + h * head_dim
            x0 = C_ref[:, h0:h0 + rotary_half].clone()
            x1 = C_ref[:, h0 + rotary_half:h0 + rotary_dim].clone()
            C_ref[:, h0:h0 + rotary_half] = x0 * cos_row - x1 * sin_row
            C_ref[:, h0 + rotary_half:h0 + rotary_dim] = x0 * sin_row + x1 * cos_row
    # V cols [2*nhead*head_dim, N) unchanged
    C_ref_fp16 = C_ref.to(torch.float16)

# ── compare ────────────────────────────────────────────────────────────────────
print("\n=== Comparison: fused INT8 kernel vs OpenFish reference ===")
abs_err = (C_fused.float() - C_ref_fp16.float()).abs()
rel_err = abs_err / (C_ref_fp16.float().abs().clamp(min=1e-6))
print(f"  Max  abs error:  {abs_err.max().item():.6f}")
print(f"  Mean abs error:  {abs_err.mean().item():.8f}")
print(f"  Median abs err:  {abs_err.median().item():.8f}")
print(f"  Max  rel error:  {rel_err.max().item():.6f}")

print("\n=== Per-region error breakdown ===")
q_end = nhead * head_dim
k_end = 2 * nhead * head_dim
for region_name, col_sl in [
    ("Q (rotary)", slice(0, q_end)),
    ("K (rotary)", slice(q_end, k_end)),
    ("V (passthrough)", slice(k_end, N)),
]:
    e = abs_err[:, col_sl]
    print(f"  {region_name:16s}: max={e.max().item():.6f}  mean={e.mean().item():.8f}")

# ── large individual errors ──────────────────────────────────────────────────
threshold = 0.05
bad = (abs_err > threshold).nonzero(as_tuple=False)
if bad.numel() == 0:
    print(f"\nAll elements within {threshold} abs tolerance. PASS")
    sys.exit(0)
else:
    print(f"\n{bad.shape[0]} elements exceed {threshold} abs tolerance:")
    for i in range(min(20, bad.shape[0])):
        r, c = bad[i, 0].item(), bad[i, 1].item()
        chunk = "Q" if c < q_end else ("K" if c < k_end else "V")
        head = (c % (nhead * head_dim)) // head_dim
        pos = c % head_dim
        seq = r % seqlen
        print(f"  row={r:5d} col={c:4d} ({chunk} h={head} pos={pos:2d} seq={seq:3d})"
              f"  fused={C_fused[r, c].item():+.4f}  ref={C_ref_fp16[r, c].item():+.4f}"
              f"  err={abs_err[r, c].item():.4f}")
    sys.exit(1)
