import sys, os
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(__file__))
from ampere_gemm_i8_rotary import (
    TensorOpGemmI8Rotary, rotary_ref, create_and_permute_tensor,
)

verify_correctness = True

def quantize_tensor(t, dim=-1):
    qm = 127
    fr = t.abs().amax(dim=dim).clamp_min(1e-8)
    qs = qm / fr
    ti = (t * qs.unsqueeze(dim)).round().clamp(-qm, qm).to(torch.int8)
    return ti, qs.to(torch.float32).reciprocal()

# ── problem size (matches fly/bench_fp8_gemm_rotary.py) ───────────────────────
batch_size, sequence_len = 128, 1024
in_features = 512
nhead, head_dim, rotary_dim = 8, 64, 64
out_features = 3 * nhead * head_dim   # 1536
M = batch_size * sequence_len
K = in_features
N = out_features
seqlen = sequence_len
rotary_half = rotary_dim // 2
L = 1

torch.manual_seed(42)
A_f32 = torch.randn(M, K) * 0.1
B_f32 = torch.randn(N, K) * 0.1   # (N,K) weight
A_int8, scale_a = quantize_tensor(A_f32, dim=-1)
B_int8, scale_b = quantize_tensor(B_f32, dim=-1)

# sin/cos [seqlen, rotary_half]
theta = torch.arange(seqlen).float().unsqueeze(1) * 0.01
rot = torch.arange(rotary_half).float().unsqueeze(0)
sin_buf = torch.sin(theta + rot).contiguous()
cos_buf = torch.cos(theta + rot).contiguous()

# ── kernel config: bN MUST equal head_dim ────────────────────────────────────
a_dtype = b_dtype = cutlass.Int8
c_dtype = cutlass.Float16
acc_dtype = cutlass.Int32
atom_layout_mnk = (2, 4, 1)   # 8 warps: bigger N-tile needs the extra warps to split N
num_stages = 3
use_k32 = True
bm, bN, bK = 128, 256, 64   # bN=256 (4 heads/tile): fastest at this shape (~+3% vs bN=128)

M_pad = ((M + bm - 1) // bm) * bm
N_pad = ((N + bN - 1) // bN) * bN
K_pad = ((K + bK - 1) // bK) * bK

mA, a_torch = create_and_permute_tensor(L, M_pad, K_pad, False, a_dtype)
mB, b_torch = create_and_permute_tensor(L, N_pad, K_pad, False, b_dtype)
mC, c_torch = create_and_permute_tensor(L, M_pad, N_pad, False, c_dtype)
a_torch[:M, :K, 0] = A_int8.cuda()
b_torch[:N, :K, 0] = B_int8.cuda()
for t, r, c in [(a_torch, M, K), (b_torch, N, K)]:
    if t.shape[0] > r: t[r:, :, :] = 0
    if t.shape[1] > c: t[:, c:, :] = 0

scale_a_t = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); scale_a_t[:M, 0] = scale_a.cuda()
scale_b_t = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scale_b_t[:N, 0] = scale_b.cuda()
mScaleA = from_dlpack(scale_a_t, assumed_align=16)
mScaleB = from_dlpack(scale_b_t, assumed_align=16)
sin_g = sin_buf.cuda(); cos_g = cos_buf.cuda()
mSin = from_dlpack(sin_g, assumed_align=16)
mCos = from_dlpack(cos_g, assumed_align=16)
# seqlen is a runtime scalar now (table extent stays baked). Pass cutlass.Int32
# so it lowers to a runtime modulo divisor rather than a folded constant.
seqlen_arg = cutlass.Int32(seqlen)

gemm = TensorOpGemmI8Rotary(
    a_dtype, b_dtype, c_dtype, acc_dtype, atom_layout_mnk, use_k32, bm,
    bn=bN, num_stages=num_stages,
    nhead=nhead, head_dim=head_dim, rotary_dim=rotary_dim, seqlen=seqlen,
)
print(f"M={M} K={K} N={N}  tile={bm}x{bN}x{bK}  nhead={nhead} head_dim={head_dim} rotary_dim={rotary_dim}")
print("=== Compiling GEMM + rotary ===")
compiled = cute.compile(gemm, mA, mB, mC, mScaleA, mScaleB, mSin, mCos, seqlen_arg)

if verify_correctness:
    print("=== Verifying correctness ===")
    compiled(mA, mB, mC, mScaleA, mScaleB, mSin, mCos, seqlen_arg)
    torch.cuda.synchronize()
    with torch.inference_mode():
        ref = rotary_ref(A_int8.cuda(), B_int8.cuda(), scale_a.cuda(), scale_b.cuda(),
                         sin_g, cos_g, nhead, head_dim, rotary_dim, seqlen).to(torch.float16)
        out = c_torch[:M, :N, 0]
        err = (out.float() - ref.float()).abs()
        print(f"  Max abs error:  {err.max().item():.4f}")
        print(f"  Mean abs error: {err.mean().item():.6f}")

print("=== Benchmarking ===")
# A cold A100 ramps its SM clock over a few hundred ms; cute.testing.benchmark's
# short warmup finishes before clocks boost, under-reporting TOPS by ~25%. Warm the
# GPU with a sustained run first so the measured number reflects steady-state clocks.
for _ in range(400):
    compiled(mA, mB, mC, mScaleA, mScaleB, mSin, mCos, seqlen_arg)
torch.cuda.synchronize()
t = cute.testing.benchmark(
    compiled,
    kernel_arguments=cute.testing.JitArguments(mA, mB, mC, mScaleA, mScaleB, mSin, mCos, seqlen_arg),
    warmup_iterations=10, iterations=100,
)
tops = (2 * M * N * K) / (t * 1e-6) / 1e12
print(f"  Time: {t:.2f} us   {tops:.1f} TOPS  ({tops/624*100:.1f}% of 624 peak)")
