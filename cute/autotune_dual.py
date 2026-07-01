"""Autotune the dual INT8 GEMM + SiLU kernel (ampere_dual_gemm_i8_silu.py).

Mirrors autotune.py (which tunes the single-GEMM TensorOpGemmI8), but instantiates
TensorOpDualGemmI8Silu with its 7 tensor args (mA, mB_gate, mB_up, mC, and three fp32
scales) and reports TOPS with the dual-GEMM op count (2 * 2*M*N*K).

Findings baked into the default grid (A100, K=512, N=2048; warmed clocks + interleaved
repeats -- cold-clock sweeps are noisy, always warm up before trusting a comparison):
  - bN=32 ties bN=64 at large M (~215 TOPS at M=131072) and clearly wins at small M
    (M=256: ~61-72 vs ~54 TOPS). Halving the dual accumulator raises occupancy, which
    only helps when there's little work to hide latency -> bN=32 is the robust default.
  - num_stages=3 is best for short K (K=512 -> only 8 k-tiles); 4+ stages reduce occupancy.
  - bN=128 spills catastrophically; excluded.
  - bm=64 helps at small M (more CTAs/SM); bm=128 is best at the deployed large M.

Config validity: bN must be >= atom_N * mmaN * 2 (the permutation_mnk N term doubles
the MMA-N atom). bN=32 is therefore only valid with atom_N=2; bN=32 + atom_N=4 produces
a degenerate tiling that hangs compilation. _valid_config() enforces this before compile.

Run:
    pyvenv/bin/python cute/autotune_dual.py
    pyvenv/bin/python cute/autotune_dual.py --M 4096 --N 2048 --K 512
"""
import argparse
import itertools
import sys, os

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampere_dual_gemm_i8_silu import (
    TensorOpDualGemmI8Silu, dual_gemm_silu_ref, create_and_permute_tensor,
)


def _valid_config(bN, atom_mnk, use_k32):
    """bN must cover one full MMA-N permutation tile: atom_N * mmaN * 2."""
    mma_n = 8
    atom_n = atom_mnk[1]
    if bN < atom_n * mma_n * 2:
        return False
    if bN % (atom_n * mma_n) != 0:
        return False
    return True


def _quantize(t, dim=-1):
    qm = 127
    fp = t.abs().amax(dim=dim).clamp_min(1e-8)
    qs = qm / fp
    ti = (t * qs.unsqueeze(dim)).round().clamp(-qm, qm).to(torch.int8)
    return ti, qs.to(torch.float32).reciprocal()


def _build_inputs(M, N, K, bm, bN, bK, L=1):
    torch.manual_seed(42)
    A = torch.randn(M, K) * 0.1
    Bg = torch.randn(N, K) * 0.1
    Bu = torch.randn(N, K) * 0.1
    Ai, sa = _quantize(A)
    Bgi, sbg = _quantize(Bg)
    Bui, sbu = _quantize(Bu)

    Mp = ((M + bm - 1) // bm) * bm
    Np = ((N + bN - 1) // bN) * bN
    Kp = ((K + bK - 1) // bK) * bK

    mA, at = create_and_permute_tensor(L, Mp, Kp, False, cutlass.Int8)
    mBg, bgt = create_and_permute_tensor(L, Np, Kp, False, cutlass.Int8)
    mBu, but = create_and_permute_tensor(L, Np, Kp, False, cutlass.Int8)
    mC, ct = create_and_permute_tensor(L, Mp, Np, False, cutlass.Float16)
    at[:M, :K, 0] = Ai.cuda(); bgt[:N, :K, 0] = Bgi.cuda(); but[:N, :K, 0] = Bui.cuda()
    for t, r, c in [(at, M, K), (bgt, N, K), (but, N, K)]:
        if t.shape[0] > r: t[r:, :, :] = 0
        if t.shape[1] > c: t[:, c:, :] = 0

    sat = torch.zeros(Mp, L, dtype=torch.float32, device='cuda'); sat[:M, 0] = sa.cuda()
    sbgt = torch.zeros(Np, L, dtype=torch.float32, device='cuda'); sbgt[:N, 0] = sbg.cuda()
    sbut = torch.zeros(Np, L, dtype=torch.float32, device='cuda'); sbut[:N, 0] = sbu.cuda()
    mSa = from_dlpack(sat, assumed_align=16)
    mSbg = from_dlpack(sbgt, assumed_align=16)
    mSbu = from_dlpack(sbut, assumed_align=16)

    refs = (Ai, Bgi, Bui, sa, sbg, sbu, ct)
    return (mA, mBg, mBu, mC, mSa, mSbg, mSbu), refs


def autotune(M, N, K, configs, check=True):
    bK = 64
    results = []
    for bm, bN, atom_mnk, num_stages, use_k32 in configs:
        tag = f"bm={bm:3d} bN={bN:3d} atom={str(atom_mnk):9s} stages={num_stages} k32={use_k32}"
        if not _valid_config(bN, atom_mnk, use_k32):
            print(f"{tag}  SKIP (invalid: bN < atom_N*mmaN*2)")
            continue
        try:
            args, refs = _build_inputs(M, N, K, bm, bN, bK)
            kernel = TensorOpDualGemmI8Silu(
                cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
                atom_mnk, use_k32, bm, bn=bN, num_stages=num_stages,
            )
            compiled = cute.compile(kernel, *args)

            if check:
                compiled(*args); torch.cuda.synchronize()
                Ai, Bgi, Bui, sa, sbg, sbu, ct = refs
                with torch.inference_mode():
                    ref = dual_gemm_silu_ref(
                        Ai.cuda(), Bgi.cuda(), Bui.cuda(),
                        sa.cuda(), sbg.cuda(), sbu.cuda(),
                    ).to(torch.float16)
                    err = (ct[:M, :N, 0].float() - ref.float()).abs().max().item()
            else:
                err = float('nan')

            avg_us = cute.testing.benchmark(
                compiled,
                kernel_arguments=cute.testing.JitArguments(*args),
                warmup_iterations=10, iterations=100,
            )
            ops = 2 * (2 * M * N * K)
            tops = ops / (avg_us * 1e-6) / 1e12
            print(f"{tag}  {avg_us:9.1f}us  {tops:6.1f} TOPS ({tops/624*100:4.1f}%) err={err:.4f}")
            results.append((avg_us, bm, bN, atom_mnk, num_stages, use_k32))
        except Exception as e:
            print(f"{tag}  FAILED: {str(e)[:80]}")

    results.sort()
    print(f"\n=== Top configs for M={M} N={N} K={K} ===")
    for avg_us, bm, bN, atom_mnk, num_stages, use_k32 in results[:3]:
        ops = 2 * (2 * M * N * K)
        tops = ops / (avg_us * 1e-6) / 1e12
        print(f"  bm={bm} bN={bN} atom={atom_mnk} stages={num_stages} k32={use_k32}"
              f"  {avg_us:.1f}us  {tops:.1f} TOPS")
    return results


DEFAULT_CONFIGS = [
    # (bm, bN, atom_mnk, num_stages, use_k32)
    (128,  32, (2, 2, 1), 3, True),   # recommended default for K=512
    (128,  32, (2, 2, 1), 4, True),
    (128,  64, (2, 2, 1), 3, True),   # previous default
    (128,  64, (2, 2, 1), 4, True),
    (128,  64, (2, 4, 1), 3, True),   # wider atom needs bN>=64
    (64,   32, (2, 2, 1), 3, True),   # smaller bM -> more CTAs/SM
    (64,   64, (2, 4, 1), 3, True),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autotune the dual INT8 GEMM + SiLU kernel.")
    p.add_argument("--M", type=int, default=None, help="single M to tune (default: sweep 256,4096,131072)")
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--no-check", action="store_true", help="skip correctness verification")
    args = p.parse_args()

    Ms = [args.M] if args.M is not None else [256, 4096, 131072]
    for M in Ms:
        print(f"\n########## M={M} N={args.N} K={args.K} ##########")
        autotune(M, args.N, args.K, DEFAULT_CONFIGS, check=not args.no_check)
