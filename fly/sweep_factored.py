"""Parameter sweep for the FP8 factored LSTM GEMM kernel.

Sweeps tile_m, tile_n_h, tile_k1, k_unroll, group_m at B=512, H=1024, R=128.
Skips configs where estimated VGPR count exceeds budget.
Prints results sorted by time (fastest first).

Run:
    python fly/sweep_factored.py
"""

import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from rdna_fp8_factored_lstm_gemm import (
    compile_fp8_factored_lstm_gemm,
    fp8_quantize_per_channel,
    fp8_quantize_per_token,
    preshuffle_b_fp8,
    preshuffle_b_f16,
)

# ── Problem dimensions ────────────────────────────────────────────────────────
B  = 512
H  = 1024
R  = 128
FH = 4 * H

WARMUP = 20
ITERS  = 200

WMMA_M  = 16
WMMA_N  = 16
WMMA_K  = 16
WAVES_M = 1
WAVES_N = 2
VGPR_BUDGET = 220


def estimate_vgprs_factored(tile_m, tile_n_h, tile_k1, k_unroll):
    """Estimate per-thread VGPR usage for the factored LSTM kernel.

    Phase 1: fp8 WMMA over K=H with tile_k1
    Phase 2: f16 WMMA over K=R (tile_k2=16 fixed, 8 K-tiles)
    """
    wave_reg_m   = (tile_m   // WMMA_M) // WAVES_M
    wave_reg_n_h = (tile_n_h // WMMA_N) // WAVES_N
    wave_reg_n_y = (R // WMMA_N) // WAVES_N         # 4
    reg_k1       = tile_k1 // WMMA_K
    reg_k2       = 1                                  # tile_k2=16 fixed

    # Phase 1 accumulators: [wave_reg_m × wave_reg_n_y] × 8 f32
    acc1_vgprs = wave_reg_m * wave_reg_n_y * 8
    # Phase 2 accumulators: 4 gates × [wave_reg_m × wave_reg_n_h] × 8 f32
    acc2_vgprs = 4 * wave_reg_m * wave_reg_n_h * 8

    # A/B fragment buffers for phase 1 (double-buffered = k_unroll)
    a1_vgprs  = k_unroll * reg_k1 * wave_reg_m * 2
    b1_vgprs  = k_unroll * reg_k1 * wave_reg_n_y * 2

    # Phase 2: A from LDS (no reg buffer needed beyond current tile), B in registers
    a2_vgprs  = reg_k2 * wave_reg_m * 0   # loaded from LDS per tile, not double-buffered
    b2_vgprs  = reg_k2 * 4 * wave_reg_n_h * 2  # 4 gates, one tile at a time

    return acc1_vgprs + acc2_vgprs + a1_vgprs + b1_vgprs + a2_vgprs + b2_vgprs


# ── Input tensors ─────────────────────────────────────────────────────────────

def make_inputs(device, seed=42):
    torch.manual_seed(seed)

    hh_f32        = torch.randn(B, H,  device=device) * 0.1
    dn_weight_f32 = torch.randn(R, H,  device=device) * 0.1
    up_weight_f32 = torch.randn(FH, R, device=device) * 0.1
    up_bias_f32   = torch.randn(FH,    device=device) * 0.05
    ih_t_f32      = torch.randn(B, FH, device=device) * 0.1
    c_f32         = torch.randn(B, H,  device=device) * 0.1

    hh_fp8, scale_hh = fp8_quantize_per_token(hh_f32)

    dn_fp8, scale_dn = fp8_quantize_per_channel(dn_weight_f32.t().contiguous())
    dn_shuf = preshuffle_b_fp8(dn_fp8)

    up_weight_rn = up_weight_f32.t().contiguous()   # [R, 4H]
    up_shuf_f16  = preshuffle_b_f16(up_weight_rn.to(torch.float16))

    return dict(
        hh_fp8    = hh_fp8,
        scale_hh  = scale_hh,
        dn_shuf   = dn_shuf,
        scale_dn  = scale_dn,
        up_shuf_f16 = up_shuf_f16,
        up_bias   = up_bias_f32,
        ih_t_f16  = ih_t_f32.to(torch.float16),
        c_base    = c_f32,
    )


# ── Benchmark helper ──────────────────────────────────────────────────────────

def time_launcher(launcher, inp, device, stream):
    h_fp16_out = torch.zeros(B, H, dtype=torch.float16, device=device)
    c_inout    = inp["c_base"].clone()

    def run():
        launcher(
            h_fp16_out, c_inout,
            inp["hh_fp8"], inp["scale_hh"],
            inp["dn_shuf"], inp["scale_dn"],
            inp["up_shuf_f16"],
            inp["up_bias"], inp["ih_t_f16"],
            stream, B,
        )

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record(stream)
    for _ in range(ITERS):
        run()
    t1.record(stream)
    torch.cuda.synchronize()

    return t0.elapsed_time(t1) * 1e3 / ITERS


# ── Sweep ─────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required")
        sys.exit(1)

    device = "cuda"
    stream = torch.cuda.current_stream()

    print(f"Device : {torch.cuda.get_device_name()}")
    print(f"Problem: B={B}, H={H}, R={R}")
    print(f"Warmup : {WARMUP}  Iters: {ITERS}")
    print(f"VGPR budget: {VGPR_BUDGET}")
    print()

    inp = make_inputs(device)

    tile_m_vals   = [32, 64]
    tile_n_h_vals = [32, 64]
    tile_k1_vals  = [16, 32]
    k_unroll_vals = [1, 2, 4]
    group_m_vals  = [1, 4, 8, 16, 32]

    configs = []
    for tile_m in tile_m_vals:
        for tile_n_h in tile_n_h_vals:
            for tile_k1 in tile_k1_vals:
                for k_unroll in k_unroll_vals:
                    if B % tile_m != 0: continue
                    if H % tile_n_h != 0: continue
                    if H % tile_k1 != 0: continue
                    if R % 16 != 0: continue          # tile_k2=16 fixed
                    if tile_k1 * k_unroll > H: continue
                    vgpr = estimate_vgprs_factored(tile_m, tile_n_h, tile_k1, k_unroll)
                    if vgpr > VGPR_BUDGET: continue
                    for group_m in group_m_vals:
                        configs.append((tile_m, tile_n_h, tile_k1, k_unroll, group_m, vgpr))

    print(f"Total valid configs: {len(configs)}")
    print()

    results = []

    for tile_m, tile_n_h, tile_k1, k_unroll, group_m, vgpr in configs:
        tag = f"m={tile_m:2d} nh={tile_n_h:2d} k1={tile_k1:2d} ku={k_unroll} gm={group_m:2d}"
        print(f"  {tag}  vgpr={vgpr:3d} ... ", end="", flush=True)
        try:
            launcher = compile_fp8_factored_lstm_gemm(
                B=B, H=H, R=R,
                tile_m=tile_m,
                tile_n_h=tile_n_h,
                tile_k1=tile_k1,
                k_unroll=k_unroll,
                group_m=group_m,
            )
            us = time_launcher(launcher, inp, device, stream)
            print(f"{us:.1f} µs")
            results.append((us, tile_m, tile_n_h, tile_k1, k_unroll, group_m, vgpr, None))
        except Exception as e:
            short = str(e).split("\n")[0][:100]
            print(f"ERR: {short}")
            results.append((float("inf"), tile_m, tile_n_h, tile_k1, k_unroll, group_m, vgpr, short))

    print()
    print("=" * 82)
    print("Results sorted by time (fastest first)")
    print("=" * 82)
    print(f"  {'tile_m':>6}  {'tile_n_h':>8}  {'tile_k1':>7}  {'k_unroll':>8}  {'group_m':>7}  {'vgpr':>4}  {'time µs':>8}")
    print("-" * 82)

    ok = [r for r in results if r[-1] is None]
    ok.sort(key=lambda x: x[0])

    for us, tm, tn, tk1, ku, gm, vg, _ in ok[:30]:
        base = " <--" if (tm==32 and tn==32 and tk1==32 and ku==2 and gm==8) else ""
        print(f"  {tm:>6}  {tn:>8}  {tk1:>7}  {ku:>8}  {gm:>7}  {vg:>4}  {us:>8.1f}{base}")

    bad = [r for r in results if r[-1] is not None]
    if bad:
        print()
        print("Compile errors:")
        for _, tm, tn, tk1, ku, gm, vg, err in bad[:10]:
            print(f"  m={tm} nh={tn} k1={tk1} ku={ku} gm={gm}: {err}")

    if ok:
        best = ok[0]
        print()
        print(f"Best: tile_m={best[1]} tile_n_h={best[2]} tile_k1={best[3]} "
              f"k_unroll={best[4]} group_m={best[5]}  vgpr={best[6]}  time={best[0]:.1f} µs")
        baseline = next(
            (r for r in ok if r[1]==32 and r[2]==32 and r[3]==32 and r[4]==2 and r[5]==8), None
        )
        if baseline:
            print(f"Speedup vs default ({baseline[0]:.1f} µs): {baseline[0]/best[0]:.2f}×")


if __name__ == "__main__":
    main()
