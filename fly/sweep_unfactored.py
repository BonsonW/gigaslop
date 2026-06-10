"""Parameter sweep for the FP8 unfactored LSTM GEMM kernel.

Sweeps tile_m, tile_n_h, tile_k, k_unroll, group_m at B=512, H=1024.
Skips configs where the estimated VGPR count exceeds the budget.
Prints results sorted by time (fastest first).

Run:
    python fly/sweep_unfactored.py
"""

import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from rdna_fp8_unfactored_lstm_gemm import (
    compile_fp8_unfactored_lstm_gemm,
    fp8_quantize_per_channel,
    fp8_quantize_per_token,
    make_ih_t_interleaved,
    make_w_fused,
    preshuffle_b_fp8,
)

# ── Problem dimensions ────────────────────────────────────────────────────────
B  = 512
H  = 1024
FH = 4 * H

WARMUP = 20
ITERS  = 200

# ── VGPR estimation ───────────────────────────────────────────────────────────
WMMA_M  = 16
WMMA_N  = 16
WMMA_K  = 16
WAVES_M = 1
WAVES_N = 2
VGPR_BUDGET = 220


def estimate_vgprs(tile_m, tile_n_h, tile_k, k_unroll):
    """Estimate per-thread VGPR usage for the unfactored LSTM kernel."""
    wave_reg_m   = (tile_m   // WMMA_M) // WAVES_M
    wave_reg_n_h = (tile_n_h // WMMA_N) // WAVES_N
    reg_k        = tile_k // WMMA_K

    acc_vgprs = 4 * wave_reg_m * wave_reg_n_h * 8
    a_vgprs   = k_unroll * reg_k * wave_reg_m * 2
    b_vgprs   = k_unroll * 4 * reg_k * wave_reg_n_h * 2

    return acc_vgprs + a_vgprs + b_vgprs


# ── Input tensors (shared across all configs) ─────────────────────────────────

def make_inputs(device, seed=42):
    torch.manual_seed(seed)

    hh_f32        = torch.randn(B, H,  device=device) * 0.1
    dn_weight_f32 = torch.randn(128, H, device=device) * 0.1
    up_weight_f32 = torch.randn(FH,  128, device=device) * 0.1
    up_bias_f32   = torch.randn(FH,       device=device) * 0.05
    ih_t_f32      = torch.randn(B, FH,   device=device) * 0.1
    c_f32         = torch.randn(B, H,    device=device) * 0.1

    hh_fp8, scale_hh = fp8_quantize_per_token(hh_f32)
    wf_shuf, scale_wf = make_w_fused(dn_weight_f32, up_weight_f32)
    ih_t_interleaved = make_ih_t_interleaved(ih_t_f32.to(torch.float16))

    return dict(
        hh_fp8   = hh_fp8,
        scale_hh = scale_hh,
        wf_shuf  = wf_shuf,
        scale_wf = scale_wf,
        up_bias  = up_bias_f32,
        ih_t_f16 = ih_t_interleaved,
        c_base   = c_f32,
    )


# ── Benchmark helper ──────────────────────────────────────────────────────────

def time_launcher(launcher, inp, device, stream):
    h_fp8_out = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_inout   = inp["c_base"].clone()

    def run():
        launcher(
            h_fp8_out, c_inout,
            inp["hh_fp8"], inp["scale_hh"],
            inp["wf_shuf"], inp["scale_wf"],
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

    return t0.elapsed_time(t1) * 1e3 / ITERS   # microseconds


# ── Sweep ─────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required")
        sys.exit(1)

    device = "cuda"
    stream = torch.cuda.current_stream()

    print(f"Device : {torch.cuda.get_device_name()}")
    print(f"Problem: B={B}, H={H}")
    print(f"Warmup : {WARMUP}  Iters: {ITERS}")
    print(f"VGPR budget (skip threshold): {VGPR_BUDGET}")
    print()

    inp = make_inputs(device)

    tile_m_vals   = [32, 64, 128]
    tile_n_h_vals = [32, 64]
    tile_k_vals   = [16, 32]
    k_unroll_vals = [1, 2, 4, 8]
    group_m_vals  = [1, 4, 8, 16, 32]

    configs = []
    for tile_m in tile_m_vals:
        for tile_n_h in tile_n_h_vals:
            for tile_k in tile_k_vals:
                for k_unroll in k_unroll_vals:
                    if B % tile_m != 0: continue
                    if H % tile_n_h != 0: continue
                    if H % tile_k != 0: continue
                    # k_unroll must divide num_k_tiles - 1 evenly OR be handled by rem loop
                    # Just require tile_k * k_unroll <= H (don't over-unroll)
                    if tile_k * k_unroll > H: continue
                    vgpr = estimate_vgprs(tile_m, tile_n_h, tile_k, k_unroll)
                    if vgpr > VGPR_BUDGET: continue
                    for group_m in group_m_vals:
                        configs.append((tile_m, tile_n_h, tile_k, k_unroll, group_m, vgpr))

    print(f"Total valid configs: {len(configs)}")
    print()

    results = []

    # Track best per (tile_m, tile_n_h, tile_k, k_unroll) across group_m to avoid redundant compiles
    compiled = {}

    for tile_m, tile_n_h, tile_k, k_unroll, group_m, vgpr in configs:
        key = (tile_m, tile_n_h, tile_k, k_unroll)
        tag = f"m={tile_m:3d} nh={tile_n_h:2d} k={tile_k:2d} ku={k_unroll} gm={group_m:2d}"
        print(f"  {tag}  vgpr={vgpr:3d} ... ", end="", flush=True)
        try:
            launcher = compile_fp8_unfactored_lstm_gemm(
                B=B, H=H,
                tile_m=tile_m,
                tile_n_h=tile_n_h,
                tile_k=tile_k,
                k_unroll=k_unroll,
                group_m=group_m,
            )
            us = time_launcher(launcher, inp, device, stream)
            print(f"{us:.1f} µs")
            results.append((us, tile_m, tile_n_h, tile_k, k_unroll, group_m, vgpr, None))
        except Exception as e:
            short = str(e).split("\n")[0][:100]
            print(f"ERR: {short}")
            results.append((float("inf"), tile_m, tile_n_h, tile_k, k_unroll, group_m, vgpr, short))

    # ── Print sorted results ───────────────────────────────────────────────────
    print()
    print("=" * 82)
    print("Results sorted by time (fastest first)")
    print("=" * 82)
    print(f"  {'tile_m':>6}  {'tile_n_h':>8}  {'tile_k':>6}  {'k_unroll':>8}  {'group_m':>7}  {'vgpr':>4}  {'time µs':>8}")
    print("-" * 82)

    ok = [r for r in results if r[-1] is None]
    ok.sort(key=lambda x: x[0])

    for us, tm, tn, tk, ku, gm, vg, _ in ok[:30]:
        base = " <--" if (tm == 32 and tn == 32 and tk == 32 and ku == 2 and gm == 8) else ""
        print(f"  {tm:>6}  {tn:>8}  {tk:>6}  {ku:>8}  {gm:>7}  {vg:>4}  {us:>8.1f}{base}")

    bad = [r for r in results if r[-1] is not None]
    if bad:
        print()
        print("Compile errors:")
        for _, tm, tn, tk, ku, gm, vg, err in bad[:10]:
            print(f"  m={tm} nh={tn} k={tk} ku={ku} gm={gm}: {err}")

    if ok:
        best = ok[0]
        print()
        print(f"Best: tile_m={best[1]} tile_n_h={best[2]} tile_k={best[3]} "
              f"k_unroll={best[4]} group_m={best[5]}  vgpr={best[6]}  time={best[0]:.1f} µs")
        baseline = next(
            (r for r in ok if r[1]==32 and r[2]==32 and r[3]==32 and r[4]==2 and r[5]==8), None
        )
        if baseline:
            print(f"Speedup vs tile_m=32/nh=32/k=32/ku=2/gm=8 ({baseline[0]:.1f} µs): "
                  f"{baseline[0]/best[0]:.2f}×")


if __name__ == "__main__":
    main()
