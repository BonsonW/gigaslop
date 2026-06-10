"""Verify and benchmark rdna_fp8_factored_lstm_gemm against a PyTorch reference.

Verifies one LSTM timestep:
  - Factored hh GEMM: hh_fp8 @ dn_weight^T @ up_weight^T
  - sigmoid_hard / tanh_hard gates
  - Cell + hidden state update with exact tanh(c)

Also benchmarks the fused kernel vs. a naive two-GEMM PyTorch baseline.

Run:
    python bench_fp8_factored_lstm.py [--no-verify] [--batch B] [--iters N]
"""

import argparse
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
from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize

# ── Problem dimensions ────────────────────────────────────────────────────────
H    = 1024   # hidden dim
R    = 128    # bottleneck rank
FH   = 4 * H  # gate output dim

# RDNA4 AI Pro R9700 peak numbers
PEAK_FP8_TOPS  = 383.0   # TFLOPS
PEAK_BW_GBS    = 640.0   # GB/s GDDR6


# =============================================================================
# PyTorch reference (exact match to C++ FLSTMLayerImpl::forward)
# =============================================================================

def lstm_step_ref(hh_f32, dn_weight_f32, up_weight_f32, up_bias_f32, ih_t_f32, c_f32):
    """One LSTM timestep in f32, matching the C++ source exactly.

    hh_f32:       [B, H]
    dn_weight_f32:[R, H]  (dn = down-projection)
    up_weight_f32:[4H, R] (up = up-projection)
    up_bias_f32:  [4H]
    ih_t_f32:     [B, 4H]
    c_f32:        [B, H]
    """
    # Factored two-GEMM chain
    y     = hh_f32 @ dn_weight_f32.t()                     # [B, R]
    gates = y @ up_weight_f32.t() + up_bias_f32 + ih_t_f32 # [B, 4H]

    i_l, f_l, g_l, o_l = gates.chunk(4, dim=1)

    # sigmoid_hard = clamp(0.2*x + 0.5, 0, 1)
    i_a = (i_l * 0.2 + 0.5).clamp(0.0, 1.0)
    f_a = (f_l * 0.2 + 0.5).clamp(0.0, 1.0)
    # tanh_hard = clamp(x, -1, 1)
    g_a = g_l.clamp(-1.0, 1.0)
    o_a = (o_l * 0.2 + 0.5).clamp(0.0, 1.0)

    c_new = f_a * c_f32 + i_a * g_a
    h_new = o_a * c_new.tanh()   # exact tanh
    return h_new, c_new


# =============================================================================
# Setup
# =============================================================================

def make_inputs(B, device, seed=42):
    torch.manual_seed(seed)

    hh_f32        = torch.randn(B, H,  device=device) * 0.1
    dn_weight_f32 = torch.randn(R, H,  device=device) * 0.1
    up_weight_f32 = torch.randn(FH, R, device=device) * 0.1
    up_bias_f32   = torch.randn(FH,    device=device) * 0.05
    ih_t_f32      = torch.randn(B, FH, device=device) * 0.1
    c_f32         = torch.randn(B, H,  device=device) * 0.1

    # Quantise hh to fp8 per-token
    hh_fp8, scale_hh = fp8_quantize_per_token(hh_f32)

    # Quantise dn_weight per-channel, preshuffle for Phase 1
    dn_fp8, scale_dn = fp8_quantize_per_channel(dn_weight_f32.t().contiguous())
    # dn_fp8 shape: [H, R];  scale_dn shape: [R]
    dn_shuf = preshuffle_b_fp8(dn_fp8)

    # Phase 2 up_weight is f16 (no fp8 requantization needed)
    up_weight_rn = up_weight_f32.t().contiguous()      # [R, 4H]
    up_shuf = preshuffle_b_f16(up_weight_rn.to(torch.float16))

    # ih_t as f16
    ih_t_f16 = ih_t_f32.to(torch.float16)

    return dict(
        hh_fp8    = hh_fp8,
        scale_hh  = scale_hh,
        dn_shuf   = dn_shuf,
        scale_dn  = scale_dn,
        up_shuf   = up_shuf,
        up_bias   = up_bias_f32,
        ih_t_f16  = ih_t_f16,
        c_inout   = c_f32.clone(),
        # For reference
        hh_f32        = hh_f32,
        dn_weight_f32 = dn_weight_f32,
        up_weight_f32 = up_weight_f32,
        up_bias_f32   = up_bias_f32,
        ih_t_f32      = ih_t_f32,
        c_ref         = c_f32.clone(),
    )


# =============================================================================
# Correctness verification
# =============================================================================

def verify(B, device):
    print(f"\n{'='*60}")
    print(f"Correctness verification  B={B}, H={H}, R={R}")
    print(f"{'='*60}")

    inp    = make_inputs(B, device)
    stream = torch.cuda.current_stream()

    # ── Reference ─────────────────────────────────────────────────────────────
    h_ref, c_ref = lstm_step_ref(
        inp["hh_f32"], inp["dn_weight_f32"], inp["up_weight_f32"],
        inp["up_bias_f32"], inp["ih_t_f32"], inp["c_ref"],
    )

    # ── Kernel ────────────────────────────────────────────────────────────────
    launcher = compile_fp8_factored_lstm_gemm(B=B, H=H, R=R)

    h_fp16_out = torch.zeros(B, H, dtype=torch.float16,  device=device)
    c_inout    = inp["c_inout"]

    launcher(
        h_fp16_out, c_inout,
        inp["hh_fp8"], inp["scale_hh"],
        inp["dn_shuf"], inp["scale_dn"],
        inp["up_shuf"],
        inp["up_bias"], inp["ih_t_f16"],
        stream, B,
    )
    torch.cuda.synchronize()

    # ── h_new comparison ──────────────────────────────────────────────────────
    h_kernel = h_fp16_out.float()
    h_ref_f  = h_ref.float()
    h_err    = (h_kernel - h_ref_f).abs()
    # Allow fp8 quantisation noise (~1% of value range) + f16 truncation
    h_tol    = 0.08

    print(f"\nh_new  (f32 ref vs f16 kernel):")
    print(f"  Max  abs error : {h_err.max().item():.5f}")
    print(f"  Mean abs error : {h_err.mean().item():.7f}")
    print(f"  Max  ref value : {h_ref_f.abs().max().item():.5f}")
    h_ok = h_err.max().item() < h_tol
    print(f"  Tolerance {h_tol}  : {'PASS ✓' if h_ok else 'FAIL ✗'}")

    # ── c_new comparison ──────────────────────────────────────────────────────
    c_kernel = c_inout.float()
    c_ref_f  = c_ref.float()
    c_err    = (c_kernel - c_ref_f).abs()
    c_tol    = 0.08

    print(f"\nc_new  (f32 ref vs f32 kernel):")
    print(f"  Max  abs error : {c_err.max().item():.5f}")
    print(f"  Mean abs error : {c_err.mean().item():.7f}")
    c_ok = c_err.max().item() < c_tol
    print(f"  Tolerance {c_tol}  : {'PASS ✓' if c_ok else 'FAIL ✗'}")

    # ── FP8 round-trip sanity (chain with per-token quantize) ─────────────────
    quant_launcher = compile_fp8_per_token_quantize(K=H)
    h_fp8_out  = torch.zeros(B, H, dtype=torch.uint8,    device=device)
    h_scale_out = torch.zeros(B,    dtype=torch.float32, device=device)
    quant_launcher(h_fp8_out, h_scale_out, h_fp16_out, B, stream)
    torch.cuda.synchronize()

    h_dequant = h_fp8_out.view(torch.float8_e4m3fn).float() * h_scale_out[:, None]
    h_fp8_err = (h_dequant - h_ref_f).abs()
    fp8_tol   = 0.12   # fp8 adds more quantisation noise
    fp8_ok    = h_fp8_err.max().item() < fp8_tol

    print(f"\nh_new  (f32 ref vs fp8 dequant after per-token quantise):")
    print(f"  Max  abs error : {h_fp8_err.max().item():.5f}")
    print(f"  Mean abs error : {h_fp8_err.mean().item():.7f}")
    print(f"  Tolerance {fp8_tol}  : {'PASS ✓' if fp8_ok else 'FAIL ✗'}")

    return h_ok and c_ok and fp8_ok


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(B, device, warmup=20, iters=500):
    print(f"\n{'='*60}")
    print(f"Benchmark  B={B}, H={H}, R={R},  warmup={warmup}, iters={iters}")
    print(f"{'='*60}")

    inp    = make_inputs(B, device)
    stream = torch.cuda.current_stream()

    launcher       = compile_fp8_factored_lstm_gemm(B=B, H=H, R=R)
    quant_launcher = compile_fp8_per_token_quantize(K=H)

    h_fp16_out  = torch.zeros(B, H,  dtype=torch.float16, device=device)
    h_fp8_out   = torch.zeros(B, H,  dtype=torch.uint8,   device=device)
    h_scale_out = torch.zeros(B,     dtype=torch.float32, device=device)

    def run_fused():
        # Restore c_inout so we measure one representative step
        c_work = inp["c_inout"].clone()
        launcher(
            h_fp16_out, c_work,
            inp["hh_fp8"], inp["scale_hh"],
            inp["dn_shuf"], inp["scale_dn"],
            inp["up_shuf"],
            inp["up_bias"], inp["ih_t_f16"],
            stream, B,
        )
        quant_launcher(h_fp8_out, h_scale_out, h_fp16_out, B, stream)

    def run_baseline():
        """PyTorch reference in fp16 (unfactored: uses full fused weight)."""
        W_fused_f16 = (inp["up_weight_f32"] @ inp["dn_weight_f32"]).to(torch.float16)
        h16 = inp["hh_f32"].to(torch.float16)
        gates = h16 @ W_fused_f16.t() + inp["up_bias_f32"].to(torch.float16) + inp["ih_t_f16"]
        i_l, f_l, g_l, o_l = gates.float().chunk(4, dim=1)
        i_a = (i_l * 0.2 + 0.5).clamp(0, 1)
        f_a = (f_l * 0.2 + 0.5).clamp(0, 1)
        g_a = g_l.clamp(-1, 1)
        o_a = (o_l * 0.2 + 0.5).clamp(0, 1)
        c_work = inp["c_inout"].clone()
        c_work = f_a * c_work + i_a * g_a
        return o_a * c_work.tanh()

    def time_fn(fn, n_warmup, n_iters):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(stream)
        for _ in range(n_iters):
            fn()
        t1.record(stream)
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / n_iters   # microseconds

    us_fused    = time_fn(run_fused,    warmup, iters)
    us_baseline = time_fn(run_baseline, warmup, iters)

    # ── FLOP / byte accounting ─────────────────────────────────────────────────
    # Phase 1 GEMM:  B * H * R * 2
    # Phase 2 GEMM:  B * R * 4H * 2
    total_flops = 2 * B * H * R + 2 * B * R * FH

    # Memory touched per step (fused kernel):
    #   hh_fp8 [B*H bytes], dn_weight_fp8 [H*R bytes], up_weight_fp8 [R*4H bytes]
    #   up_bias [4H*4 bytes], ih_t_f16 [B*4H*2 bytes]
    #   c_inout read+write [B*H*4*2 bytes], h_fp16_out [B*H*2 bytes]
    wt_bytes   = (H * R * 1) + (R * FH * 2)                       # phase1 fp8 + phase2 f16
    act_bytes  = (B * H * 1) + (B * FH * 2) + (B * H * 4 * 2) + (B * H * 2)
    total_bytes = wt_bytes + act_bytes

    def print_stats(label, us):
        s = us * 1e-6
        tops  = total_flops / s / 1e12
        bw    = total_bytes / s / 1e9
        ai    = total_flops / total_bytes
        print(f"\n  [{label}]")
        print(f"    Time per step :  {us:.2f} µs")
        print(f"    T=1024 steps  :  {us * 1024 / 1e3:.2f} ms")
        print(f"    FLOPS         :  {tops:.3f} TFLOPS  ({tops/PEAK_FP8_TOPS*100:.1f}% of {PEAK_FP8_TOPS})")
        print(f"    Bandwidth     :  {bw:.1f} GB/s  ({bw/PEAK_BW_GBS*100:.1f}% of {PEAK_BW_GBS})")
        print(f"    Arith intens  :  {ai:.1f} FLOP/byte")

    print(f"\n  Problem: B={B}, H={H}, R={R}, 4H={FH}")
    print(f"  Total FLOPS/step : {total_flops/1e9:.3f} GFLOPS")
    print(f"  Weight bytes     : {wt_bytes/1e3:.1f} KB  (factored: {wt_bytes/1e3:.1f} vs unfactored {H*FH/1e3:.1f} KB fp8)")
    print(f"                     (phase1 fp8: {H*R/1e3:.1f} KB, phase2 f16: {R*FH*2/1e3:.1f} KB)")

    print_stats("fused kernel (FP8 phase1 + F16 phase2 + per-token quant)", us_fused)
    print_stats("PyTorch baseline (fp16 unfactored GEMM + element-wise)", us_baseline)

    speedup = us_baseline / us_fused
    print(f"\n  Speedup (fused / baseline): {speedup:.2f}×")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device required")
        sys.exit(1)

    device = "cuda"
    B = args.batch
    # tile_m=32, so B must be divisible by 32
    assert B % 32 == 0, f"B={B} must be divisible by tile_m=32"
    # tile_n_h=32 and H=1024, so H % tile_n_h == 0 ✓

    print(f"Device: {torch.cuda.get_device_name()}")

    all_ok = True
    if not args.no_verify:
        all_ok = verify(B=32, device=device)   # use small B for verify

    benchmark(B=B, device=device, iters=args.iters)

    if not args.no_verify:
        print(f"\nOverall: {'PASS ✓' if all_ok else 'FAIL ✗'}")
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
