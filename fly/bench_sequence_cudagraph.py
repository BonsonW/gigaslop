"""Benchmark a full T-step LSTM sequence with and without CUDA graph capture.

Two modes compared:
  eager  — Python loop calls the FlyDSL launcher once per step (CPU-launch-bound)
  graph  — T launches captured in one CUDAGraph and replayed with a single CPU call

The unfactored kernel is used because it already fuses the fp8 output (fixed scale
1/448), so h_buf can alias input/output and no separate quantize launch is needed.

Run:
    python bench_sequence_cudagraph.py [--batch B] [--seq T] [--iters N]
"""

import argparse
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_unfactored_lstm_gemm import (
    compile_fp8_unfactored_lstm_gemm,
    fp8_quantize_per_token,
    make_w_fused,
)

H         = 1024
R         = 128
FH        = 4 * H
FP8_SCALE = 1.0 / 448.0


def make_problem(B, T, device, seed=42):
    torch.manual_seed(seed)
    dn_w  = torch.randn(R,  H,  device=device) * 0.1
    up_w  = torch.randn(FH, R,  device=device) * 0.1
    bias  = torch.randn(FH,     device=device) * 0.05
    wf_shuf, scale_wf = make_w_fused(dn_w, up_w)

    # Input contributions for all T steps, pre-laid out in [B, H, 4] kernel format
    ih_t_raw = torch.randn(T, B, 4, H, device=device, dtype=torch.float16) * 0.1
    ih_t_all = ih_t_raw.permute(0, 1, 3, 2).contiguous()   # [T, B, H, 4]

    h0_fp8, _ = fp8_quantize_per_token(torch.randn(B, H, device=device) * 0.1)
    h0_fp8    = h0_fp8.view(torch.uint8)
    c0        = torch.randn(B, H, device=device) * 0.1

    return wf_shuf, scale_wf, bias, ih_t_all, h0_fp8, c0


def build_closures(launcher, B, wf_shuf, scale_wf, bias, ih_t_all, h_buf, scale_hh, c_buf):
    """Return (run_eager_step_t, run_graph_step_t) closures for step t.

    Both call torch.cuda.current_stream() at invocation time so they work both
    for normal eager calls and inside a torch.cuda.graph() capture context.
    """
    T = ih_t_all.shape[0]

    def run_step(t):
        launcher(h_buf, c_buf,
                 h_buf, scale_hh,
                 wf_shuf, scale_wf, bias, ih_t_all[t],
                 torch.cuda.current_stream(), B)

    return run_step, T


def time_eager(run_step, T, warmup=5, rounds=3, iters=20):
    """Time the T-step eager loop. Returns µs per full sequence."""
    for _ in range(warmup):
        for t in range(T):
            run_step(t)
    torch.cuda.synchronize()

    best = float("inf")
    stream = torch.cuda.current_stream()
    for _ in range(rounds):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(stream)
        for _ in range(iters):
            for t in range(T):
                run_step(t)
        t1.record(stream)
        torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1) * 1e3 / iters)
    return best


def time_graph(run_step, T, warmup=5, rounds=20, replays=8):
    """Capture the T-step loop in a CUDAGraph and time replay. Returns µs per sequence."""
    # Warmup on a side stream so all JIT paths are compiled before capture
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            for t in range(T):
                run_step(t)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Capture: torch.cuda.current_stream() inside the context resolves to the
    # capture stream, so FlyDSL submits work onto the correct stream.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for t in range(T):
            run_step(t)
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(rounds):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(replays):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1) * 1e3 / replays)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seq",   type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device required"); sys.exit(1)

    device = "cuda"
    B, T = args.batch, args.seq
    assert B % 32 == 0, f"B={B} must be divisible by tile_m=32"

    print(f"Device : {torch.cuda.get_device_name()}")
    print(f"Problem: B={B}, H={H}, T={T}")
    print()

    wf_shuf, scale_wf, bias, ih_t_all, h0_fp8, c0 = make_problem(B, T, device)

    print("Compiling kernel…")
    launcher = compile_fp8_unfactored_lstm_gemm(B=B, H=H)

    h_buf    = h0_fp8.clone()
    c_buf    = c0.clone()
    scale_hh = torch.full((B,), FP8_SCALE, device=device)

    # Force JIT compilation before timing
    run_step, _ = build_closures(launcher, B, wf_shuf, scale_wf, bias, ih_t_all,
                                 h_buf, scale_hh, c_buf)
    run_step(0)
    torch.cuda.synchronize()
    print("Compiled.\n")

    print(f"Timing eager  (T={T} per-step Python dispatches)…")
    us_eager = time_eager(run_step, T, iters=args.iters)

    print(f"Timing graph  (T={T} launches captured, single CPU replay call)…")
    us_graph = time_graph(run_step, T)

    per_step_eager = us_eager / T
    per_step_graph = us_graph / T

    print(f"\n{'='*58}")
    print(f"  T={T}-step sequence  (B={B}, H={H})")
    print(f"{'='*58}")
    print(f"  Eager : {us_eager:9.1f} µs total  |  {per_step_eager:6.2f} µs/step")
    print(f"  Graph : {us_graph:9.1f} µs total  |  {per_step_graph:6.2f} µs/step")
    print(f"  Speedup: {us_eager / us_graph:.2f}×")
    overhead = per_step_eager - per_step_graph
    print(f"\n  CPU launch overhead eliminated per step: ~{overhead:.1f} µs")
    print(f"  GPU-only step time (from single-kernel graph bench): ~28 µs")


if __name__ == "__main__":
    main()
