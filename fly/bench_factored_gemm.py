"""Correctness + performance benchmark for the fused fp8 low-rank GEMM.

Computes ih[M,N] = (x_fp8 @ dn_fp8) @ up_f16 + bias  — the LSTM input projection,
fused through LDS (down-proj fp8 → x_down f16 in LDS → up-proj f16 → bias → f16).

Reference: ih_ref = (x @ dn.t()) @ up.t() + bias.

  python fly/bench_factored_gemm.py                  # correctness + perf sweep
  python fly/bench_factored_gemm.py --no-verify       # perf only
  python fly/bench_factored_gemm.py --M 131072        # add a custom M to the sweep
"""
import argparse, os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_factored_gemm import (
    compile_fp8_factored_gemm, preshuffle_b_fp8, preshuffle_b_f16,
    fp8_quantize_per_token, fp8_quantize_scalar,
)

C = 1024; R = 128; N = 4096
PEAK_FP8_TOPS = 383.0; PEAK_BW_GBS = 640.0


# ── Inputs / weights (weights built once, reused across M) ─────────────────────

def make_weights(device, seed=0):
    torch.manual_seed(seed)
    dn   = torch.randn(R, C, device=device) * 0.1   # [R, C]  (dn_weight_ih layout)
    up   = torch.randn(N, R, device=device) * 0.1   # [N, R]  (up_weight_ih layout)
    bias = torch.randn(N,    device=device) * 0.05
    dn_fp8, scale_dn = fp8_quantize_scalar(dn.t().contiguous())   # B[C,R] scalar-scaled fp8
    dn_shuf = preshuffle_b_fp8(dn_fp8)
    up_shuf = preshuffle_b_f16(up.t().contiguous().to(torch.float16))  # B[R,N] f16
    return dict(dn=dn, up=up, bias=bias, dn_shuf=dn_shuf, scale_dn=scale_dn, up_shuf=up_shuf)


def make_x(M, device, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(M, C, device=device, generator=g) * 0.1


def run_kernel(launcher, x, w, device):
    M = x.shape[0]
    x_fp8, scale_x = fp8_quantize_per_token(x)
    ih = torch.zeros(M, N, dtype=torch.float16, device=device)
    launcher(ih, x_fp8, scale_x, w["dn_shuf"], w["scale_dn"], w["up_shuf"], w["bias"],
             torch.cuda.current_stream(), M)
    torch.cuda.synchronize()
    return ih


def ref_rows(x_rows, w):
    return (x_rows @ w["dn"].t()) @ w["up"].t() + w["bias"]


# ── Correctness (full ref for small M; row-subsets for large M) ────────────────

def verify(launcher, M, device):
    x = make_x(M, device)
    w = make_weights(device)
    ih = run_kernel(launcher, x, w, device).float()
    if M <= 32768:
        ref = ref_rows(x, w)
        err = (ih - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        ok = rel < 0.08
        print(f"  M={M:7d} (M%32={M%32:2d}): max_err {err:.4f}  rel {rel*100:4.1f}%  {'PASS' if ok else 'FAIL'}")
    else:
        # subset rows incl. the i32-offset overflow region (g_row*N > 2^31 at M>262144)
        worst = 0.0
        for r0 in (0, M // 2, M - 64):
            rows = slice(r0, r0 + 64)
            e = (ih[rows] - ref_rows(x[rows], w)).abs().max().item()
            worst = max(worst, e)
        ok = worst < 0.12
        print(f"  M={M:7d} (M%32={M%32:2d}): subset max_err {worst:.4f}  {'PASS' if ok else 'FAIL'}  (rows 0/mid/last)")
    return ok


# ── Timing ─────────────────────────────────────────────────────────────────────

def time_eager(fn, iters):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): fn()
        t1.record(); torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1) / iters)
    return best  # ms


def time_graph(fn, capture=10, rounds=6):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(capture): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(4): g.replay()
        t1.record(); torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1) / (4 * capture))
    return best  # ms


def benchmark(launcher, M, device):
    w = make_weights(device)
    x = make_x(M, device)
    x_fp8, scale_x = fp8_quantize_per_token(x)
    ih = torch.zeros(M, N, dtype=torch.float16, device=device)
    st = torch.cuda.current_stream()
    fn = lambda: launcher(ih, x_fp8, scale_x, w["dn_shuf"], w["scale_dn"],
                          w["up_shuf"], w["bias"], torch.cuda.current_stream(), M)

    ms_eager = time_eager(fn, iters=50 if M <= 65536 else 20)
    ms_graph = time_graph(fn)

    def run_torch():
        return (x @ w["dn"].t()) @ w["up"].t() + w["bias"]
    for _ in range(3): run_torch()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(10): run_torch()
    t1.record(); torch.cuda.synchronize()
    ms_torch = t0.elapsed_time(t1) / 10

    flops = 2 * M * (C * R + R * N)
    # algorithmic bytes: x read (fp8) + ih write (f16) + weights (amortized, L2)
    nbytes = M * C * 1 + M * N * 2 + (C * R) * 1 + (R * N) * 2 + N * 4
    s = ms_graph * 1e-3
    print(f"  M={M:7d} | graph {ms_graph:7.3f} ms  eager {ms_eager:7.3f} ms  torch {ms_torch:8.2f} ms"
          f"  | {flops/s/1e12:5.0f} TFLOPS ({flops/s/1e12/PEAK_FP8_TOPS*100:4.1f}%)"
          f"  {nbytes/s/1e9:5.0f} GB/s ({nbytes/s/1e9/PEAK_BW_GBS*100:4.1f}%)"
          f"  AI {flops/nbytes:5.1f}  | {ms_torch/ms_graph:5.1f}x torch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--M", type=int, default=None, help="add a custom M to the sweep")
    args = p.parse_args()
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}   C={C} R={R} N={N}")

    launcher = compile_fp8_factored_gemm(C=C, R=R, N=N)

    if not args.no_verify:
        print(f"\n{'='*92}\nCorrectness (incl. M%32!=0 tail + M>262144 i32-offset region)\n{'='*92}")
        for M in (64, 512, 8192, 8200, 262144, 524288):
            verify(launcher, M, dev)

    print(f"\n{'='*92}\nPerformance  (graph = true kernel GPU time; eager incl. launch overhead)\n{'='*92}")
    sweep = [8192, 32768, 131072, 262144, 524288]
    if args.M and args.M not in sweep:
        sweep.append(args.M); sweep.sort()
    for M in sweep:
        benchmark(launcher, M, dev)


if __name__ == "__main__":
    main()
