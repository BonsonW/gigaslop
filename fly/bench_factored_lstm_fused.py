"""Correctness + perf for Option A: single-kernel factored LSTM.

compile_fp8_factored_lstm_fused folds the recurrent down-projection hh_down = h@dn_hh INTO
the LSTM kernel (Phase A in-kernel -> LDS), then both up-projections + epilogue in one launch.
One launch floor instead of two (down_proj + factored_lstm). Sweeps h_split for occupancy.

Compares against:
  - two-kernel path: down_proj(n_split=4) + factored_lstm  (the shipped 1.48x)
  - current fused_lstm (the K=H=1024 baseline)
"""
import argparse, os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_unfactored_lstm_gemm import (
    compile_fp8_factored_lstm_fused, compile_fp8_factored_lstm, compile_fp8_fused_lstm,
    preshuffle_b_fp8, preshuffle_b_f16, fp8_quantize_per_channel,
)
from rdna_fp8_factored_gemm import compile_fp8_down_proj, fp8_quantize_scalar

H = 1024; K_HH = 128; R = 128; FH = 4 * H
OUT_SCALE = 1.0 / 448.0


def make_inputs(B, device, seed=42):
    torch.manual_seed(seed)
    h_prev = (torch.rand(B, H, device=device) * 2 - 1) * 0.5
    h_prev_fp8 = (h_prev / OUT_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
    h_prev_q   = h_prev_fp8.float() * OUT_SCALE

    dn_hh = torch.randn(K_HH, H, device=device) * 0.1
    up_hh = torch.randn(FH, K_HH, device=device) * 0.1
    dn_ih = torch.randn(R, H, device=device) * 0.1
    up_ih = torch.randn(FH, R, device=device) * 0.1
    bhh   = torch.randn(FH, device=device) * 0.05
    bih   = torch.randn(FH, device=device) * 0.05
    x     = torch.randn(B, H, device=device) * 0.1
    c     = torch.randn(B, H, device=device) * 0.1

    dn_hh_fp8, dn_hh_scale = fp8_quantize_scalar(dn_hh)
    dn_hh_shuf = preshuffle_b_fp8(dn_hh_fp8.t().contiguous())
    up_hh_shuf = preshuffle_b_f16(up_hh.t().contiguous().to(torch.float16))
    up_ih_shuf = preshuffle_b_f16(up_ih.t().contiguous().to(torch.float16))
    x_down = (x @ dn_ih.t()).to(torch.float16)

    return dict(h_prev_q=h_prev_q, h_prev_fp8=h_prev_fp8.view(torch.uint8),
                dn_hh_fp8=dn_hh_fp8, dn_hh_scale=dn_hh_scale, dn_hh_shuf=dn_hh_shuf,
                up_hh=up_hh, up_ih=up_ih, bhh=bhh, bih=bih, c=c,
                up_hh_shuf=up_hh_shuf, up_ih_shuf=up_ih_shuf, x_down=x_down)


def reference(inp):
    dn_hh_dq = inp["dn_hh_fp8"].float() * inp["dn_hh_scale"].item()
    hh_down = inp["h_prev_q"] @ dn_hh_dq.t()
    x_down  = inp["x_down"].float()
    gates = hh_down @ inp["up_hh"].t() + x_down @ inp["up_ih"].t() + inp["bhh"] + inp["bih"]
    i, f, g, o = gates.chunk(4, dim=1)
    i = (i * 0.2 + 0.5).clamp(0, 1); f = (f * 0.2 + 0.5).clamp(0, 1)
    g = g.clamp(-1, 1); o = (o * 0.2 + 0.5).clamp(0, 1)
    c_new = f * inp["c"] + i * g
    return o * c_new.tanh(), c_new


def run_fused(inp, B, device, h_split):
    lst = compile_fp8_factored_lstm_fused(B=B, H=H, K_hh=K_HH, R=R, h_split=h_split)
    h_fp8 = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w   = inp["c"].clone()
    lst(h_fp8, c_w, inp["h_prev_fp8"], inp["dn_hh_scale"], inp["dn_hh_shuf"],
        inp["up_hh_shuf"], inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
        torch.cuda.current_stream(), B)
    torch.cuda.synchronize()
    return h_fp8.view(torch.float8_e4m3fn).float() * OUT_SCALE, c_w


def verify(B, device, h_split=4):
    print(f"\n{'='*56}\nOption A correctness  B={B}, h_split={h_split}\n{'='*56}")
    inp = make_inputs(B, device)
    h_ref, c_ref = reference(inp)
    h_out, c_out = run_fused(inp, B, device, h_split)
    he = (h_out - h_ref.float()).abs().max().item()
    ce = (c_out - c_ref).abs().max().item()
    ok = he < 0.08 and ce < 0.08
    print(f"  h max err: {he:.5f}   c max err: {ce:.5f}   {'PASS' if ok else 'FAIL'}")
    return ok


def _graph_time(fn, cap=40):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(cap): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(10):
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(20): g.replay()
        t1.record(); torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1) * 1e3 / (20 * cap))
    return best


def benchmark(B, device):
    print(f"\n{'='*56}\nPerf  B={B}  (graph, per-step)\n{'='*56}")
    inp = make_inputs(B, device)
    h_fp8 = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w   = inp["c"].clone()

    # Option A — sweep h_split
    print("  Option A (single kernel):")
    best_a = 1e9
    for hs in (1, 2, 4, 8):
        if H % hs or (H // hs) % 32: continue
        lst = compile_fp8_factored_lstm_fused(B=B, H=H, K_hh=K_HH, R=R, h_split=hs)
        def fn():
            lst(h_fp8, c_w, inp["h_prev_fp8"], inp["dn_hh_scale"], inp["dn_hh_shuf"],
                inp["up_hh_shuf"], inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
                torch.cuda.current_stream(), B)
        us = _graph_time(fn)
        best_a = min(best_a, us)
        print(f"    h_split={hs} ({(B//32)*hs:>4} blocks): {us:6.2f} us/step")

    # two-kernel path (shipped 1.48x)
    dp  = compile_fp8_down_proj(C=H, R=K_HH, n_split=4)
    lst2 = compile_fp8_factored_lstm(B=B, H=H, K_hh=K_HH, R=R)
    hh_down = torch.zeros(B, K_HH, dtype=torch.float16, device=device)
    sx = torch.full((B,), OUT_SCALE, dtype=torch.float32, device=device)
    def fn_two():
        dp(hh_down, inp["h_prev_fp8"], sx, inp["dn_hh_shuf"], inp["dn_hh_scale"], torch.cuda.current_stream(), B)
        lst2(h_fp8, c_w, hh_down, inp["up_hh_shuf"], inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
             torch.cuda.current_stream(), B)
    us_two = _graph_time(fn_two)

    # current fused (K=H baseline)
    fused = compile_fp8_fused_lstm(B=B, H=H, R=R)
    hh = torch.zeros(B, H, dtype=torch.uint8, device=device)
    shh = torch.ones(B, dtype=torch.float32, device=device) * OUT_SCALE
    Wf  = torch.randn(FH, H, device=device) * 0.02
    Wf_fp8, swf = fp8_quantize_per_channel(Wf.t().contiguous()); wf = preshuffle_b_fp8(Wf_fp8)
    c_w2 = inp["c"].clone()
    def fn_fused():
        fused(hh, c_w2, hh, shh, wf, swf, inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
              torch.cuda.current_stream(), B)
    us_fused = _graph_time(fn_fused)

    print(f"  two-kernel (dp+lstm): {us_two:6.2f} us/step")
    print(f"  fused (K=H baseline): {us_fused:6.2f} us/step")
    print(f"  --> Option A best {best_a:.2f} us:  {us_fused/best_a:.2f}x vs fused,  {us_two/best_a:.2f}x vs two-kernel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    if not args.no_verify:
        verify(64, dev); verify(args.batch, dev)
    benchmark(256, dev)
    benchmark(args.batch, dev)


if __name__ == "__main__":
    main()
