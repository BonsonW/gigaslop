"""Correctness + perf for the fused LSTM (Option A): hidden (hh@W_fused, fp8) +
input (x_down@up_ih, f16) folded into one kernel — no ih[B,4H] tensor.

Reference: gates = hh@W_fused + x_down@up_ih.t() + bias_hh + bias_ih ; LSTM step.
Equivalence: must match the 2-kernel path (factored ih + unfactored LSTM) within fp8 noise.
"""
import argparse, os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_unfactored_lstm_gemm import (
    compile_fp8_fused_lstm, compile_fp8_unfactored_lstm_gemm,
    make_w_fused, make_ih_t_interleaved, preshuffle_b_fp8, preshuffle_b_f16,
    fp8_quantize_per_token, fp8_quantize_per_channel,
)

H = 1024; R = 128; FH = 4 * H
PEAK_FP8_TOPS = 383.0; PEAK_BW_GBS = 640.0


def make_inputs(B, device, seed=42):
    torch.manual_seed(seed)
    hh   = torch.randn(B, H,  device=device) * 0.1
    Wf   = torch.randn(FH, H, device=device) * 0.02   # W_fused [4H, H] (hidden weight)
    dn   = torch.randn(R, H,  device=device) * 0.1    # dn_ih [R, C=H]
    up   = torch.randn(FH, R, device=device) * 0.1    # up_ih [4H, R]
    bhh  = torch.randn(FH,    device=device) * 0.05
    bih  = torch.randn(FH,    device=device) * 0.05
    x    = torch.randn(B, H,  device=device) * 0.1    # input (C=H)
    c    = torch.randn(B, H,  device=device) * 0.1

    hh_fp8, scale_hh = fp8_quantize_per_token(hh)
    # W_fused per-channel fp8 + preshuffle  ([H,4H] for K=H GEMM)
    Wf_hk = Wf.t().contiguous()                       # [H, 4H]
    Wf_fp8, scale_wf = fp8_quantize_per_channel(Wf_hk)
    wf_shuf = preshuffle_b_fp8(Wf_fp8)
    # up_ih f16 preshuffle ([R,4H] for K=R GEMM)
    up_rn = up.t().contiguous().to(torch.float16)     # [R, 4H]
    up_shuf = preshuffle_b_f16(up_rn)
    # x_down = x @ dn.t()  -> [B, R]  (precomputed, f16 here for the test)
    x_down = (x @ dn.t()).to(torch.float16)

    return dict(hh=hh, Wf=Wf, dn=dn, up=up, bhh=bhh, bih=bih, x=x, c=c,
                hh_fp8=hh_fp8, scale_hh=scale_hh, wf_shuf=wf_shuf, scale_wf=scale_wf,
                up_shuf=up_shuf, x_down=x_down)


def reference(inp, device):
    hh, Wf, up, bhh, bih, c = inp["hh"], inp["Wf"], inp["up"], inp["bhh"], inp["bih"], inp["c"]
    x_down = inp["x_down"].float()
    gates = hh @ Wf.t() + x_down @ up.t() + bhh + bih
    i, f, g, o = gates.chunk(4, dim=1)
    i = (i * 0.2 + 0.5).clamp(0, 1); f = (f * 0.2 + 0.5).clamp(0, 1)
    g = g.clamp(-1, 1); o = (o * 0.2 + 0.5).clamp(0, 1)
    c_new = f * c + i * g
    return o * c_new.tanh(), c_new


def run_fused(inp, B, device):
    fac = compile_fp8_fused_lstm(B=B, H=H, R=R)
    h_fp8 = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w = inp["c"].clone()
    fac(h_fp8, c_w, inp["hh_fp8"], inp["scale_hh"], inp["wf_shuf"], inp["scale_wf"],
        inp["bhh"], inp["x_down"], inp["up_shuf"], inp["bih"],
        torch.cuda.current_stream(), B)
    torch.cuda.synchronize()
    return h_fp8.view(torch.float8_e4m3fn).float() * (1.0 / 448.0), c_w


def verify(B, device):
    print(f"\n{'='*56}\nFused LSTM correctness  B={B}, H={H}, R={R}\n{'='*56}")
    inp = make_inputs(B, device)
    h_ref, c_ref = reference(inp, device)
    h_out, c_out = run_fused(inp, B, device)
    he = (h_out - h_ref.float()).abs().max().item()
    ce = (c_out - c_ref).abs().max().item()
    ok = he < 0.08 and ce < 0.08
    print(f"  h max err: {he:.5f}   c max err: {ce:.5f}   {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(B, device):
    print(f"\n{'='*56}\nPerf  B={B}  (graph, per-step)\n{'='*56}")
    inp = make_inputs(B, device)
    fused = compile_fp8_fused_lstm(B=B, H=H, R=R)
    h_fp8 = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w = inp["c"].clone()
    def fn():
        fused(h_fp8, c_w, inp["hh_fp8"], inp["scale_hh"], inp["wf_shuf"], inp["scale_wf"],
              inp["bhh"], inp["x_down"], inp["up_shuf"], inp["bih"],
              torch.cuda.current_stream(), B)
    def gt(fn, cap=50):
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
    us = gt(fn)
    s = us * 1e-6
    # FLOPs: hidden hh@W_fused (2·B·H·4H) + input x_down@up (2·B·R·4H)
    flops = 2 * B * H * FH + 2 * B * R * FH
    # Bytes: weights W_fused fp8 (H·4H) + up_ih f16 (R·4H·2) + hh fp8 (B·H)
    #        + x_down f16 (B·R·2) + c read&write f32 (B·H·8) + h out fp8 (B·H)
    nbytes = (H*FH) + (R*FH*2) + (B*H) + (B*R*2) + (B*H*8) + (B*H)
    print(f"  fused LSTM step: {us:.2f} µs/step  (no ih tensor; reads x_down[B,R] = {B*R*2/1024:.0f} KB)")
    print(f"    TFLOPS    : {flops/s/1e12:.1f}  ({flops/s/1e12/PEAK_FP8_TOPS*100:.1f}% of peak)")
    print(f"    BW GB/s   : {nbytes/s/1e9:.1f}  ({nbytes/s/1e9/PEAK_BW_GBS*100:.1f}% of peak)")
    print(f"    Arith. int: {flops/nbytes:.1f} FLOP/byte")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    if not args.no_verify:
        verify(64, dev)
        verify(args.batch, dev)
    benchmark(args.batch, dev)


if __name__ == "__main__":
    main()
