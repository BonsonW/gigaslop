"""Correctness + perf for the FACTORED LSTM: BOTH hidden and input projections kept
low-rank. Replaces the K=H=1024 hh@W_fused GEMM with hh_down@up_hh (K=K_hh=128).

Per step the real pipeline runs TWO kernels:
  1. hh_down = h_prev_fp8 @ dn_hh   (compile_fp8_down_proj, recurrent)
  2. gates  = hh_down@up_hh + x_down@up_ih + bias_hh + bias_ih -> LSTM  (compile_fp8_factored_lstm)

This bench checks correctness vs a torch reference and times the per-step pipeline
(down_proj + factored_lstm) under CUDA graph.
"""
import argparse, os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_factored_lstm import (
    compile_fp8_factored_lstm,
    preshuffle_b_fp8, preshuffle_b_f16,
    fp8_quantize_per_token, fp8_quantize_per_channel,
)
from rdna_fp8_factored_gemm import compile_fp8_down_proj, fp8_quantize_scalar

H = 1024; K_HH = 128; R = 128; FH = 4 * H
PEAK_FP8_TOPS = 383.0; PEAK_BW_GBS = 640.0
OUT_SCALE = 1.0 / 448.0


def make_inputs(B, device, seed=42):
    torch.manual_seed(seed)
    # h_prev is the recurrent hidden state: fp8 with fixed 1/448 scale (LSTM output convention)
    h_prev = (torch.rand(B, H, device=device) * 2 - 1) * 0.5          # in [-0.5,0.5], bounded like tanh
    h_prev_fp8 = (h_prev / OUT_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
    h_prev_q   = h_prev_fp8.float() * OUT_SCALE                        # what the kernels actually see

    dn_hh = torch.randn(K_HH, H, device=device) * 0.1                  # [K_hh, H]
    up_hh = torch.randn(FH, K_HH, device=device) * 0.1                 # [4H, K_hh]
    dn_ih = torch.randn(R, H, device=device) * 0.1                     # [R, H]
    up_ih = torch.randn(FH, R, device=device) * 0.1                    # [4H, R]
    bhh   = torch.randn(FH, device=device) * 0.05
    bih   = torch.randn(FH, device=device) * 0.05
    x     = torch.randn(B, H, device=device) * 0.1
    c     = torch.randn(B, H, device=device) * 0.1

    # --- down_proj inputs for hh_down = h_prev @ dn_hh^T ---
    dn_hh_fp8, dn_hh_scale = fp8_quantize_scalar(dn_hh)               # scalar fp8
    dn_hh_shuf = preshuffle_b_fp8(dn_hh_fp8.t().contiguous())          # B[K=H, N=K_hh] -> [K_hh/16,H/16,2,16,8]
    scale_x_hh = torch.full((B,), OUT_SCALE, dtype=torch.float32, device=device)

    # --- up-projection weights (f16 preshuffled, gate-major [4H, rank]) ---
    up_hh_shuf = preshuffle_b_f16(up_hh.t().contiguous().to(torch.float16))   # [K_hh, 4H] -> [4H/16,K_hh/16,...]
    up_ih_shuf = preshuffle_b_f16(up_ih.t().contiguous().to(torch.float16))   # [R, 4H]   -> [4H/16,R/16,...]

    # x_down precomputed (f16), like the real pipeline
    x_down = (x @ dn_ih.t()).to(torch.float16)

    return dict(h_prev_q=h_prev_q, h_prev_fp8=h_prev_fp8.view(torch.uint8),
                dn_hh=dn_hh, up_hh=up_hh, dn_ih=dn_ih, up_ih=up_ih, bhh=bhh, bih=bih, x=x, c=c,
                dn_hh_fp8=dn_hh_fp8, dn_hh_scale=dn_hh_scale, dn_hh_shuf=dn_hh_shuf, scale_x_hh=scale_x_hh,
                up_hh_shuf=up_hh_shuf, up_ih_shuf=up_ih_shuf, x_down=x_down)


def reference(inp):
    # hh_down computed exactly as the down_proj kernel does: fp8 h_prev @ scalar-fp8 dn_hh.
    # Use the DEQUANTIZED dn_hh so this isolates kernel correctness from fp8 noise.
    dn_hh_dq = inp["dn_hh_fp8"].float() * inp["dn_hh_scale"].item()   # [K_hh, H]
    hh_down = inp["h_prev_q"] @ dn_hh_dq.t()                          # [B, K_hh]
    x_down  = inp["x_down"].float()
    gates = hh_down @ inp["up_hh"].t() + x_down @ inp["up_ih"].t() + inp["bhh"] + inp["bih"]
    i, f, g, o = gates.chunk(4, dim=1)
    i = (i * 0.2 + 0.5).clamp(0, 1); f = (f * 0.2 + 0.5).clamp(0, 1)
    g = g.clamp(-1, 1); o = (o * 0.2 + 0.5).clamp(0, 1)
    c_new = f * inp["c"] + i * g
    return o * c_new.tanh(), c_new


def run_factored(inp, B, device):
    dp  = compile_fp8_down_proj(C=H, R=K_HH, n_split=4)
    lst = compile_fp8_factored_lstm(B=B, H=H, K_hh=K_HH, R=R)
    hh_down = torch.zeros(B, K_HH, dtype=torch.float16, device=device)
    h_fp8   = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w     = inp["c"].clone()
    dp(hh_down, inp["h_prev_fp8"], inp["scale_x_hh"], inp["dn_hh_shuf"], inp["dn_hh_scale"],
       torch.cuda.current_stream(), B)
    lst(h_fp8, c_w, hh_down, inp["up_hh_shuf"], inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
        torch.cuda.current_stream(), B)
    torch.cuda.synchronize()
    return h_fp8.view(torch.float8_e4m3fn).float() * OUT_SCALE, c_w, hh_down


def verify(B, device):
    print(f"\n{'='*56}\nFactored LSTM correctness  B={B}, H={H}, K_hh={K_HH}, R={R}\n{'='*56}")
    inp = make_inputs(B, device)
    h_ref, c_ref = reference(inp)
    h_out, c_out, hh_down = run_factored(inp, B, device)
    # sanity: down_proj hh_down vs torch
    hh_ref = inp["h_prev_q"] @ inp["dn_hh"].t()
    hd_err = (hh_down.float() - hh_ref).abs().max().item()
    he = (h_out - h_ref.float()).abs().max().item()
    ce = (c_out - c_ref).abs().max().item()
    ok = he < 0.08 and ce < 0.08
    print(f"  hh_down max err: {hd_err:.5f}")
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
    dp  = compile_fp8_down_proj(C=H, R=K_HH, n_split=4)
    lst = compile_fp8_factored_lstm(B=B, H=H, K_hh=K_HH, R=R)
    hh_down = torch.zeros(B, K_HH, dtype=torch.float16, device=device)
    h_fp8   = torch.zeros(B, H, dtype=torch.uint8, device=device)
    c_w     = inp["c"].clone()

    def fn_dp():
        dp(hh_down, inp["h_prev_fp8"], inp["scale_x_hh"], inp["dn_hh_shuf"], inp["dn_hh_scale"],
           torch.cuda.current_stream(), B)
    def fn_lstm():
        lst(h_fp8, c_w, hh_down, inp["up_hh_shuf"], inp["bhh"], inp["x_down"], inp["up_ih_shuf"], inp["bih"],
            torch.cuda.current_stream(), B)
    def fn_both():
        fn_dp(); fn_lstm()

    us_dp   = _graph_time(fn_dp)
    us_lstm = _graph_time(fn_lstm)
    us_both = _graph_time(fn_both)
    s = us_both * 1e-6
    # FLOPs: hh_down (2BHK_hh) + hidden up (2B*K_hh*4H) + input up (2B*R*4H)
    flops = 2*B*H*K_HH + 2*B*K_HH*FH + 2*B*R*FH
    print(f"  down_proj  (hh_down): {us_dp:6.2f} us/step")
    print(f"  factored_lstm       : {us_lstm:6.2f} us/step")
    print(f"  TOTAL  (dp + lstm)  : {us_both:6.2f} us/step")
    print(f"    TFLOPS  : {flops/s/1e12:.1f}  ({flops/s/1e12/PEAK_FP8_TOPS*100:.1f}% of fp8 peak)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    if not args.no_verify:
        verify(64, dev)
        verify(args.batch, dev)
    benchmark(256, dev)
    benchmark(args.batch, dev)


if __name__ == "__main__":
    main()
