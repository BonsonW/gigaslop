"""Correctness + benchmark: unfactored LSTM GEMM vs factored kernel."""
import argparse, os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_unfactored_lstm_gemm import (
    compile_fp8_unfactored_lstm_gemm, make_w_fused, make_ih_t_interleaved,
    fp8_quantize_per_token, fp8_quantize_per_channel, preshuffle_b_fp8,
)
from rdna_fp8_factored_lstm_gemm import compile_fp8_factored_lstm_gemm, preshuffle_b_f16
from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize

H = 1024; R = 128; FH = 4 * H
PEAK_FP8_TOPS = 383.0; PEAK_BW_GBS = 640.0


def lstm_step_ref(hh, dn_w, up_w, bias, ih_t, c):
    y     = hh @ dn_w.t()
    gates = y @ up_w.t() + bias + ih_t
    i_l, f_l, g_l, o_l = gates.chunk(4, dim=1)
    i_a = (i_l * 0.2 + 0.5).clamp(0, 1)
    f_a = (f_l * 0.2 + 0.5).clamp(0, 1)
    g_a = g_l.clamp(-1, 1)
    o_a = (o_l * 0.2 + 0.5).clamp(0, 1)
    c_new = f_a * c + i_a * g_a
    return o_a * c_new.tanh(), c_new


def make_inputs(B, device, seed=42):
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

    up_weight_rn = up_weight_f32.t().contiguous()      # [R, 4H]
    up_fp8, scale_up = fp8_quantize_per_channel(up_weight_rn)
    up_shuf = preshuffle_b_fp8(up_fp8)
    up_shuf_f16 = preshuffle_b_f16(up_weight_rn.to(torch.float16))

    # Unfactored: fuse weights offline; permute ih_t to [B, H, 4] for vec4 loads
    wf_shuf, scale_wf = make_w_fused(dn_weight_f32, up_weight_f32)
    ih_t_f16          = ih_t_f32.to(torch.float16)            # [B, 4H] — for factored
    ih_t_interleaved  = make_ih_t_interleaved(ih_t_f16)       # [B, H, 4] — for unfactored

    return dict(
        hh_fp8=hh_fp8, scale_hh=scale_hh,
        dn_shuf=dn_shuf, scale_dn=scale_dn,
        up_shuf=up_shuf, scale_up=scale_up, up_shuf_f16=up_shuf_f16,
        wf_shuf=wf_shuf, scale_wf=scale_wf,
        up_bias=up_bias_f32, ih_t_f16=ih_t_f16, ih_t_interleaved=ih_t_interleaved,
        c_inout=c_f32.clone(),
        hh_f32=hh_f32, dn_weight_f32=dn_weight_f32, up_weight_f32=up_weight_f32,
        up_bias_f32=up_bias_f32, ih_t_f32=ih_t_f32, c_ref=c_f32.clone(),
    )


def verify(B, device):
    print(f"\n{'='*60}\nCorrectness B={B}, H={H}, R={R}\n{'='*60}")
    inp = make_inputs(B, device)
    stream = torch.cuda.current_stream()

    h_ref, c_ref = lstm_step_ref(
        inp["hh_f32"], inp["dn_weight_f32"], inp["up_weight_f32"],
        inp["up_bias_f32"], inp["ih_t_f32"], inp["c_ref"],
    )

    launcher = compile_fp8_unfactored_lstm_gemm(B=B, H=H)
    h_fp8 = torch.zeros(B, H, dtype=torch.uint8, device=device)
    # Clone c so the in-place cell update doesn't corrupt inp["c_inout"] for the factored check.
    c_inout = inp["c_inout"].clone()
    # Input hh keeps its own per-token scale; only the OUTPUT uses the fixed 1/448 scale.
    launcher(h_fp8, c_inout, inp["hh_fp8"], inp["scale_hh"],
             inp["wf_shuf"], inp["scale_wf"], inp["up_bias"], inp["ih_t_interleaved"],
             stream, B)
    torch.cuda.synchronize()

    # Fused kernel writes fp8 with a fixed scale of 1/448; dequantize to compare.
    h_out = h_fp8.view(torch.float8_e4m3fn).float() * (1.0 / 448.0)
    h_err = (h_out - h_ref.float()).abs()
    c_err = (c_inout - c_ref).abs()
    h_ok  = h_err.max().item() < 0.08
    c_ok  = c_err.max().item() < 0.08
    print(f"[unfactored]  h_new max err: {h_err.max().item():.5f}  {'PASS' if h_ok else 'FAIL'}")
    print(f"[unfactored]  c_new max err: {c_err.max().item():.5f}  {'PASS' if c_ok else 'FAIL'}")

    # ── Factored kernel: also fused fp8 output (fixed scale 1/448) ──────────────
    fac_ok = True
    for nh in (1, 4, 8):
        if (H // 32) % nh != 0:
            continue
        fac = compile_fp8_factored_lstm_gemm(B=B, H=H, R=R, nh_per_block=nh)
        h_fp8_f = torch.zeros(B, H, dtype=torch.uint8, device=device)
        c_f     = inp["c_inout"].clone()
        fac(h_fp8_f, c_f, inp["hh_fp8"], inp["scale_hh"],
            inp["dn_shuf"], inp["scale_dn"], inp["up_shuf_f16"],
            inp["up_bias"], inp["ih_t_f16"], stream, B)
        torch.cuda.synchronize()
        h_out_f = h_fp8_f.view(torch.float8_e4m3fn).float() * (1.0 / 448.0)
        h_e = (h_out_f - h_ref.float()).abs().max().item()
        c_e = (c_f - c_ref).abs().max().item()
        ok  = h_e < 0.08 and c_e < 0.08
        fac_ok = fac_ok and ok
        print(f"[factored nh={nh}]  h max err: {h_e:.5f}  c max err: {c_e:.5f}  {'PASS' if ok else 'FAIL'}")
        if nh == 1:
            h_fp8_ref_bits = h_fp8_f.clone()
        else:
            same = torch.equal(h_fp8_f, h_fp8_ref_bits)
            print(f"[factored nh={nh}]  bitwise-identical to nh=1: {'YES' if same else 'NO'}")
            fac_ok = fac_ok and same

    return h_ok and c_ok and fac_ok


def benchmark(B, device, warmup=20, iters=500):
    print(f"\n{'='*60}\nBenchmark  B={B}, H={H}, R={R},  iters={iters}\n{'='*60}")
    inp    = make_inputs(B, device)
    stream = torch.cuda.current_stream()

    fac_launcher  = compile_fp8_factored_lstm_gemm(B=B, H=H, R=R)
    unf_launcher  = compile_fp8_unfactored_lstm_gemm(B=B, H=H)
    quant_launcher = compile_fp8_per_token_quantize(K=H)

    h_fp16_out  = torch.zeros(B, H,  dtype=torch.float16, device=device)
    h_fp8_out   = torch.zeros(B, H,  dtype=torch.uint8,   device=device)
    h_scale_out = torch.zeros(B,     dtype=torch.float32, device=device)
    # Persistent c tensors: updated in-place each step, matching real inference.
    # Clone is NOT used in the timing loop to avoid artificial cache pollution.
    c_fac = inp["c_inout"].clone()
    c_unf = inp["c_inout"].clone()

    # NOTE: read the current stream dynamically so the same closure works both for
    # eager timing and for CUDA-graph capture (capture swaps in its own stream).
    def run_factored():
        fac_launcher(h_fp8_out, c_fac,
                     inp["hh_fp8"], inp["scale_hh"],
                     inp["dn_shuf"], inp["scale_dn"],
                     inp["up_shuf_f16"],
                     inp["up_bias"], inp["ih_t_f16"], torch.cuda.current_stream(), B)

    def run_unfactored():
        unf_launcher(h_fp8_out, c_unf,
                     inp["hh_fp8"], inp["scale_hh"],
                     inp["wf_shuf"], inp["scale_wf"],
                     inp["up_bias"], inp["ih_t_interleaved"], torch.cuda.current_stream(), B)

    def time_fn(fn, nw, ni):
        # Eager wall-time per step. NOTE: dominated by FlyDSL's CPU launch overhead
        # (~12-20 µs/call) and noisy (±several µs) — this is NOT the kernel's GPU time.
        for _ in range(nw): fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record(stream)
            for _ in range(ni): fn()
            t1.record(stream)
            torch.cuda.synchronize()
            best = min(best, t0.elapsed_time(t1) * 1e3 / ni)
        return best

    def time_fn_graph(fn, capture=50, rounds=10):
        # True kernel GPU time: capture `capture` launches in a CUDA graph and replay.
        # Removes CPU launch overhead so TFLOPS/BW reflect the kernel, not dispatch.
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
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(20): g.replay()
            t1.record(); torch.cuda.synchronize()
            best = min(best, t0.elapsed_time(t1) * 1e3 / (20 * capture))
        return best

    total_flops_fac = 2 * B * H * R + 2 * B * R * FH          # factored
    total_flops_unf = 2 * B * H * FH                            # unfactored

    # weights(1B) + hh(1B) + ih_t(2B f16) + c read&write(8B f32) + h out(1B fp8)
    fac_bytes = (H*R + R*FH)*1 + (B*H + B*FH*2 + B*H*8 + B*H*1)
    unf_bytes = (H*FH)*1       + (B*H + B*FH*2 + B*H*8 + B*H*1)

    us_fac       = time_fn(run_factored,  warmup, iters)
    us_unf       = time_fn(run_unfactored, warmup, iters)
    us_fac_graph = time_fn_graph(run_factored)
    us_unf_graph = time_fn_graph(run_unfactored)

    def print_stats(label, us_graph, us_eager, flops, nbytes):
        # TFLOPS/BW computed from the GRAPH time (true kernel GPU throughput).
        s = us_graph * 1e-6
        print(f"\n  [{label}]")
        print(f"    Time/step : {us_graph:.2f} µs graph  |  {us_eager:.2f} µs eager (launch-bound)")
        print(f"               (T=1024 graph: {us_graph*1024/1e3:.1f} ms)")
        print(f"    TFLOPS    : {flops/s/1e12:.1f}  ({flops/s/1e12/PEAK_FP8_TOPS*100:.1f}% of peak)")
        print(f"    BW GB/s   : {nbytes/s/1e9:.1f}  ({nbytes/s/1e9/PEAK_BW_GBS*100:.1f}% of peak)")
        print(f"    Arith. int: {flops/nbytes:.1f} FLOP/byte")
        wt_kb = (H*R + R*FH) if label.startswith("factored") else H*FH
        print(f"    Weight KB : {wt_kb/1e3:.0f}")

    print_stats("factored  (Phase1+Phase2 fused)", us_fac_graph, us_fac, total_flops_fac, fac_bytes)
    print_stats("unfactored (single GEMM fused)",  us_unf_graph, us_unf, total_flops_unf, unf_bytes)
    print(f"\n  Unfactored speedup over factored (graph): {us_fac_graph/us_unf_graph:.2f}×")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    device = "cuda"
    B = args.batch
    assert B % 32 == 0
    print(f"Device: {torch.cuda.get_device_name()}")

    if not args.no_verify:
        ok = verify(B=64, device=device)
        print(f"\nUnfactored verify: {'PASS ✓' if ok else 'FAIL ✗'}")

    benchmark(B=B, device=device, iters=args.iters)

if __name__ == "__main__":
    main()
