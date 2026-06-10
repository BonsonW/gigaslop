"""Correctness + benchmark test for the per-token quantize kernel."""
import os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize

B, K = 512, 1024
device = "cuda"
stream = torch.cuda.current_stream()
torch.manual_seed(42)

h_fp16  = torch.randn(B, K, dtype=torch.float16, device=device)
h_fp8   = torch.zeros(B, K, dtype=torch.uint8, device=device)
h_scale = torch.zeros(B, dtype=torch.float32, device=device)

launch = compile_fp8_per_token_quantize(K=K)
launch(h_fp8, h_scale, h_fp16, B, stream)
torch.cuda.synchronize()

# Reference: match kernel arithmetic (multiply by inv_sc, not divide by scale)
amax      = h_fp16.float().abs().amax(dim=-1).clamp(min=1e-12)
inv_scale = 448.0 / amax
h_ref     = (h_fp16.float() * inv_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)

scale_ref = amax / 448.0
scale_err = (h_scale - scale_ref).abs().max().item()

diff = (h_fp8.view(torch.float8_e4m3fn).float() - h_ref.float()).abs()
fp8_err = diff.max().item()
fp8_wrong = (diff > 0.5).sum().item()

print(f"Scale err : {scale_err:.3e}")
print(f"FP8 max err: {fp8_err:.2f}  (elements > 0.5: {fp8_wrong} / {B*K})")

# Benchmark
WARMUP, ITERS = 20, 500
for _ in range(WARMUP):
    launch(h_fp8, h_scale, h_fp16, B, stream)
torch.cuda.synchronize()
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record(stream)
for _ in range(ITERS):
    launch(h_fp8, h_scale, h_fp16, B, stream)
t1.record(stream)
torch.cuda.synchronize()
us = t0.elapsed_time(t1) * 1e3 / ITERS
bw = (B * K * 2 + B * K + B * 4) / us / 1e6  # read fp16 + write fp8 + write scale
print(f"Time: {us:.2f} µs  ({bw:.1f} GB/s)")
