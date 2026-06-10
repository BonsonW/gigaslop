"""Probe whether quantize performance is BW-limited or overhead-limited."""
import os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize

K = 1024
device = "cuda"
stream = torch.cuda.current_stream()
launch = compile_fp8_per_token_quantize(K=K)

def bench(B, warmup=20, iters=500):
    h_fp16  = torch.randn(B, K, dtype=torch.float16, device=device)
    h_fp8   = torch.zeros(B, K, dtype=torch.uint8, device=device)
    h_scale = torch.zeros(B, dtype=torch.float32, device=device)
    for _ in range(warmup):
        launch(h_fp8, h_scale, h_fp16, B, stream)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record(stream)
    for _ in range(iters):
        launch(h_fp8, h_scale, h_fp16, B, stream)
    t1.record(stream)
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / iters
    nbytes = B * K * 2 + B * K + B * 4
    return us, nbytes / us / 1e3

print(f"{'B':>6}  {'time µs':>9}  {'GB/s':>8}  {'MB data':>8}")
print("-" * 42)
for B in [64, 128, 256, 512, 1024, 2048, 4096]:
    us, gbs = bench(B)
    mb = (B * K * 2 + B * K + B * 4) / 1e6
    print(f"{B:>6}  {us:>9.2f}  {gbs:>8.1f}  {mb:>8.2f}")
