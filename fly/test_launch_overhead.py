"""Measure CPU-side launch overhead of FlyDSL @flyc.jit functions."""
import os, sys, time
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

# Warm up CPU path
for _ in range(5):
    launch(h_fp8, h_scale, h_fp16, B, stream)
torch.cuda.synchronize()

N = 500

# Wall-clock time (includes CPU overhead + GPU execution)
t_wall_s = time.perf_counter()
for _ in range(N):
    launch(h_fp8, h_scale, h_fp16, B, stream)
torch.cuda.synchronize()
t_wall_e = time.perf_counter()
wall_us = (t_wall_e - t_wall_s) * 1e6 / N

# GPU timer (from stream, excludes CPU stall between submissions)
for _ in range(5):
    launch(h_fp8, h_scale, h_fp16, B, stream)
torch.cuda.synchronize()
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record(stream)
for _ in range(N):
    launch(h_fp8, h_scale, h_fp16, B, stream)
t1.record(stream)
torch.cuda.synchronize()
gpu_us = t0.elapsed_time(t1) * 1e3 / N

print(f"Wall-clock per call : {wall_us:.2f} µs")
print(f"GPU timer per call  : {gpu_us:.2f} µs")
print(f"CPU overhead (approx): {max(0, wall_us - gpu_us):.2f} µs  (wall - gpu)")
print()
print("If wall ≈ gpu: GPU is the bottleneck.")
print("If wall >> gpu: CPU launch overhead is the bottleneck.")
