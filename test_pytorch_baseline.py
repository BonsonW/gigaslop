import os, torch
os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '1')

# T=1024 trivial CUDA graph with pure PyTorch operations
B, H = 512, 1024
a = torch.ones(B, H, device='cuda')
b = torch.ones(B, H, device='cuda')
c = torch.zeros(B, H, device='cuda')
T = 1024

# Capture T=1024 in-place add operations
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for t in range(T):
        c.add_(a)  # trivial op
torch.cuda.synchronize()
print("capture ok")

# Replay 200 times in batches of 20 without sync
for i in range(10):
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    print(f"round {i}: ok")
