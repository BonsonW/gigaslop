import os, torch
os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '1')

B, H = 512, 1024
a = torch.ones(B, H, device='cuda')
b = torch.ones(B, H, device='cuda')
c = torch.zeros(B, H, device='cuda')
T = 1024

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for t in range(T):
        c.add_(a)
torch.cuda.synchronize()
print("capture ok")

# Test with smaller batches
for i in range(20):  # 20 iterations
    for _ in range(15):  # 15 replays each = 300 total
        g.replay()
    torch.cuda.synchronize()
    print(f"round {i}: ok (15 replays)")
print("SUCCESS: completed all batches")
