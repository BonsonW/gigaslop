import os, torch
os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '1')

B, H = 512, 1024
a = torch.ones(B, H, device='cuda')
c = torch.zeros(B, H, device='cuda')
T = 1024

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for t in range(T):
        c.add_(a)
torch.cuda.synchronize()
print("capture ok")

# Test with progressively larger batches
for batch_size in [16, 17, 18, 19, 20, 25, 30]:
    print(f"Testing {batch_size} replays without sync...", end=' ')
    try:
        for _ in range(batch_size):
            g.replay()
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"CRASH: {e}")
        break
