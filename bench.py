import sys
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm import TensorOpGemmI8

def quantize_tensor(t, scale=None):
    """Quantize float tensor to int8 with optional per-tensor scaling."""
    if scale is None:
        scale = t.abs().max() / 127.0
    t_int8 = torch.clamp(t / scale, -128, 127).to(torch.int8)
    return t_int8, scale

def dequantize_tensor(t_int8, scale):
    """Dequantize int8 tensor back to float using scale."""
    return t_int8.to(torch.float32) * scale

# Simple test params - use larger size to match tile boundaries
batch_size = 1
M = 128
N = 128
K = 128

# Create float input tensors
A_float = torch.randn((M, K, batch_size), dtype=torch.float32, device='cuda')
B_float = torch.randn((N, K, batch_size), dtype=torch.float32, device='cuda')

print(f"A_float shape: {A_float.shape}, B_float shape: {B_float.shape}")

# Quantize to int8
A_int8, A_scale = quantize_tensor(A_float)
B_int8, B_scale = quantize_tensor(B_float)

print(f"A_scale: {A_scale}, B_scale: {B_scale}")
print(f"A_int8 shape: {A_int8.shape}, B_int8 shape: {B_int8.shape}")

# Create output buffer
C = torch.zeros((M, N, batch_size), dtype=torch.int32, device='cuda')

print(f"C shape: {C.shape}")

# Convert to CUTE tensors
mA = from_dlpack(A_int8, assumed_align=16)
mB = from_dlpack(B_int8, assumed_align=16)
mC = from_dlpack(C, assumed_align=16)

# Setup and compile kernel
tensor_op_gemm = TensorOpGemmI8(
    cutlass.Int8,
    cutlass.Int8,
    cutlass.Int32,
    cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    use_k32=False,
    bm=128,
)

print('=== Compiling ampere_gemm kernel ===')
compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)

print('=== Running GEMM === ')
with torch.autograd.profiler.profile(use_device='cuda') as prof:
    compiled_gemm(mA, mB, mC)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Dequantize output: C_int32 * A_scale * B_scale
C_dequantized = dequantize_tensor(C, A_scale * B_scale)

# Float reference: A_float @ B_float^T
A_ref = A_float[:, :, 0]  # (M, K)
B_ref = B_float[:, :, 0]  # (N, K)
C_ref = torch.matmul(A_ref, B_ref.t())  # (M, N)

print(f"\nC_dequantized shape: {C_dequantized.shape}")
print(f"C_ref shape: {C_ref.shape}")
print(f"\nC_dequantized[:3, :3]:\n{C_dequantized[:3, :3, 0]}")
print(f"C_ref[:3, :3]:\n{C_ref[:3, :3]}")

# Compare
if torch.allclose(C_dequantized[:, :, 0], C_ref, atol=2.0, rtol=1e-1):
    print("\n✓ Results match!")
else:
    print("\n✗ Results differ")
    print(f"Max diff: {(C_dequantized[:, :, 0] - C_ref).abs().max()}")
    print(f"Mean diff: {(C_dequantized[:, :, 0] - C_ref).abs().mean()}")
