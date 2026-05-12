import itertools
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from ampere_gemm_i8_quant import TensorOpGemmI8
import torch

M, N, K, L = 524288, 4096, 512, 1

def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_dtype = cutlass_torch.dtype(dtype)
    if dtype.signed:
        torch_tensor = torch.randint(-2, 3, shape, dtype=torch_dtype)
    else:
        torch_tensor = torch.randint(0, 5, shape, dtype=torch_dtype)
    torch_tensor = torch_tensor.permute(permute_order).cuda()
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
        .mark_compact_shape_dynamic(
            mode=(1 if not is_mode0_major else 0),
            stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
            divisibility=(128 // dtype.width),
        )
    )
    return cute_tensor, torch_tensor

configs = [
    # (bm, bN, atom_mnk,    num_stages, use_k32)
    (128, 128, (2, 2, 1),   3,          True),   # your current baseline
    (128, 128, (2, 2, 1),   4,          True),   # +1 stage
    (128, 128, (2, 2, 1),   5,          True),   # +2 stages
    (128, 128, (2, 4, 1),   3,          True),   # more warps
    (128, 128, (2, 4, 1),   4,          True),   # more warps + stage
    (128, 256, (2, 4, 1),   3,          True),   # wider N tile
    (128, 256, (2, 4, 1),   4,          True),
    (64,  128, (2, 2, 1),   3,          True),   # smaller bM → more CTAs/SM
    (64,  128, (2, 4, 1),   3,          True),
    (128, 128, (2, 2, 1),   3,          False),  # k16 vs k32
    (128, 128, (2, 4, 1),   3,          False),
]

results = []

for bm, bN, atom_mnk, num_stages, use_k32 in configs:
    bK = 64
    M_pad = ((M + bm  - 1) // bm)  * bm
    N_pad = ((N + bN  - 1) // bN)  * bN
    K_pad = ((K + bK  - 1) // bK)  * bK

    try:
        mA, a_t = create_and_permute_tensor(L, M_pad, K_pad, False, cutlass.Int8)
        mB, b_t = create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
        mC, c_t = create_and_permute_tensor(L, M_pad, N_pad, False, cutlass.Int32)

        # zero padding
        a_t[M:, :, :] = 0;  a_t[:, K:, :] = 0
        b_t[N:, :, :] = 0;  b_t[:, K:, :] = 0

        # dummy scales
        import torch
        sa = from_dlpack(torch.ones(M_pad, L, dtype=torch.float32, device='cuda').contiguous(), assumed_align=4)
        sb = from_dlpack(torch.ones(N_pad, L, dtype=torch.float32, device='cuda').contiguous(), assumed_align=4)

        kernel = TensorOpGemmI8(
            cutlass.Int8, cutlass.Int8, cutlass.Int32, cutlass.Int32,
            atom_mnk, use_k32, bm, bN, num_stages
        )
        # patch num_stages before compile
        kernel.num_stages = num_stages
        kernel.cta_tiler  = (bm, bN, bK)
        kernel.bM, kernel.bN, kernel.bK = bm, bN, bK

        compiled = cute.compile(kernel, mA, mB, mC, sa, sb)

        avg_us = cute.testing.benchmark(
            compiled,
            kernel_arguments=cute.testing.JitArguments(mA, mB, mC, sa, sb),
            warmup_iterations=3,
            iterations=50,
        )

        tops = (2 * M * N * K) / (avg_us * 1e-6) / 1e12
        print(f"bm={bm:3d} bN={bN:3d} atom={str(atom_mnk):10s} "
              f"stages={num_stages} k32={use_k32}  "
              f"{avg_us:7.1f}us  {tops:.3f} TOPS")
        results.append((avg_us, bm, bN, atom_mnk, num_stages, use_k32))

    except Exception as e:
        print(f"bm={bm:3d} bN={bN:3d} atom={str(atom_mnk):10s} "
              f"stages={num_stages} k32={use_k32}  FAILED: {e}")

results.sort()
print("\n=== Top 3 configs ===")
for avg_us, bm, bN, atom_mnk, num_stages, use_k32 in results[:3]:
    tops = (2 * M * N * K) / (avg_us * 1e-6) / 1e12
    print(f"bm={bm} bN={bN} atom={atom_mnk} stages={num_stages} "
          f"k32={use_k32}  {avg_us:.1f}us  {tops:.3f} TOPS")