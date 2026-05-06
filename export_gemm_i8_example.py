#!/usr/bin/env python3
"""
Example script demonstrating how to export TensorOpGemmI8 with custom parameters.

This script exports an INT8 GEMM kernel to C code using the CuTe DSL.
The exported kernel can be integrated into C/C++ projects.
"""

import cutlass
from ampere_gemm_i8_quant_rmem import export_tensor_op_gemm_i8

# Example 1: Standard INT8 GEMM with Int8 x Int8 -> Int32
print("=" * 70)
print("Example 1: Standard INT8 GEMM (Int8 x Int8 -> Int32)")
print("=" * 70)
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    file_path="./artifacts",
    file_name="gemm_i8_standard",
    function_prefix="gemm_i8_std",
    bm=128,
    bn=128,
    num_stages=3,
)
print()

# Example 2: Uint8 x Int8 GEMM (mixed signedness)
print("=" * 70)
print("Example 2: Uint8 x Int8 GEMM (mixed signedness)")
print("=" * 70)
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Uint8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 4, 1),
    file_path="./artifacts",
    file_name="gemm_u8i8_mixed",
    function_prefix="gemm_u8i8",
    bm=128,
    bn=128,
    num_stages=3,
)
print()

# Example 3: Smaller tile with K32 MMA instruction
print("=" * 70)
print("Example 3: Smaller tile (64x64) with K32 MMA instruction")
print("=" * 70)
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    file_path="./artifacts",
    file_name="gemm_i8_k32_small",
    function_prefix="gemm_i8_k32",
    use_k32=True,
    bm=64,
    bn=64,
    num_stages=3,
)
print()

# Example 4: Larger atom layout (8 warps) for high compute
print("=" * 70)
print("Example 4: Larger atom layout (2x4x1 = 8 warps) for high throughput")
print("=" * 70)
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Float16,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 4, 1),
    file_path="./artifacts",
    file_name="gemm_i8_high_throughput",
    function_prefix="gemm_i8_ht",
    bm=128,
    bn=128,
    num_stages=3,
)
print()

print("=" * 70)
print("All exports complete! Check ./artifacts/ for generated C headers.")
print("=" * 70)
