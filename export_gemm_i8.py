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
    c_dtype=cutlass.Float16,
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

print("=" * 70)
print("All exports complete! Check ./artifacts/ for generated C headers.")
print("=" * 70)
