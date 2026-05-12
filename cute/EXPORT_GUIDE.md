# TensorOpGemmI8 Export Guide

This guide explains how to export `TensorOpGemmI8` kernels to C code for integration into C/C++ projects.

## Overview

The `export_tensor_op_gemm_i8()` function allows you to compile a CuTe DSL INT8 GEMM kernel with custom parameters and export it to C/C++ header files. This enables:

- **AOT (Ahead-of-Time) Compilation**: Compile kernels offline and integrate into C/C++ applications
- **Custom Parameters**: Create specialized kernels for specific hardware and performance requirements
- **Type Safety**: Leverage CuTe's type system for compile-time correctness checking

## Function Signature

```python
def export_tensor_op_gemm_i8(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    file_path: str = "./artifacts",
    file_name: str = "tensor_op_gemm_i8",
    function_prefix: str = "tensor_op_gemm_i8",
    use_k32: bool = False,
    bm: int = 128,
    bn: int = 128,
    num_stages: int = 3,
    m_size: int = 128,
    n_size: int = 128,
    k_size: int = 128,
    l_size: int = 1,
) -> None
```

## Parameters

### Data Types
- **a_dtype**: Input matrix A type (`cutlass.Int8` or `cutlass.Uint8`)
- **b_dtype**: Input matrix B type (`cutlass.Int8` or `cutlass.Uint8`)
- **c_dtype**: Output matrix C type (`cutlass.Int32`)
- **acc_dtype**: Accumulator type (`cutlass.Int32`)

**Note**: Mixed signedness is supported: `Uint8 x Int8` is allowed, but `Uint8 x Uint8` is not.

### Tile and MMA Configuration
- **atom_layout_mnk**: Warp arrangement tuple `(atom_m, atom_n, atom_k)`
  - Common values: `(2,2,1)` for 4 warps, `(2,4,1)` for 8 warps
  - Affects thread count, register pressure, and occupancy
  
- **bm, bn**: Tile dimensions (block tile size for M and N)
  - Default: 128x128
  - Must be divisible by `atom_m * mma_m` and `atom_n * mma_n`
  - mma_m=16, mma_n=8 (from hardware spec)
  
- **use_k32**: Use K=32 MMA instruction instead of K=16
  - K=16: Default, lower latency, smaller K dimension coverage
  - K=32: Higher throughput for larger K dimensions
  
- **num_stages**: Pipeline stages for multi-stage optimization
  - Minimum: 3 (supports async copy overlap)
  - Higher values improve latency hiding at memory cost

### Output Configuration
- **file_path**: Directory to save exported C headers
- **file_name**: Base name for generated files
- **function_prefix**: Prefix for generated C function names

### Symbolic Dimensions
- **m_size, n_size, k_size, l_size**: Nominal dimension sizes
  - These are informational and used for layout optimization
  - Actual runtime dimensions can differ (handled via symbolic shapes)

## Usage Examples

### Basic Usage

```python
import cutlass
from ampere_gemm_i8_quant_rmem import export_tensor_op_gemm_i8

# Export a standard INT8 GEMM
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    file_path="./kernels",
    file_name="my_gemm",
)
```

### Mixed Signedness (Uint8 x Int8)

```python
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Uint8,  # A must be unsigned for mixed
    b_dtype=cutlass.Int8,   # B must be signed
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 4, 1),  # 8 warps for higher throughput
)
```

### High-Throughput Configuration

```python
# Large atom layout (8 warps) for high compute density
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 4, 1),  # 2x4 = 8 warps
    bm=128,
    bn=128,
    num_stages=3,
)
```

### K32 Optimization for Larger K Dimensions

```python
# Use K=32 MMA for efficient large-K operations
export_tensor_op_gemm_i8(
    a_dtype=cutlass.Int8,
    b_dtype=cutlass.Int8,
    c_dtype=cutlass.Int32,
    acc_dtype=cutlass.Int32,
    atom_layout_mnk=(2, 2, 1),
    use_k32=True,  # Use 16x8x32 MMA instead of 16x8x16
    bm=128,
    bn=128,
)
```

## Output Files

The export function generates a C header file like:

```
artifacts/
└── my_gemm.h
    ├── Type definitions for kernels
    ├── Module loading/unloading functions
    └── Wrapper functions for kernel execution
```

## Compilation and Execution

The generated C code includes:

1. **Kernel Module Management**
   ```c
   void my_gemm_Kernel_Module_Load(my_gemm_Kernel_Module_t *module);
   void my_gemm_Kernel_Module_Unload(my_gemm_Kernel_Module_t *module);
   ```

2. **Tensor Wrappers**
   ```c
   typedef struct {
       void *data;
       int32_t dynamic_shapes[1];  // For dynamic dimensions
   } my_gemm_Tensor_a_t;
   ```

3. **Kernel Execution**
   ```c
   int32_t my_gemm_wrapper(
       my_gemm_Kernel_Module_t *module,
       my_gemm_Tensor_a_t *a,
       my_gemm_Tensor_b_t *b,
       my_gemm_Tensor_c_t *c,
       my_gemm_Tensor_scale_a_t *scale_a,
       my_gemm_Tensor_scale_b_t *scale_b,
       cudaStream_t stream
   );
   ```

## Running the Examples

Execute the example script to generate multiple kernel variants:

```bash
python export_gemm_i8_example.py
```

This creates optimized kernels for:
- Standard INT8 operations
- Mixed signedness (Uint8 x Int8)
- K32 MMA instruction optimization
- High-throughput configurations (8 warps)

## Hardware Requirements

- **GPU Architecture**: NVIDIA Ampere (A100, A10, A30, RTX 30-series, etc.)
- **CUDA Compute Capability**: 8.0 or higher
- **CuTe DSL**: Requires NVIDIA CUTLASS library with Python bindings

## Advanced Tuning

The `TensorOpGemmI8` class selects default tile sizes and atom layouts based on M dimension:

| M Range | bm  | Atom Layout | Thread Count | Use Case |
|---------|-----|-------------|--------------|----------|
| ≤16     | 16  | (1, 2, 1)   | 64           | Small matrices |
| ≤32     | 32  | (2, 2, 1)   | 128          | Small-medium |
| ≤64     | 64  | (2, 4, 1)   | 256          | Medium |
| ≤256    | 128 | (2, 4, 1)   | 256          | Large |
| >256    | 128 | (2, 2, 1)   | 128          | Very large M |

For custom tuning, override `bm` and `atom_layout_mnk` parameters in the export function.

## Troubleshooting

### Mixed Signedness Error
```
ValueError: A=Uint8, B=Int8 is not supported...
```
**Solution**: Ensure A is unsigned and B is signed. Swap if needed.

### Alignment Error
```
Column-major INT8 does not meet ldmatrix 128-bit alignment
```
**Note**: INT8 GEMM requires K-major (row-major) inputs. This is enforced by design.

### Compilation Failures
Ensure:
- CUTLASS library with Python bindings is installed
- CUDA 12.0+ with appropriate GPU driver
- Compatible CuTe DSL version
