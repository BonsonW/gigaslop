"""Export rdna_fp8_preshuffle_gemm as a single self-contained C header.

The HSACO binary is embedded as a byte array — no separate .hsaco file needed.

Run with:
    python export_fp8_gemm_embedded.py                                    # N=8192, K=6144 (defaults)
    python export_fp8_gemm_embedded.py --N 4096 --K 3072                  # single config
    python export_fp8_gemm_embedded.py --config export_configs/rdna_fp8_gemm.toml  # batch

Outputs (single):  artifacts/rdna_fp8_gemm.h
Outputs (batch):   artifacts/rdna_fp8_gemm_N{N}_K{K}.h  (one per config)
"""
import argparse
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_fp8_gemm import (
    _compute_config, _compile_and_get_ir, _extract_hsaco, _find_kernel_name,
    _load_configs, _COMMON_COMMENT, _format_hex, _fmt, _write_text,
)

# ── C header template ─────────────────────────────────────────────────────────

_EMBEDDED_HEADER_TEMPLATE = _COMMON_COMMENT + """\
 * Build: hipcc -o main_rdna_fp8_gemm main_rdna_fp8_gemm.cpp
 */
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ── Compiled-in constants (N, K, tile sizes fixed at export time) ── */
#define FP8_GEMM_N                  {N}
#define FP8_GEMM_K                  {K}
#define FP8_GEMM_TILE_M             {tile_m}
#define FP8_GEMM_TILE_N             {tile_n}
#define FP8_GEMM_THREADS_PER_BLOCK  {threads_per_block}

/* ── Embedded HSACO binary ({size} bytes) ── */
static const uint8_t fp8_gemm_hsaco[] = {{
{hex_array}
}};

/* ── Module handle ── */
typedef struct {{
    hipModule_t   module;
    hipFunction_t func;
}} fp8_gemm_Module_t;

static inline int fp8_gemm_Module_Load(fp8_gemm_Module_t *m) {{
    hipError_t err = hipModuleLoadData(&m->module, fp8_gemm_hsaco);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_gemm: hipModuleLoadData: %s\\n", hipGetErrorString(err));
        return (int)err;
    }}
    err = hipModuleGetFunction(&m->func, m->module, "{kernel_name}");
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_gemm: hipModuleGetFunction: %s\\n", hipGetErrorString(err));
        (void)hipModuleUnload(m->module);
        return (int)err;
    }}
    return 0;
}}

static inline void fp8_gemm_Module_Unload(fp8_gemm_Module_t *m) {{
    if (m->module) (void)hipModuleUnload(m->module);
    m->module = NULL;
    m->func   = NULL;
}}

/* Launch the kernel.
 * M must be a runtime multiple of FP8_GEMM_TILE_M; N and K are fixed at export time.
 * d_B_shuf must already be in preshuffled layout [{N_div16},{K_div16},2,16,8].
 * Returns 0 on success, non-zero HIP error code otherwise. */
static inline int fp8_gemm_wrapper(
        const fp8_gemm_Module_t *m,
        hipDeviceptr_t d_C,       /* bf16  [M, {N}] */
        hipDeviceptr_t d_A,       /* fp8   [M, {K}] */
        hipDeviceptr_t d_B_shuf,  /* fp8   [{N_div16},{K_div16},2,16,8] */
        hipDeviceptr_t d_scale_a, /* f32   [M] */
        hipDeviceptr_t d_scale_b, /* f32   [{N}] */
        int32_t M,
        hipStream_t stream) {{
    if (M % FP8_GEMM_TILE_M != 0) {{
        fprintf(stderr, "fp8_gemm: M=%d not divisible by tile_m=%d\\n", M, FP8_GEMM_TILE_M);
        return -1;
    }}
    int32_t grid_m = M / FP8_GEMM_TILE_M;
    unsigned int total_blocks = (unsigned int)grid_m * ({N} / FP8_GEMM_TILE_N);
    void *args[] = {{ &d_C, &d_A, &d_B_shuf, &d_scale_a, &d_scale_b, &grid_m }};
    hipError_t err = hipModuleLaunchKernel(
            m->func,
            total_blocks, 1, 1,
            FP8_GEMM_THREADS_PER_BLOCK, 1, 1,
            0, stream, args, NULL);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_gemm: hipModuleLaunchKernel: %s\\n", hipGetErrorString(err));
        return (int)err;
    }}
    return 0;
}}
"""


def _generate_embedded(hsaco: bytes, kernel_name: str, artifacts_dir: str,
                       M: int, N: int, K: int, cfg: dict,
                       name: str = "rdna_fp8_gemm") -> None:
    output_path = os.path.join(artifacts_dir, f"{name}.h")
    _write_text(output_path, _fmt(_EMBEDDED_HEADER_TEMPLATE, hsaco, kernel_name, M, N, K, cfg))
    print(f"  Wrote {len(hsaco):,} bytes of HSACO (embedded) → {output_path}")


def _export_one(M: int, N: int, K: int, artifacts_dir: str) -> None:
    cfg = _compute_config(M, N, K)
    name = f"rdna_fp8_gemm_N{N}_K{K}_TM{cfg['tile_m']}_TN{cfg['tile_n']}"
    print(f"\n[M={M}, N={N}, K={K}]  tile_m={cfg['tile_m']}, tile_n={cfg['tile_n']}, "
          f"waves=({cfg['waves_m']},{cfg['waves_n']}), "
          f"threads={cfg['threads_per_block']}, blocks={cfg['total_blocks']}")

    ir_text = _compile_and_get_ir(M, N, K)
    kernel_name = _find_kernel_name(ir_text)
    print(f"  Kernel: {kernel_name}  IR: {len(ir_text):,} chars")

    hsaco = _extract_hsaco(ir_text, prefer_no_wave64=True)
    if hsaco[:4] != b"\x7fELF":
        raise RuntimeError(f"Expected ELF magic, got: {hsaco[:8].hex()}")

    _generate_embedded(hsaco, kernel_name, artifacts_dir, M, N, K, cfg, name=name)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fp8 GEMM kernel as embedded C header.")
    parser.add_argument("--config", metavar="FILE",
                        help="TOML config file with a list of M,N,K configs to batch-export")
    parser.add_argument("--M", type=int, default=32,  help="Batch dimension for tile selection (default: 32)")
    parser.add_argument("--N", type=int, default=8192, help="Output dimension (default: 8192)")
    parser.add_argument("--K", type=int, default=6144, help="Reduction dimension (default: 6144)")
    args = parser.parse_args()

    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

    if args.config:
        configs = _load_configs(args.config, default_m=args.M)
        print(f"Batch export: {len(configs)} config(s) from {args.config}")
        for M, N, K in configs:
            _export_one(M, N, K, artifacts_dir)
        print(f"\nDone. {len(configs)} kernel(s) exported to {artifacts_dir}/")
    else:
        _export_one(args.M, args.N, args.K, artifacts_dir)
