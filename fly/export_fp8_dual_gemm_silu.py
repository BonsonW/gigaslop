"""Export rdna_fp8_dual_gemm_silu + rdna_fp8_per_token_quantize as HSACO + C headers.

Variant A (two dispatches): dual-GEMM + silu_mul → FP16, then per-token FP8 quantize.

Run with:
    python fly/export_fp8_dual_gemm_silu.py                        # defaults
    python fly/export_fp8_dual_gemm_silu.py --N 4096 --K 512       # single config
    python fly/export_fp8_dual_gemm_silu.py --config fly/export_configs/rdna_fp8_dual_gemm_silu.toml

Outputs per config (in fly/artifacts/):
    rdna_fp8_dual_gemm_silu_N{N}_K{K}.hsaco + .h   (dual-GEMM + silu_mul → FP16)
    rdna_fp8_ptq_K{N}.hsaco + .h                   (FP16 → FP8 per-token quantize)
"""
import argparse
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdna_fp8_dual_gemm_silu import compile_fp8_dual_gemm_silu
from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize


# ── Tile / wave config (mirrors compile_fp8_dual_gemm_silu) ───────────────────

def _compute_config_gemm(M: int, N: int, K: int, tile_m: int = 32) -> dict:
    tile_n = 128  # opt kernel fixes tile_n=128 to avoid dual-accumulator VGPR overflow
    if tile_m >= 128 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 64 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_n >= 256:
        waves_m, waves_n = 1, 2
    elif tile_m >= 64:
        waves_m, waves_n = 2, 1
    elif tile_n >= 128:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1
    return dict(
        tile_m=tile_m,
        tile_n=tile_n,
        waves_m=waves_m,
        waves_n=waves_n,
        threads_per_block=waves_m * waves_n * 32,
        total_blocks=(M // tile_m) * (N // tile_n),
    )


def _load_configs(path: str, default_m: int = 256) -> list[tuple[int, int, int]]:
    if tomllib is None:
        raise RuntimeError(
            "tomllib not available — upgrade to Python 3.11+ or `pip install tomli`"
        )
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [(int(c.get("M", default_m)), int(c["N"]), int(c["K"])) for c in data["configs"]]


# ── Compilation ───────────────────────────────────────────────────────────────

def _compile_and_get_ir_gemm(M: int, N: int, K: int) -> str:
    launcher  = compile_fp8_dual_gemm_silu(M=M, N=N, K=K)
    C         = torch.zeros(M, N, dtype=torch.float16)
    A         = torch.zeros(M, K, dtype=torch.uint8)
    B_gate    = torch.zeros(N // 16, K // 16, 2, 16, 8, dtype=torch.uint8)
    B_up      = torch.zeros(N // 16, K // 16, 2, 16, 8, dtype=torch.uint8)
    scale_a   = torch.zeros(M, dtype=torch.float32)
    scale_bg  = torch.zeros(N, dtype=torch.float32)
    scale_bu  = torch.zeros(N, dtype=torch.float32)
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launcher(C, A, B_gate, B_up, scale_a, scale_bg, scale_bu, 0, M)
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev
    artifacts = list(launcher._mem_cache.values())
    if not artifacts:
        raise RuntimeError("_mem_cache is empty after dual-GEMM compilation")
    return artifacts[0]._ir_text


def _compile_and_get_ir_ptq(M: int, K: int) -> str:
    """K is the row width (= N from the preceding GEMM)."""
    launcher  = compile_fp8_per_token_quantize(K=K)
    out_fp8   = torch.zeros(M, K, dtype=torch.uint8)
    out_scale = torch.zeros(M, dtype=torch.float32)
    inp       = torch.zeros(M, K, dtype=torch.float16)
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launcher(out_fp8, out_scale, inp, M, 0)
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev
    artifacts = list(launcher._mem_cache.values())
    if not artifacts:
        raise RuntimeError("_mem_cache is empty after PTQ compilation")
    return artifacts[0]._ir_text


# ── MLIR binary extraction ────────────────────────────────────────────────────

def _decode_mlir_bin(ir_text: str, start: int) -> bytes:
    marker = 'bin = "'
    pos = ir_text.find(marker, start)
    if pos == -1:
        raise ValueError("'bin = \"' not found in IR text after position %d" % start)
    i = pos + len(marker)
    result = bytearray()
    while i < len(ir_text):
        c = ir_text[i]
        if c == '\\':
            nxt = ir_text[i + 1]
            if nxt == '\\':
                result.append(ord('\\'))
                i += 2
            elif nxt == '"':
                result.append(ord('"'))
                i += 2
            else:
                result.append(int(ir_text[i + 1: i + 3], 16))
                i += 3
        elif c == '"':
            break
        else:
            result.append(ord(c))
            i += 1
    return bytes(result)


def _extract_hsaco(ir_text: str, prefer_no_wave64: bool = True) -> bytes:
    if prefer_no_wave64:
        anchor = ir_text.find("no_wave64")
        if anchor == -1:
            print("Warning: no_wave64 variant not found, falling back to first object")
            anchor = 0
    else:
        anchor = 0
    return _decode_mlir_bin(ir_text, anchor)


def _find_kernel_name(ir_text: str) -> str:
    m = re.search(r'#gpu\.kernel_metadata<"([^"]+)"', ir_text)
    return m.group(1) if m else "kernel_0"


# ── C header templates ────────────────────────────────────────────────────────

_GEMM_HEADER = """\
/* Auto-generated by export_fp8_dual_gemm_silu.py — do not edit.
 *
 * HSACO ELF for {kernel_name} (gfx1201, wave32/RDNA4).
 *
 * Computes: C[M,N] = silu(A @ B_gate) * (A @ B_up)
 * where A, B_gate, B_up are fp8_e4m3fn; output C is fp16.
 *
 * Fixed dimensions: M={M}, N={N}, K={K}
 *   A:           fp8_e4m3fn  [{M}, {K}]
 *   B_gate/B_up: fp8_e4m3fn  [{N_div16}, {K_div16}, 2, 16, 8]  — preshuffled
 *   C:           fp16        [{M}, {N}]
 *   scale_a:     f32         [{M}]   — per-token activation scale
 *   scale_b_*:   f32         [{N}]   — per-channel weight scale
 *
 * Grid:  {total_blocks} x 1 x 1  (for M={M})
 * Block: {threads_per_block} x 1 x 1
 */
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define FP8_DUAL_SILU_N                  {N}
#define FP8_DUAL_SILU_K                  {K}
#define FP8_DUAL_SILU_TILE_M             {tile_m}
#define FP8_DUAL_SILU_TILE_N             {tile_n}
#define FP8_DUAL_SILU_THREADS_PER_BLOCK  {threads_per_block}

typedef struct {{
    hipModule_t   module;
    hipFunction_t func;
}} fp8_dual_silu_Module_t;

static inline int fp8_dual_silu_Module_Load(fp8_dual_silu_Module_t *m, const char *hsaco_path) {{
    hipError_t err = hipModuleLoad(&m->module, hsaco_path);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_dual_silu: hipModuleLoad(%s): %s\\n", hsaco_path, hipGetErrorString(err));
        return (int)err;
    }}
    err = hipModuleGetFunction(&m->func, m->module, "{kernel_name}");
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_dual_silu: hipModuleGetFunction: %s\\n", hipGetErrorString(err));
        (void)hipModuleUnload(m->module);
        return (int)err;
    }}
    return 0;
}}

static inline void fp8_dual_silu_Module_Unload(fp8_dual_silu_Module_t *m) {{
    if (m->module) (void)hipModuleUnload(m->module);
    m->module = NULL;
    m->func   = NULL;
}}

/* Launch the dual-GEMM+silu_mul kernel.
 * M must be a runtime multiple of FP8_DUAL_SILU_TILE_M; N and K are fixed at export time.
 * d_B_gate_shuf and d_B_up_shuf must be in preshuffled layout [{N_div16},{K_div16},2,16,8].
 * Output d_C is fp16 [{M},{N}]; feed to fp8_ptq_wrapper to get FP8. */
static inline int fp8_dual_silu_wrapper(
        const fp8_dual_silu_Module_t *m,
        hipDeviceptr_t d_C,            /* fp16  [M, {N}] */
        hipDeviceptr_t d_A,            /* fp8   [M, {K}] */
        hipDeviceptr_t d_B_gate_shuf,  /* fp8   [{N_div16},{K_div16},2,16,8] */
        hipDeviceptr_t d_B_up_shuf,    /* fp8   [{N_div16},{K_div16},2,16,8] */
        hipDeviceptr_t d_scale_a,      /* f32   [M] */
        hipDeviceptr_t d_scale_b_gate, /* f32   [{N}] */
        hipDeviceptr_t d_scale_b_up,   /* f32   [{N}] */
        int32_t M,
        hipStream_t stream) {{
    if (M % FP8_DUAL_SILU_TILE_M != 0) {{
        fprintf(stderr, "fp8_dual_silu: M=%d not divisible by tile_m=%d\\n", M, FP8_DUAL_SILU_TILE_M);
        return -1;
    }}
    int32_t grid_m       = M / FP8_DUAL_SILU_TILE_M;
    unsigned int total_blocks = (unsigned int)grid_m * ({N} / FP8_DUAL_SILU_TILE_N);
    void *args[] = {{ &d_C, &d_A, &d_B_gate_shuf, &d_B_up_shuf,
                     &d_scale_a, &d_scale_b_gate, &d_scale_b_up, &grid_m }};
    hipError_t err = hipModuleLaunchKernel(
            m->func,
            total_blocks, 1, 1,
            FP8_DUAL_SILU_THREADS_PER_BLOCK, 1, 1,
            0, stream, args, NULL);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_dual_silu: hipModuleLaunchKernel: %s\\n", hipGetErrorString(err));
        return (int)err;
    }}
    return 0;
}}
"""

_PTQ_HEADER = """\
/* Auto-generated by export_fp8_dual_gemm_silu.py — do not edit.
 *
 * HSACO ELF for {kernel_name} (gfx1201, wave32/RDNA4).
 *
 * Per-token FP16 → FP8 quantize.
 * Computes per-row amax, writes scale_a[row] = amax/448.0 and quantizes each row.
 *
 * Fixed dimension: K={K} (number of elements per row)
 *   inp:      fp16  [M, {K}]
 *   out_fp8:  fp8_e4m3fn bytes [M, {K}]
 *   out_scale: f32  [M]  — per-token scale (amax / 448.0); compatible with scale_a of fp8_gemm
 *
 * Grid:  M x 1 x 1  (runtime M)
 * Block: {block_threads} x 1 x 1
 */
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define FP8_PTQ_K             {K}
#define FP8_PTQ_BLOCK_THREADS {block_threads}

typedef struct {{
    hipModule_t   module;
    hipFunction_t func;
}} fp8_ptq_Module_t;

static inline int fp8_ptq_Module_Load(fp8_ptq_Module_t *m, const char *hsaco_path) {{
    hipError_t err = hipModuleLoad(&m->module, hsaco_path);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_ptq: hipModuleLoad(%s): %s\\n", hsaco_path, hipGetErrorString(err));
        return (int)err;
    }}
    err = hipModuleGetFunction(&m->func, m->module, "{kernel_name}");
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_ptq: hipModuleGetFunction: %s\\n", hipGetErrorString(err));
        (void)hipModuleUnload(m->module);
        return (int)err;
    }}
    return 0;
}}

static inline void fp8_ptq_Module_Unload(fp8_ptq_Module_t *m) {{
    if (m->module) (void)hipModuleUnload(m->module);
    m->module = NULL;
    m->func   = NULL;
}}

/* Launch the per-token FP8 quantize kernel.
 * M is the number of rows (tokens); K={K} is fixed at export time.
 * out_scale[M] is written as (amax / 448.0) and is compatible with scale_a in fp8_gemm. */
static inline int fp8_ptq_wrapper(
        const fp8_ptq_Module_t *m,
        hipDeviceptr_t d_out_fp8,   /* fp8 bytes [M, {K}] */
        hipDeviceptr_t d_out_scale, /* f32        [M] */
        hipDeviceptr_t d_inp,       /* fp16       [M, {K}] */
        int32_t M,
        hipStream_t stream) {{
    void *args[] = {{ &d_out_fp8, &d_out_scale, &d_inp }};
    hipError_t err = hipModuleLaunchKernel(
            m->func,
            (unsigned int)M, 1, 1,
            FP8_PTQ_BLOCK_THREADS, 1, 1,
            0, stream, args, NULL);
    if (err != hipSuccess) {{
        fprintf(stderr, "fp8_ptq: hipModuleLaunchKernel: %s\\n", hipGetErrorString(err));
        return (int)err;
    }}
    return 0;
}}
"""

def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _export_gemm(M: int, N: int, K: int, artifacts_dir: str) -> None:
    cfg  = _compute_config_gemm(M, N, K)
    name = f"rdna_fp8_dual_gemm_silu_N{N}_K{K}_TM{cfg['tile_m']}_TN{cfg['tile_n']}"
    print(f"\n[GEMM M={M}, N={N}, K={K}]  tile_m={cfg['tile_m']}, tile_n={cfg['tile_n']}, "
          f"waves=({cfg['waves_m']},{cfg['waves_n']}), "
          f"threads={cfg['threads_per_block']}, blocks={cfg['total_blocks']}")

    ir_text     = _compile_and_get_ir_gemm(M, N, K)
    kernel_name = _find_kernel_name(ir_text)
    print(f"  Kernel: {kernel_name}  IR: {len(ir_text):,} chars")

    hsaco = _extract_hsaco(ir_text, prefer_no_wave64=True)
    if hsaco[:4] != b"\x7fELF":
        raise RuntimeError(f"Expected ELF magic, got: {hsaco[:8].hex()}")

    hsaco_path  = os.path.join(artifacts_dir, f"{name}.hsaco")
    header_path = os.path.join(artifacts_dir, f"{name}.h")
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(hsaco_path, "wb") as f:
        f.write(hsaco)

    header = _GEMM_HEADER.format(
        kernel_name=kernel_name,
        M=M, N=N, K=K,
        N_div16=N // 16, K_div16=K // 16,
        tile_m=cfg["tile_m"], tile_n=cfg["tile_n"],
        threads_per_block=cfg["threads_per_block"],
        total_blocks=cfg["total_blocks"],
    )
    _write_text(header_path, header)
    print(f"  Wrote {len(hsaco):,} bytes → {hsaco_path}")
    print(f"  Wrote header → {header_path}")


def _export_ptq(M: int, N: int, artifacts_dir: str) -> None:
    """N is the GEMM output width = PTQ row width."""
    from rdna_fp8_per_token_quantize import BLOCK_THREADS
    name = f"rdna_fp8_ptq_K{N}"
    print(f"\n[PTQ K={N}]  block_threads={BLOCK_THREADS}, grid=(M,1,1)")

    ir_text     = _compile_and_get_ir_ptq(M, N)
    kernel_name = _find_kernel_name(ir_text)
    print(f"  Kernel: {kernel_name}  IR: {len(ir_text):,} chars")

    hsaco = _extract_hsaco(ir_text, prefer_no_wave64=True)
    if hsaco[:4] != b"\x7fELF":
        raise RuntimeError(f"Expected ELF magic, got: {hsaco[:8].hex()}")

    hsaco_path  = os.path.join(artifacts_dir, f"{name}.hsaco")
    header_path = os.path.join(artifacts_dir, f"{name}.h")
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(hsaco_path, "wb") as f:
        f.write(hsaco)

    header = _PTQ_HEADER.format(
        kernel_name=kernel_name,
        K=N,
        block_threads=BLOCK_THREADS,
    )
    _write_text(header_path, header)
    print(f"  Wrote {len(hsaco):,} bytes → {hsaco_path}")
    print(f"  Wrote header → {header_path}")


def _export_one(M: int, N: int, K: int, artifacts_dir: str) -> None:
    _export_gemm(M, N, K, artifacts_dir)
    _export_ptq(M, N, artifacts_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export fp8 dual-GEMM+silu_mul and per-token quantize kernels as HSACO + C headers."
    )
    parser.add_argument("--config", metavar="FILE",
                        help="TOML config file with a list of M,N,K configs to batch-export")
    parser.add_argument("--M", type=int, default=256, help="Batch dimension (default: 256)")
    parser.add_argument("--N", type=int, default=2048, help="Output (inter_dim) dimension (default: 2048)")
    parser.add_argument("--K", type=int, default=512,  help="Input (model_dim) dimension (default: 512)")
    args = parser.parse_args()

    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

    if args.config:
        configs = _load_configs(args.config, default_m=args.M)
        print(f"Batch export: {len(configs)} config(s) from {args.config}")
        for M, N, K in configs:
            _export_one(M, N, K, artifacts_dir)
        print(f"\nDone. {len(configs)} config(s) exported to {artifacts_dir}/")
    else:
        _export_one(args.M, args.N, args.K, artifacts_dir)
