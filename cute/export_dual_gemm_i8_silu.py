#!/usr/bin/env python3
"""Export the dual INT8 GEMM + SiLU kernel to C headers (AOT).

Mirrors the FlyDSL TOML-config-driven export setup (fly/export_fp8_*.py), but uses
our CuteDSL `export_to_c` path.

Run:
    pyvenv/bin/python cute/export_dual_gemm_i8_silu.py                         # defaults
    pyvenv/bin/python cute/export_dual_gemm_i8_silu.py --N 2048 --K 512        # single config
    pyvenv/bin/python cute/export_dual_gemm_i8_silu.py \\
        --config cute/export_configs/dual_gemm_i8_silu.toml                    # batch sweep

Output (in cute/artifacts/):
    gemm_i8_dual_silu_N{N}_K{K}.h
"""
import argparse
import os
import sys

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampere_dual_gemm_i8_silu import export_dual_gemm_i8_silu


def _parse_simple_toml(path):
    """Minimal parser for our `[[configs]]` + `key = int` configs (Py<3.11 fallback)."""
    configs = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line == "[[configs]]":
                configs.append({})
            elif "=" in line and configs:
                key, val = (s.strip() for s in line.split("=", 1))
                configs[-1][key] = int(val)
    return {"configs": configs}


def _load_configs(path, default_m=256):
    data = tomllib.load(open(path, "rb")) if tomllib is not None else _parse_simple_toml(path)
    return [(int(c.get("M", default_m)), int(c["N"]), int(c["K"])) for c in data["configs"]]


def _export_one(M, N, K, bm, bn, num_stages, atom_layout, artifacts_dir):
    name = f"gemm_i8_dual_silu_N{N}_K{K}"
    print(f"\n[M={M} N={N} K={K}]  -> {name}.h")
    export_dual_gemm_i8_silu(
        atom_layout_mnk=atom_layout,
        file_path=artifacts_dir,
        file_name=name,
        function_prefix=name,
        bm=bm, bn=bn, num_stages=num_stages,
        m_size=M, n_size=N, k_size=K,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export dual INT8 GEMM + SiLU kernel to C headers.")
    p.add_argument("--config", type=str, default=None, help="TOML config with [[configs]] M/N/K entries")
    p.add_argument("--M", type=int, default=256, help="Symbolic M for the trace (M is dynamic at runtime)")
    p.add_argument("--N", type=int, default=2048, help="Inter dim (gate/up width)")
    p.add_argument("--K", type=int, default=512, help="Model dim (fc1 input)")
    p.add_argument("--bm", type=int, default=128)
    p.add_argument("--bn", type=int, default=64, help="N tile (small: dual accumulator pressure)")
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--atom-layout", type=str, default="2,2,1")
    p.add_argument("--out", type=str, default=None, help="Artifacts dir (default: cute/artifacts)")
    args = p.parse_args()

    atom_layout = tuple(int(x) for x in args.atom_layout.split(","))
    artifacts_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    if args.config:
        configs = _load_configs(args.config, default_m=args.M)
    else:
        configs = [(args.M, args.N, args.K)]

    for (M, N, K) in configs:
        _export_one(M, N, K, args.bm, args.bn, args.num_stages, atom_layout, artifacts_dir)

    print(f"\nAll exports complete -> {artifacts_dir}/")
