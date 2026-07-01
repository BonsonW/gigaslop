#!/usr/bin/env python3
"""Export the fused INT8 GEMM + rotary kernel to C headers (AOT).

Mirrors the FlyDSL TOML-config-driven export setup (fly/export_fp8_gemm_rotary.py),
but uses our CuteDSL `export_to_c` path.

Run:
    pyvenv/bin/python cute/export_gemm_i8_rotary.py                            # defaults
    pyvenv/bin/python cute/export_gemm_i8_rotary.py --K 512 --nhead 8 --head-dim 64
    pyvenv/bin/python cute/export_gemm_i8_rotary.py \\
        --config cute/export_configs/gemm_i8_rotary.toml                       # batch sweep

Output (in cute/artifacts/):
    gemm_i8_rotary_N{N}_K{K}_H{nhead}D{head_dim}R{rotary_dim}S{seqlen}.h
    (N = 3*nhead*head_dim)
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
from ampere_gemm_i8_rotary import export_gemm_i8_rotary


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


def _load_configs(path, defaults):
    data = tomllib.load(open(path, "rb")) if tomllib is not None else _parse_simple_toml(path)
    out = []
    for c in data["configs"]:
        out.append(dict(
            M=int(c.get("M", defaults["M"])),
            K=int(c["K"]),
            nhead=int(c.get("nhead", defaults["nhead"])),
            head_dim=int(c.get("head_dim", defaults["head_dim"])),
            rotary_dim=int(c.get("rotary_dim", defaults["rotary_dim"])),
            seqlen=int(c.get("seqlen", defaults["seqlen"])),
        ))
    return out


def _export_one(cfg, bm, bn, num_stages, atom_layout, artifacts_dir):
    N = 3 * cfg["nhead"] * cfg["head_dim"]
    name = (f"gemm_i8_rotary_N{N}_K{cfg['K']}"
            f"_H{cfg['nhead']}D{cfg['head_dim']}R{cfg['rotary_dim']}S{cfg['seqlen']}")
    print(f"\n[M={cfg['M']} N={N} K={cfg['K']}]  nhead={cfg['nhead']} "
          f"head_dim={cfg['head_dim']} rotary_dim={cfg['rotary_dim']} seqlen={cfg['seqlen']}  -> {name}.h")
    export_gemm_i8_rotary(
        nhead=cfg["nhead"], head_dim=cfg["head_dim"],
        rotary_dim=cfg["rotary_dim"], seqlen=cfg["seqlen"],
        atom_layout_mnk=atom_layout,
        file_path=artifacts_dir,
        file_name=name,
        function_prefix=name,
        bm=bm, bn=bn, num_stages=num_stages,
        m_size=cfg["M"], k_size=cfg["K"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export fused INT8 GEMM + rotary kernel to C headers.")
    p.add_argument("--config", type=str, default=None, help="TOML config with [[configs]] entries")
    p.add_argument("--M", type=int, default=256, help="Symbolic M for the trace (M is dynamic at runtime)")
    p.add_argument("--K", type=int, default=512, help="Input dim")
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--rotary-dim", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--bm", type=int, default=128)
    p.add_argument("--bn", type=int, default=256, help="N tile (must be a multiple of head_dim)")
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--atom-layout", type=str, default="2,4,1")
    p.add_argument("--out", type=str, default=None, help="Artifacts dir (default: cute/artifacts)")
    args = p.parse_args()

    atom_layout = tuple(int(x) for x in args.atom_layout.split(","))
    artifacts_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    defaults = dict(M=args.M, nhead=args.nhead, head_dim=args.head_dim,
                    rotary_dim=args.rotary_dim, seqlen=args.seqlen)
    if args.config:
        configs = _load_configs(args.config, defaults)
    else:
        configs = [dict(M=args.M, K=args.K, nhead=args.nhead, head_dim=args.head_dim,
                        rotary_dim=args.rotary_dim, seqlen=args.seqlen)]

    for cfg in configs:
        _export_one(cfg, args.bm, args.bn, args.num_stages, atom_layout, artifacts_dir)

    print(f"\nAll exports complete -> {artifacts_dir}/")
