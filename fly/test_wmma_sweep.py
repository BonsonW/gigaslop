"""Sweep all interesting WMMA shapes to find what gfx1201 actually supports."""
import subprocess, sys, textwrap

CASES = [
    # (name, a_size, a_type, b_size, b_type, c_type, op, extra_args)
    ("wmma_f32_16x16x16_fp8_fp8 [K=16 fp8→f32] known-good",
     8, "fx.Int32", 8, "fx.Int32", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_fp8_fp8(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x16_bf8_fp8 [K=16 bf8/fp8→f32]",
     8, "fx.Int32", 8, "fx.Int32", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_bf8_fp8(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x16_fp8_bf8 [K=16 fp8/bf8→f32]",
     8, "fx.Int32", 8, "fx.Int32", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_fp8_bf8(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x16_bf8_bf8 [K=16 bf8→f32]",
     8, "fx.Int32", 8, "fx.Int32", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_bf8_bf8(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x16_f16 [K=16 f16→f32]",
     16, "fx.Float16", 16, "fx.Float16", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_f16(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x32_f16 [K=32 f16→f32]",
     16, "fx.Float16", 16, "fx.Float16", "fx.Float32",
     "rocdl.wmma_f32_16x16x32_f16(c.type, a, b, c).result", ""),
    ("wmma_f16_16x16x16_f16 [K=16 f16→f16]",
     16, "fx.Float16", 16, "fx.Float16", "fx.Float16",
     "rocdl.wmma_f16_16x16x16_f16(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x16_bf16 [K=16 bf16→f32]",
     16, "fx.BFloat16", 16, "fx.BFloat16", "fx.Float32",
     "rocdl.wmma_f32_16x16x16_bf16(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x32_bf16 [K=32 bf16→f32]",
     16, "fx.BFloat16", 16, "fx.BFloat16", "fx.Float32",
     "rocdl.wmma_f32_16x16x32_bf16(c.type, a, b, c).result", ""),
    ("wmma_f32_16x16x64_fp8_fp8 [K=64 fp8→f32]",
     8, "fx.Int32", 8, "fx.Int32", "fx.Float32",
     "rocdl.wmma_f32_16x16x64_fp8_fp8(c.type, a, b, c).result", ""),
    ("wmma_scale_f32_16x16x128_f8f6f4 [scale K=128]",
     16, "fx.Int32", 16, "fx.Int32", "fx.Float32",
     "rocdl.wmma_scale_f32_16x16x128_f8f6f4(c.type, a, b, c, sa, sb, fmtA=0, fmtB=0).result",
     "sa = arith.constant(0, type=fx.T.i32()); sb = arith.constant(0, type=fx.T.i32())"),
    ("wmma_scale16_f32_16x16x128_f8f6f4 [scale16 K=128]",
     16, "fx.Int32", 16, "fx.Int32", "fx.Float32",
     "rocdl.wmma_scale16_f32_16x16x128_f8f6f4(c.type, a, b, c, sa, sb, fmtA=0, fmtB=0).result",
     "sa = arith.constant(0, type=fx.T.i32()); sb = arith.constant(0, type=fx.T.i32())"),
]

TEMPLATE = textwrap.dedent("""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, rocdl, buffer_ops

@flyc.kernel
def test_kernel(arg_out: fx.Tensor):
    a = fx.full({a_size}, {a_init}, {a_type})
    b = fx.full({b_size}, {b_init}, {b_type})
    c = fx.full(8, {c_init}, {c_type})
    {extra}
    result = {op}
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)

@flyc.jit
def launch(out: fx.Tensor, stream: fx.Stream):
    l = test_kernel(out)
    l.launch(grid=(1,1,1), block=(32,1,1), stream=stream)

out    = torch.zeros(1, dtype=torch.float32, device='cuda')
stream = torch.cuda.current_stream()
launch(out, stream)
torch.cuda.synchronize()
print("PASS  out[0]=%g" % out[0].item())
""")

import os, tempfile

for (label, a_sz, a_ty, b_sz, b_ty, c_ty, op, extra) in CASES:
    a_init = "0.0" if "Float" in a_ty or "BFloat" in a_ty else "0"
    b_init = "0.0" if "Float" in b_ty or "BFloat" in b_ty else "0"
    c_init = "0.0" if "Float" in c_ty else "0"

    code = TEMPLATE.format(
        a_size=a_sz, a_type=a_ty, a_init=a_init,
        b_size=b_sz, b_type=b_ty, b_init=b_init,
        c_type=c_ty, c_init=c_init,
        op=op, extra=extra,
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name

    env = {**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1"}
    res = subprocess.run(
        [sys.executable, fname],
        capture_output=True, text=True, env=env, timeout=30,
    )
    os.unlink(fname)

    out_text = res.stdout.strip() + res.stderr.strip()
    if "PASS" in out_text:
        status = "PASS"
    elif "Cannot select" in out_text:
        status = "FAIL (Cannot select)"
    elif "Verification failed" in out_text or "MLIRError" in out_text:
        # Extract the key error
        line = next((l for l in out_text.splitlines() if "error:" in l.lower()), "MLIR error")
        status = f"FAIL (MLIR: {line[:80]})"
    else:
        status = f"FAIL ({out_text[:80]})"

    print(f"  {status:40s}  {label}")
