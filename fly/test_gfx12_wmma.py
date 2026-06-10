"""Test gfx12-specific WMMA shapes: f16 K=32, fp8 K=64, fp8 K=128."""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, rocdl, buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T


# ---------------------------------------------------------------------------
# Test 1: wmma_f32_16x16x32_f16  (K=32, f16 inputs, f32 accumulator)
# A: 16×32 / 32 lanes = 16 f16/lane = 8 i32
# B: 32×16 / 32 lanes = 16 f16/lane = 8 i32
# C: 16×16 / 32 lanes = 8 f32/lane
# ---------------------------------------------------------------------------
@flyc.kernel
def test_wmma_f32_16x16x32_f16(arg_out: fx.Tensor):
    c8f32    = fx.full(8, 0.0, fx.Float32)
    a_v8i32  = fx.full(8, 0, fx.Int32)   # 16 f16 packed as 8 × i32
    b_v8i32  = fx.full(8, 0, fx.Int32)
    result = rocdl.wmma_f32_16x16x32_f16(c8f32.type, a_v8i32, b_v8i32, c8f32).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)


@flyc.jit
def launch_test_f16_k32(out: fx.Tensor, stream: fx.Stream):
    launcher = test_wmma_f32_16x16x32_f16(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Test 2: wmma_f32_16x16x64_fp8_fp8  (K=64, fp8 inputs, f32 accumulator)
# A: 16×64 / 32 lanes = 32 fp8/lane = 8 i32
# B: 64×16 / 32 lanes = 32 fp8/lane = 8 i32
# C: 8 f32/lane
# ---------------------------------------------------------------------------
@flyc.kernel
def test_wmma_f32_16x16x64_fp8_fp8(arg_out: fx.Tensor):
    c8f32    = fx.full(8, 0.0, fx.Float32)
    a_v8i32  = fx.full(8, 0, fx.Int32)
    b_v8i32  = fx.full(8, 0, fx.Int32)
    result = rocdl.wmma_f32_16x16x64_fp8_fp8(c8f32.type, a_v8i32, b_v8i32, c8f32).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)


@flyc.jit
def launch_test_fp8_k64(out: fx.Tensor, stream: fx.Stream):
    launcher = test_wmma_f32_16x16x64_fp8_fp8(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Test 3: wmma_f32_16x16x128_fp8_fp8  (K=128, fp8 inputs)
# A: 16×128 / 32 lanes = 64 fp8/lane = 16 i32
# ---------------------------------------------------------------------------
@flyc.kernel
def test_wmma_f32_16x16x128_fp8_fp8(arg_out: fx.Tensor):
    c8f32     = fx.full(8, 0.0, fx.Float32)
    a_v16i32  = fx.full(16, 0, fx.Int32)
    b_v16i32  = fx.full(16, 0, fx.Int32)
    result = rocdl.wmma_f32_16x16x128_fp8_fp8(c8f32.type, a_v16i32, b_v16i32, c8f32).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)


@flyc.jit
def launch_test_fp8_k128(out: fx.Tensor, stream: fx.Stream):
    launcher = test_wmma_f32_16x16x128_fp8_fp8(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


if __name__ == "__main__":
    out    = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()

    for label, launch_fn in [
        ("wmma_f32_16x16x32_f16  (K=32 f16)", launch_test_f16_k32),
        ("wmma_f32_16x16x64_fp8  (K=64 fp8)", launch_test_fp8_k64),
        ("wmma_f32_16x16x128_fp8 (K=128 fp8)", launch_test_fp8_k128),
    ]:
        try:
            launch_fn(out, stream)
            torch.cuda.synchronize()
            print(f"PASS  {label}  out[0]={out[0].item():.1f}")
        except Exception as e:
            print(f"FAIL  {label}  {type(e).__name__}: {e}")
