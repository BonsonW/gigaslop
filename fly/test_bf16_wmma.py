"""Quick test: does wmma_f32_16x16x16_bf16 compile and run on gfx1201?"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, rocdl, gpu, buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

@flyc.kernel
def test_bf16_wmma(arg_out: fx.Tensor):
    # bf16 fragments are packed as v4i32 (8 bf16 per lane = 16 bytes = 4 i32)
    c8f32 = fx.full(8, 0.0, fx.Float32)
    a_vec = fx.full(4, 0, fx.Int32)
    b_vec = fx.full(4, 0, fx.Int32)
    result = rocdl.wmma_f32_16x16x16_bf16(c8f32.type, a_vec, b_vec, c8f32).result
    from flydsl._mlir.dialects import vector as mlir_vector
    val = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(val, rsrc, 0)

@flyc.jit
def launch_test(out: fx.Tensor, stream: fx.Stream):
    launcher = test_bf16_wmma(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

if __name__ == "__main__":
    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    launch_test(out, stream)
    torch.cuda.synchronize()
    print("bf16 WMMA compiled and ran OK, out[0] =", out[0].item())
