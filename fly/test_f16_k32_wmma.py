"""Test wmma_f32_16x16x32_f16 (K=32 f16 input) on gfx1201."""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import rocdl, buffer_ops

@flyc.kernel
def test_f16_k32(arg_out: fx.Tensor):
    c8f32 = fx.full(8, 0.0, fx.Float32)
    a     = fx.full(16, 0.0, fx.Float16)  # 16 f16 per lane (K=32 / 32 lanes × 16 A-rows)
    b     = fx.full(16, 0.0, fx.Float16)
    result = rocdl.wmma_f32_16x16x32_f16(c8f32.type, a, b, c8f32).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)

@flyc.jit
def launch(out: fx.Tensor, stream: fx.Stream):
    launcher = test_f16_k32(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

if __name__ == "__main__":
    out    = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    launch(out, stream)
    torch.cuda.synchronize()
    print("wmma_f32_16x16x32_f16 (K=32 f16): PASS  out[0]=%g" % out[0].item())
