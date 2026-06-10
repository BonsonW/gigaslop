"""Test wmma_f32_16x16x64_fp8_fp8 on gfx1201."""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import rocdl, buffer_ops

@flyc.kernel
def test_fp8_k64(arg_out: fx.Tensor):
    c8f32 = fx.full(8, 0.0, fx.Float32)
    a     = fx.full(8, 0, fx.Int32)   # 32 fp8 per lane = 8 × i32
    b     = fx.full(8, 0, fx.Int32)
    result = rocdl.wmma_f32_16x16x64_fp8_fp8(c8f32.type, a, b, c8f32).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    val  = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)

@flyc.jit
def launch(out: fx.Tensor, stream: fx.Stream):
    launcher = test_fp8_k64(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

if __name__ == "__main__":
    out    = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    launch(out, stream)
    torch.cuda.synchronize()
    print("wmma_f32_16x16x64_fp8_fp8 (K=64 fp8): PASS  out[0]=%g" % out[0].item())
