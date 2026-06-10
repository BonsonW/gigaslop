"""Test: does wmma_scale_f32_16x16x128_f8f6f4 compile on gfx1201?"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, rocdl, gpu, buffer_ops
from flydsl.expr.arith import ArithValue

@flyc.kernel
def test_wmma_scale(arg_out: fx.Tensor):
    # fp8 E4M3: 16x128 / 32 lanes = 64 bytes/lane = 16 i32 → vector<16xi32>
    # c: vector<8xf32>  (16x16 / 32 lanes = 8 outputs per lane)
    c8f32    = fx.full(8, 0.0, fx.Float32)
    a_v16i32 = fx.full(16, 0, fx.Int32)
    b_v16i32 = fx.full(16, 0, fx.Int32)
    scale_a = arith.constant(0, type=fx.T.i32())
    scale_b = arith.constant(0, type=fx.T.i32())
    # fmtA=0 → E4M3 fp8
    result = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        c8f32.type, a_v16i32, b_v16i32, c8f32, scale_a, scale_b,
        fmtA=0, fmtB=0,
    )
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    from flydsl._mlir.dialects import vector as mlir_vector
    val = mlir_vector.extract(result, static_position=[0], dynamic_position=[])
    buffer_ops.buffer_store(val, rsrc, 0)

@flyc.jit
def launch_test(out: fx.Tensor, stream: fx.Stream):
    launcher = test_wmma_scale(out)
    launcher.launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

if __name__ == "__main__":
    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    launch_test(out, stream)
    torch.cuda.synchronize()
    print("wmma_scale compiled and ran OK, out[0] =", out[0].item())
