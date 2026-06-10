"""Direct test of WMMA shapes on gfx1201 with correct fragment sizes.

Fragment sizes (wave32):
  wmma A/B fragment = (M×K or K×N elements) / 32 lanes
  - K=16 fp8:  16×16/32 = 8 fp8 = 2 i32  → vector<2xi32>
  - K=16 f16:  16×16/32 = 8 f16           → vector<8xf16>
  - K=32 f16:  16×32/32 = 16 f16          → vector<16xf16>
  - K=64 fp8:  16×64/32 = 32 fp8 = 8 i32  → vector<8xi32>
  - K=128 fp8: 16×128/32 = 64 fp8 = 16 i32 → vector<16xi32>
  C: 16×16/32 = 8 f32                     → vector<8xf32>
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, rocdl, buffer_ops
from flydsl.expr.arith import ArithValue

stream = torch.cuda.current_stream()
out = torch.zeros(1, dtype=torch.float32, device="cuda")


def try_launch(label, launch_fn):
    try:
        launch_fn(out, stream)
        torch.cuda.synchronize()
        print(f"PASS  {label}")
    except Exception as e:
        msg = str(e)
        if "Cannot select" in msg:
            print(f"FAIL  {label}  [Cannot select]")
        elif "Verification failed" in msg or "MLIRError" in msg:
            for line in msg.splitlines():
                if "operand" in line or "must be" in line:
                    print(f"FAIL  {label}  [MLIR: {line.strip()[:90]}]")
                    break
            else:
                print(f"FAIL  {label}  [MLIR error]")
        else:
            print(f"FAIL  {label}  [{msg[:80]}]")


# ── K=16 fp8 (known-good) : A/B = vector<2xi32> ────────────────────────────
@flyc.kernel
def k_fp8_k16(arg_out: fx.Tensor):
    c = fx.full(8, 0.0, fx.Float32)
    a = fx.full(2, 0, fx.Int32)          # 8 fp8 per lane = 2 × i32
    b = fx.full(2, 0, fx.Int32)
    r = rocdl.wmma_f32_16x16x16_fp8_fp8(c.type, a, b, c).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(ArithValue(mlir_vector.extract(r, static_position=[0], dynamic_position=[])), rsrc, 0)

@flyc.jit
def launch_fp8_k16(out: fx.Tensor, stream: fx.Stream):
    k_fp8_k16(out).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

try_launch("wmma_f32_16x16x16_fp8_fp8  [K=16 fp8→f32] (2×i32 frags)", launch_fp8_k16)


# ── K=16 f16 : A/B = vector<8xf16> ────────────────────────────────────────
@flyc.kernel
def k_f16_k16(arg_out: fx.Tensor):
    c = fx.full(8, 0.0, fx.Float32)
    a = fx.full(8, 0.0, fx.Float16)      # 8 f16 per lane
    b = fx.full(8, 0.0, fx.Float16)
    r = rocdl.wmma_f32_16x16x16_f16(c.type, a, b, c).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(ArithValue(mlir_vector.extract(r, static_position=[0], dynamic_position=[])), rsrc, 0)

@flyc.jit
def launch_f16_k16(out: fx.Tensor, stream: fx.Stream):
    k_f16_k16(out).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

try_launch("wmma_f32_16x16x16_f16      [K=16 f16→f32] (8×f16 frags)", launch_f16_k16)


# ── K=32 f16 : A/B = vector<16xf16> ───────────────────────────────────────
@flyc.kernel
def k_f16_k32(arg_out: fx.Tensor):
    c = fx.full(8, 0.0, fx.Float32)
    a = fx.full(16, 0.0, fx.Float16)     # 16 f16 per lane
    b = fx.full(16, 0.0, fx.Float16)
    r = rocdl.wmma_f32_16x16x32_f16(c.type, a, b, c).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(ArithValue(mlir_vector.extract(r, static_position=[0], dynamic_position=[])), rsrc, 0)

@flyc.jit
def launch_f16_k32(out: fx.Tensor, stream: fx.Stream):
    k_f16_k32(out).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

try_launch("wmma_f32_16x16x32_f16      [K=32 f16→f32] (16×f16 frags)", launch_f16_k32)


# ── K=64 fp8 : A/B = vector<8xi32> ────────────────────────────────────────
@flyc.kernel
def k_fp8_k64(arg_out: fx.Tensor):
    c = fx.full(8, 0.0, fx.Float32)
    a = fx.full(8, 0, fx.Int32)          # 32 fp8 per lane = 8 × i32
    b = fx.full(8, 0, fx.Int32)
    r = rocdl.wmma_f32_16x16x64_fp8_fp8(c.type, a, b, c).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(ArithValue(mlir_vector.extract(r, static_position=[0], dynamic_position=[])), rsrc, 0)

@flyc.jit
def launch_fp8_k64(out: fx.Tensor, stream: fx.Stream):
    k_fp8_k64(out).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

try_launch("wmma_f32_16x16x64_fp8_fp8  [K=64 fp8→f32] (8×i32 frags)", launch_fp8_k64)


# ── K=16 bf8 : same frags as fp8 ──────────────────────────────────────────
@flyc.kernel
def k_bf8_k16(arg_out: fx.Tensor):
    c = fx.full(8, 0.0, fx.Float32)
    a = fx.full(2, 0, fx.Int32)
    b = fx.full(2, 0, fx.Int32)
    r = rocdl.wmma_f32_16x16x16_bf8_bf8(c.type, a, b, c).result
    rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
    buffer_ops.buffer_store(ArithValue(mlir_vector.extract(r, static_position=[0], dynamic_position=[])), rsrc, 0)

@flyc.jit
def launch_bf8_k16(out: fx.Tensor, stream: fx.Stream):
    k_bf8_k16(out).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

try_launch("wmma_f32_16x16x16_bf8_bf8  [K=16 bf8→f32] (2×i32 frags)", launch_bf8_k16)
