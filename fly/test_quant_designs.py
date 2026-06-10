"""Compare per-token quantize kernel designs: 4-warp LDS vs 1-warp-per-row no-LDS."""
import os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functools, math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WARP_SIZE = 32
FP8_MAX   = 448.0
SHUFFLE_DISTS = [1 << i for i in range(int(math.log2(WARP_SIZE)))]

# ── Design B: 1 warp per row, ROWS_PER_BLOCK rows per block, no LDS ──────────

ROWS_PER_BLOCK = 8   # try 4, 8, 16

@functools.lru_cache(maxsize=32)
def compile_quant_1warp(*, K: int, rows_per_block: int = ROWS_PER_BLOCK):
    VEC_PER_LANE  = K // WARP_SIZE        # fp16 elements per lane (e.g. 32 for K=1024)
    LOADS         = VEC_PER_LANE // 8     # 128-bit loads per lane

    @flyc.kernel
    def kernel_quant_1warp(
        arg_out_fp8:   fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp:       fx.Tensor,
    ):
        bid     = fx.block_idx.x
        tid     = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane    = tid % WARP_SIZE

        lane_i32    = ArithValue(arith.index_cast(T.i32, lane))
        warp_id_i32 = ArithValue(arith.index_cast(T.i32, warp_id))
        bid_i32     = ArithValue(arith.index_cast(T.i32, bid))

        g_row_i32 = bid_i32 * rows_per_block + warp_id_i32

        inp_rsrc   = buffer_ops.create_buffer_resource(arg_inp,       max_size=True)
        out_rsrc   = buffer_ops.create_buffer_resource(arg_out_fp8,   max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(arg_out_scale, max_size=True)

        # Each lane handles VEC_PER_LANE fp16 elements of its row
        f16_base = g_row_i32 * K + lane_i32 * VEC_PER_LANE

        c0_f32    = arith.constant(0.0, type=T.f32)
        local_max = c0_f32
        all_vals  = []

        for li in range_constexpr(LOADS):
            vec_f16 = buffer_ops.buffer_load(
                inp_rsrc, f16_base + li * 8, vec_width=8, dtype=fx.Float16)
            vec_f32 = ArithValue(vec_f16).extf(T.vec(8, T.f32))
            for vi in range_constexpr(8):
                v = ArithValue(mlir_vector.extract(
                    vec_f32, static_position=[vi], dynamic_position=[]))
                all_vals.append(v)
                local_max = local_max.maximumf(fx_math.absf(v))

        # Intra-warp XOR → row-level max (all lanes hold the same max)
        for sh in SHUFFLE_DISTS:
            local_max = local_max.maximumf(local_max.shuffle_xor(sh, WARP_SIZE))

        eps      = arith.constant(1e-12, type=T.f32)
        fp8_max  = arith.constant(FP8_MAX, type=T.f32)
        gmax     = local_max.maximumf(eps)
        inv_sc   = ArithValue(fp8_max) / gmax
        scale    = gmax / ArithValue(fp8_max)

        if lane == 0:
            buffer_ops.buffer_store(scale, scale_rsrc, g_row_i32)

        fp8_base = g_row_i32 * K + lane_i32 * VEC_PER_LANE
        zero     = arith.constant(0, type=T.i32)

        for wg in range_constexpr(VEC_PER_LANE // 4):
            base = wg * 4
            v0 = all_vals[base + 0] * inv_sc
            v1 = all_vals[base + 1] * inv_sc
            v2 = all_vals[base + 2] * inv_sc
            v3 = all_vals[base + 3] * inv_sc
            packed = rocdl.cvt_pk_fp8_f32(T.i32, v0, v1, zero, 0)
            packed = rocdl.cvt_pk_fp8_f32(T.i32, v2, v3, packed, 1)
            buffer_ops.buffer_store(
                packed, out_rsrc, fp8_base + wg * 4, offset_is_bytes=True)

    @flyc.jit
    def launch_quant_1warp(
        arg_out_fp8:   fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp:       fx.Tensor,
        m:             fx.Int32,
        stream:        fx.Stream,
    ):
        c1   = 1
        blk  = rows_per_block * WARP_SIZE
        launcher = kernel_quant_1warp(arg_out_fp8, arg_out_scale, arg_inp)
        launcher.launch(
            grid=(m // rows_per_block, c1, c1),
            block=(blk, c1, c1),
            stream=stream,
        )

    return launch_quant_1warp


# ── Helpers ───────────────────────────────────────────────────────────────────

def bench(launch, h_fp16, B, K, stream, warmup=20, iters=500):
    h_fp8   = torch.zeros(B, K, dtype=torch.uint8, device="cuda")
    h_scale = torch.zeros(B, dtype=torch.float32, device="cuda")
    launch(h_fp8, h_scale, h_fp16, B, stream)
    torch.cuda.synchronize()
    for _ in range(warmup):
        launch(h_fp8, h_scale, h_fp16, B, stream)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record(stream)
    for _ in range(iters):
        launch(h_fp8, h_scale, h_fp16, B, stream)
    t1.record(stream)
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / iters
    return us, h_fp8, h_scale


def verify(h_fp8, h_scale, h_fp16):
    amax  = h_fp16.float().abs().amax(dim=-1).clamp(min=1e-12)
    inv_s = 448.0 / amax
    h_ref = (h_fp16.float() * inv_s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    diff  = (h_fp8.view(torch.float8_e4m3fn).float() - h_ref.float()).abs()
    scale_err = (h_scale - amax / 448.0).abs().max().item()
    return diff.max().item(), (diff > 0.5).sum().item(), scale_err


B, K = 512, 1024
device = "cuda"
stream = torch.cuda.current_stream()
torch.manual_seed(42)
h_fp16 = torch.randn(B, K, dtype=torch.float16, device=device)

from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize
launch_4warp = compile_fp8_per_token_quantize(K=K)

bytes_total = B * K * 2 + B * K + B * 4   # fp16 read + fp8 write + scale write

print("=" * 60)
print(f"B={B}  K={K}  data={bytes_total/1e6:.2f} MB")
print("=" * 60)

# 4-warp LDS design
us4, fp8_4, sc4 = bench(launch_4warp, h_fp16, B, K, stream)
e4, n4, se4 = verify(fp8_4, sc4, h_fp16)
print(f"4-warp LDS   : {us4:.2f} µs  {bytes_total/us4/1e3:.1f} GB/s  |  err={e4:.1f} ({n4} elems)  scale_err={se4:.2e}")

# 1-warp no-LDS designs
for rpb in [4, 8, 16]:
    if B % rpb != 0:
        continue
    launch_1w = compile_quant_1warp(K=K, rows_per_block=rpb)
    us1, fp8_1, sc1 = bench(launch_1w, h_fp16, B, K, stream)
    e1, n1, se1 = verify(fp8_1, sc1, h_fp16)
    print(f"1-warp rpb={rpb:2d}  : {us1:.2f} µs  {bytes_total/us1/1e3:.1f} GB/s  |  err={e1:.1f} ({n1} elems)  scale_err={se1:.2e}")
