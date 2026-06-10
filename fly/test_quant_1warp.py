"""Test 1-warp (no barrier) vs 4-warp quantize kernel performance."""
import os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functools
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

WARP_SIZE     = 32
FP8_MAX       = 448.0
SHUFFLE_DISTS = [1, 2, 4, 8, 16]

K          = 1024
BLOCK_SIZE = 32   # 1 warp per row — no inter-warp barrier needed
VPT        = K // BLOCK_SIZE   # 32 fp16 per thread
LOADS      = VPT // 8          # 4 × vec8f16 loads


@flyc.kernel
def _kern_quant_1warp(
    arg_out_fp8:   fx.Tensor,
    arg_out_scale: fx.Tensor,
    arg_inp:       fx.Tensor,
):
    bid     = fx.block_idx.x
    tid     = fx.thread_idx.x
    bid_i32 = ArithValue(arith.index_cast(T.i32, bid))
    tid_i32 = ArithValue(arith.index_cast(T.i32, tid))
    lane    = ArithValue(arith.index_cast(T.i32, tid)) % WARP_SIZE

    inp_rsrc   = buffer_ops.create_buffer_resource(arg_inp,       max_size=True)
    out_rsrc   = buffer_ops.create_buffer_resource(arg_out_fp8,   max_size=True)
    scale_rsrc = buffer_ops.create_buffer_resource(arg_out_scale, max_size=True)

    f16_base  = bid_i32 * K + tid_i32 * VPT
    c0        = arith.constant(0.0, type=T.f32)
    local_max = c0
    all_vals  = []

    for li in range_constexpr(LOADS):
        vec = buffer_ops.buffer_load(
            inp_rsrc, f16_base + li * 8, vec_width=8, dtype=fx.Float16)
        vec_f32_ty = T.vec(8, T.f32)
        vec_f32 = ArithValue(vec).extf(vec_f32_ty)
        for vi in range_constexpr(8):
            v = ArithValue(mlir_vector.extract(
                vec_f32, static_position=[vi], dynamic_position=[]))
            all_vals.append(v)
            local_max = local_max.maximumf(fx_math.absf(v))

    # Warp XOR reduction — no LDS, no barrier
    for sh in SHUFFLE_DISTS:
        local_max = local_max.maximumf(local_max.shuffle_xor(sh, WARP_SIZE))

    eps     = arith.constant(1e-12, type=T.f32)
    fp8_max = arith.constant(FP8_MAX, type=T.f32)
    gmax    = local_max.maximumf(eps)
    inv_sc  = ArithValue(fp8_max) / gmax
    scale   = gmax / ArithValue(fp8_max)

    if lane == 0:
        buffer_ops.buffer_store(scale, scale_rsrc, bid_i32)

    fp8_base  = bid_i32 * K + tid_i32 * VPT
    zero_i32  = arith.constant(0, type=T.i32)

    for wg in range_constexpr(VPT // 4):
        base   = wg * 4
        v0     = all_vals[base + 0] * inv_sc
        v1     = all_vals[base + 1] * inv_sc
        v2     = all_vals[base + 2] * inv_sc
        v3     = all_vals[base + 3] * inv_sc
        packed = rocdl.cvt_pk_fp8_f32(T.i32, v0, v1, zero_i32, 0)
        packed = rocdl.cvt_pk_fp8_f32(T.i32, v2, v3, packed,   1)
        buffer_ops.buffer_store(
            packed, out_rsrc, fp8_base + wg * 4, offset_is_bytes=True)


@functools.lru_cache(maxsize=4)
def compile_quant_1warp(*, K: int = 1024):
    assert K == 1024, "only K=1024 supported in this prototype"

    @flyc.jit
    def launch(
        arg_out_fp8:   fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp:       fx.Tensor,
        m:             fx.Int32,
        stream:        fx.Stream,
    ):
        c1       = 1
        launcher = _kern_quant_1warp(arg_out_fp8, arg_out_scale, arg_inp)
        launcher.launch(grid=(m, c1, c1), block=(BLOCK_SIZE, c1, c1), stream=stream)

    return launch


def main():
    B, H   = 512, 1024
    device = "cuda"
    stream = torch.cuda.current_stream()

    from rdna_fp8_per_token_quantize import compile_fp8_per_token_quantize
    q4w = compile_fp8_per_token_quantize(K=H)
    q1w = compile_quant_1warp(K=H)

    h_fp16   = torch.randn(B, H, dtype=torch.float16, device=device)
    fp8_4w   = torch.zeros(B, H, dtype=torch.uint8,   device=device)
    fp8_1w   = torch.zeros(B, H, dtype=torch.uint8,   device=device)
    scale_4w = torch.zeros(B, dtype=torch.float32, device=device)
    scale_1w = torch.zeros(B, dtype=torch.float32, device=device)

    WARMUP, ITERS = 20, 1000

    def time_fn(fn):
        for _ in range(WARMUP): fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(stream)
        for _ in range(ITERS): fn()
        t1.record(stream)
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1e3 / ITERS

    us_4w = time_fn(lambda: q4w(fp8_4w, scale_4w, h_fp16, B, stream))
    us_1w = time_fn(lambda: q1w(fp8_1w, scale_1w, h_fp16, B, stream))

    print(f"4-warp (128 thr, barrier): {us_4w:.2f} µs")
    print(f"1-warp (32 thr, no-barr) : {us_1w:.2f} µs")
    print(f"Speedup: {us_4w/us_1w:.2f}×")

    scale_diff = (scale_1w - scale_4w).abs().max().item()
    print(f"Scale max diff: {scale_diff:.6f}  ({'PASS' if scale_diff < 1e-5 else 'FAIL'})")


if __name__ == "__main__":
    main()
