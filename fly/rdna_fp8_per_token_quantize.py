"""FP16 → FP8 per-token quantize kernel for RDNA4 (gfx120x, wave32).

Quantizes a row-major fp16 tensor to fp8_e4m3fn with per-token (per-row) f32 scale.

  inp[M, K]  fp16  →  out_fp8[M, K]  fp8_e4m3fn bytes
                       scale_a[M]    f32  (amax / 448.0)

scale_a format matches the scale_a input of rdna_fp8_preshuffle_gemm, so the
output can be fed directly to the FP8 fc2 GEMM as activation input.

Grid:  (M, 1, 1) — one workgroup per row
Block: (256, 1, 1)

K must be a multiple of 256 and K // 256 must be a multiple of 4.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

BLOCK_THREADS = 256
WARP_SIZE = 32
NUM_WARPS = BLOCK_THREADS // WARP_SIZE  # 8
FP8_MAX = 448.0


@functools.lru_cache(maxsize=32)
def compile_fp8_per_token_quantize(*, K: int):
    """Compile FP16 → FP8 per-token quantize kernel for RDNA4.

    K = inter_dim (number of elements per row, = N of the dual-GEMM output).
    Must satisfy: K % 256 == 0 and (K // 256) % 4 == 0.

    Returns launcher(out_fp8, scale_a, inp_fp16, m, stream).
    """
    assert K % BLOCK_THREADS == 0, f"K={K} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    VEC = K // BLOCK_THREADS
    assert VEC % 4 == 0, f"VEC={VEC} must be divisible by 4 for fp8 packing"

    arch = get_rocm_arch()

    # LDS: NUM_WARPS f32 slots for inter-warp max reduction (reuse slot 0 for inv_scale broadcast)
    allocator = SmemAllocator(None, arch=arch)
    wmax_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = wmax_offset + NUM_WARPS * 4  # 8 * 4 = 32 bytes

    SHUFFLE_DISTS = [1 << i for i in range(int(math.log2(WARP_SIZE)))]  # [1,2,4,8,16]

    @flyc.kernel
    def kernel_fp8_per_token_quantize(
        arg_out_fp8: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lane    = tid % WARP_SIZE
        warp_id = tid // WARP_SIZE
        thread_id = ArithValue(tid)

        # Layout API for fp16 input
        inp_buf      = fx.rocdl.make_buffer_tensor(arg_inp)
        copy_atom    = fx.make_copy_atom(fx.rocdl.BufferCopy(VEC * 16), 16)  # 16 bits = fp16
        vec_f32_ty   = T.vec(VEC, T.f32)
        vec_reg_ty   = fx.MemRefType.get(T.f16, fx.LayoutType.get(VEC, 1), fx.AddressSpace.Register)
        vec_reg_lay  = fx.make_layout(VEC, 1)

        def _load_vec(div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        out_rsrc   = buffer_ops.create_buffer_resource(arg_out_fp8,   max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(arg_out_scale, max_size=True)

        # LDS for inter-warp max reduction
        base_ptr = allocator.get_base()
        s_wmax = SmemPtr(base_ptr, wmax_offset, T.f32, shape=(NUM_WARPS,))
        s_wmax.get()

        # Load VEC fp16 values and extend to f32
        row_x   = fx.slice(inp_buf, (bid, None))
        row_div = fx.logical_divide(row_x, fx.make_layout(VEC, 1))
        inp_f32 = _load_vec(row_div, thread_id).extf(vec_f32_ty)

        # Extract individual f32 values
        c0_f32 = arith.constant(0.0, type=T.f32)
        vals = []
        for vi in range_constexpr(VEC):
            v = ArithValue(vector.extract(inp_f32, static_position=[vi], dynamic_position=[]))
            vals.append(v)

        # Thread-local max
        local_max = c0_f32
        for vi in range_constexpr(VEC):
            local_max = local_max.maximumf(fx_math.absf(vals[vi]))

        # Warp-level max via shuffle_xor
        for sh_dist in SHUFFLE_DISTS:
            local_max = local_max.maximumf(local_max.shuffle_xor(sh_dist, WARP_SIZE))

        # Store warp max to LDS (lane 0 only), then inter-warp reduction
        if lane == 0:
            SmemPtr.store(s_wmax, local_max, [warp_id])
        gpu.barrier()

        if warp_id == 0:
            in_range = lane < NUM_WARPS
            lane_safe = in_range.select(lane, 0)
            v = SmemPtr.load(s_wmax, [lane_safe])
            v = in_range.select(v, c0_f32)
            for sh_dist in SHUFFLE_DISTS:
                v = v.maximumf(v.shuffle_xor(sh_dist, WARP_SIZE))
            if lane == 0:
                eps      = arith.constant(1e-12, type=T.f32)
                fp8_max  = arith.constant(FP8_MAX, type=T.f32)
                gmax     = v.maximumf(eps)
                scale    = gmax / fp8_max
                bid_i32  = ArithValue(fx.arith.index_cast(T.i32, bid))
                buffer_ops.buffer_store(scale, scale_rsrc, bid_i32)
                inv_sc   = fp8_max / gmax
                SmemPtr.store(s_wmax, inv_sc, [0])
        gpu.barrier()

        inv_scale = SmemPtr.load(s_wmax, [0])

        # Quantize and pack fp8 output
        bid_i32    = ArithValue(fx.arith.index_cast(T.i32, bid))
        col0       = thread_id * VEC
        fp8_byte_off = bid_i32 * K + col0

        for wg in range_constexpr(VEC // 4):
            base = wg * 4
            scaled = [vals[base + e] * inv_scale for e in range_constexpr(4)]
            packed = arith.constant(0, type=T.i32)
            packed = rocdl.cvt_pk_fp8_f32(T.i32, scaled[0], scaled[1], packed, 0)
            packed = rocdl.cvt_pk_fp8_f32(T.i32, scaled[2], scaled[3], packed, 1)
            word_off = fp8_byte_off + wg * 4
            buffer_ops.buffer_store(packed, out_rsrc, word_off, offset_is_bytes=True)

    @flyc.jit
    def launch_fp8_per_token_quantize(
        arg_out_fp8: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        c1 = 1
        launcher = kernel_fp8_per_token_quantize(arg_out_fp8, arg_out_scale, arg_inp)
        launcher.launch(
            grid=(m, c1, c1),
            block=(BLOCK_THREADS, c1, c1),
            stream=stream,
        )

    return launch_fp8_per_token_quantize


__all__ = ["compile_fp8_per_token_quantize"]
