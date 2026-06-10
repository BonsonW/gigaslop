"""FP16 → FP8 per-token quantize kernel for RDNA4 (gfx120x, wave32).

Quantizes a row-major fp16 tensor to fp8_e4m3fn with per-token (per-row) f32 scale.

  inp[M, K]  fp16  →  out_fp8[M, K]  fp8_e4m3fn bytes
                       scale_a[M]    f32  (amax / 448.0)

One block per row, BLOCK_THREADS threads per block.
Each thread loads VEC_PER_THREAD fp16 elements as a 128-bit vector.
Intra-warp max via XOR reduction; inter-warp max via LDS (NUM_WARPS slots).
All threads read all warp-max slots and compute inv_scale independently.

Grid:  (M, 1, 1)
Block: (BLOCK_THREADS=128, 1, 1) = 4 warps × 32 lanes

K must be a multiple of BLOCK_THREADS (= 128).
"""

import functools
import math

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

WARP_SIZE      = 32
BLOCK_THREADS  = 128              # 4 warps per block
NUM_WARPS      = BLOCK_THREADS // WARP_SIZE   # 4
FP8_MAX        = 448.0

SHUFFLE_DISTS = [1 << i for i in range(int(math.log2(WARP_SIZE)))]  # [1,2,4,8,16]


@functools.lru_cache(maxsize=32)
def compile_fp8_per_token_quantize(*, K: int):
    """Compile FP16 → FP8 per-token quantize kernel for RDNA4.

    K = number of fp16 elements per row.  Must satisfy K % BLOCK_THREADS == 0.

    Returns launcher(out_fp8, scale_a, inp_fp16, m, stream).
    """
    assert K % BLOCK_THREADS == 0, (
        f"K={K} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    )
    VEC_PER_THREAD = K // BLOCK_THREADS   # fp16 elements per thread (e.g. 8 for K=1024)
    assert VEC_PER_THREAD % 8 == 0 or VEC_PER_THREAD == 8, (
        f"VEC_PER_THREAD={VEC_PER_THREAD} must be 8 or a multiple of 8"
    )
    LOADS_PER_THREAD = max(1, VEC_PER_THREAD // 8)   # 128-bit buffer_loads per thread

    # LDS: NUM_WARPS f32 slots for inter-warp max + inv_scale broadcast (slot 0)
    lds_alloc    = SmemAllocator(None, global_sym_name="smem_per_token_quant")
    wmax_off     = lds_alloc._align(lds_alloc.ptr, 16)
    lds_alloc.ptr = wmax_off + NUM_WARPS * 4

    @flyc.kernel
    def kernel_fp8_per_token_quantize(
        arg_out_fp8:   fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp:       fx.Tensor,
    ):
        bid     = fx.block_idx.x
        tid     = fx.thread_idx.x
        lane    = tid % WARP_SIZE
        warp_id = tid // WARP_SIZE

        tid_i32     = ArithValue(arith.index_cast(T.i32, tid))
        bid_i32     = ArithValue(arith.index_cast(T.i32, bid))
        warp_id_i32 = ArithValue(arith.index_cast(T.i32, warp_id))

        inp_rsrc   = buffer_ops.create_buffer_resource(arg_inp,       max_size=True)
        out_rsrc   = buffer_ops.create_buffer_resource(arg_out_fp8,   max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(arg_out_scale, max_size=True)

        # LDS: NUM_WARPS f32 slots for per-warp max
        base_ptr = lds_alloc.get_base()
        s_wmax   = SmemPtr(base_ptr, wmax_off, T.f32, shape=(NUM_WARPS,))
        # Force memref materialization at the top level so it dominates all uses,
        # including those inside conditional blocks (if lane == 0, etc.).
        _        = s_wmax.get()

        # Thread-contiguous f16 base: bid * K + tid * VEC_PER_THREAD
        f16_base = bid_i32 * K + tid_i32 * VEC_PER_THREAD

        # ── Pass 1: load all data, compute local amax in f32 ─────────────────
        c0_f32    = arith.constant(0.0, type=T.f32)
        local_max = c0_f32
        all_vals  = []   # VEC_PER_THREAD f32 ArithValues

        for li in range_constexpr(LOADS_PER_THREAD):
            # 128-bit buffer_load (8 fp16 = 128 bits); offset in f16 element units
            vec_f16 = buffer_ops.buffer_load(
                inp_rsrc, f16_base + li * 8, vec_width=8, dtype=fx.Float16)
            # Extend vec<8xf16> → vec<8xf32> then extract scalars
            vec_f32_ty = T.vec(8, T.f32)
            vec_f32    = ArithValue(vec_f16).extf(vec_f32_ty)
            for vi in range_constexpr(8):
                v = ArithValue(mlir_vector.extract(
                    vec_f32, static_position=[vi], dynamic_position=[]))
                all_vals.append(v)
                local_max = local_max.maximumf(fx_math.absf(v))

        # ── Warp XOR reduction → warp-level max ──────────────────────────────
        for sh_dist in SHUFFLE_DISTS:
            local_max = local_max.maximumf(
                local_max.shuffle_xor(sh_dist, WARP_SIZE))

        # Lane 0 of each warp writes warp max to LDS slot [warp_id]
        if lane == 0:
            SmemPtr.store(s_wmax, local_max, [warp_id_i32])
        gpu.barrier()

        # ── Inter-warp reduce: all threads read all NUM_WARPS warp-max slots ──
        # Using constexpr integer indices avoids any conditional-scoped MLIR values.
        gmax = c0_f32
        for wi in range_constexpr(NUM_WARPS):
            wv   = SmemPtr.load(s_wmax, [wi])
            gmax = gmax.maximumf(wv)

        eps      = arith.constant(1e-12, type=T.f32)
        fp8_max  = arith.constant(FP8_MAX, type=T.f32)
        gmax     = gmax.maximumf(eps)
        inv_sc   = ArithValue(fp8_max) / gmax
        scale    = gmax / ArithValue(fp8_max)

        # All lane-0 threads write the same scale value — safe, no race.
        if lane == 0:
            buffer_ops.buffer_store(scale, scale_rsrc, bid_i32)

        # ── Pass 2: scale f32 values → pack fp8, store ───────────────────────
        fp8_base = bid_i32 * K + tid_i32 * VEC_PER_THREAD
        zero     = arith.constant(0, type=T.i32)

        for wg in range_constexpr(VEC_PER_THREAD // 4):
            base = wg * 4
            v0 = all_vals[base + 0] * inv_sc
            v1 = all_vals[base + 1] * inv_sc
            v2 = all_vals[base + 2] * inv_sc
            v3 = all_vals[base + 3] * inv_sc
            packed = rocdl.cvt_pk_fp8_f32(T.i32, v0, v1, zero, 0)
            packed = rocdl.cvt_pk_fp8_f32(T.i32, v2, v3, packed, 1)
            word_off = fp8_base + wg * 4
            buffer_ops.buffer_store(packed, out_rsrc, word_off, offset_is_bytes=True)

    @flyc.jit
    def launch_fp8_per_token_quantize(
        arg_out_fp8:   fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_inp:       fx.Tensor,
        m:             fx.Int32,
        stream:        fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            lds_alloc.finalized = False
            lds_alloc.finalize()

        c1 = 1
        launcher = kernel_fp8_per_token_quantize(arg_out_fp8, arg_out_scale, arg_inp)
        launcher.launch(
            grid=(m, c1, c1),
            block=(BLOCK_THREADS, c1, c1),
            stream=stream,
        )

    return launch_fp8_per_token_quantize


__all__ = ["compile_fp8_per_token_quantize"]
