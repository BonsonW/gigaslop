"""Fused dual FP8 GEMM + silu_mul + MXFP8 quantize for RDNA4 (gfx120x, wave32).

Single-dispatch variant of rdna_fp8_dual_gemm_silu: the silu_mul result is
quantized to MXFP8 (fp8_e4m3fn with per-32-element E8M0 block scales) directly
in the epilogue using LDS buffering — no intermediate fp16 global buffer.

  A[M, K]       fp8_e4m3fn  →  out_fp8[M, N]          fp8_e4m3fn bytes
  B_gate[…]                     out_scale[M, N//32]    uint8 E8M0 (row-major)
  B_up[…]
  scale_a, scale_b_gate, scale_b_up

Epilogue phases (after K-loop):
  Phase A: each thread writes its silu_mul f32 results to LDS [tile_m, tile_n].
  gpu.barrier()
  Phase B: each thread independently quantizes 4 row-blocks of 32 elements each
           from LDS to FP8 + E8M0, stores both to global memory.

Grid:  (grid_m * grid_n, 1, 1)  — same as rdna_fp8_dual_gemm_silu
Block: (THREADS_PER_BLOCK, 1, 1)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from rdna_fp8_dual_gemm_silu import preshuffle_b_fp8, split_and_preshuffle_fc1

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

FP8_MAX = 448.0
FP8_HEADROOM = 8  # exponent headroom for fp8 (vs 2 for fp4)


@functools.lru_cache(maxsize=64)
def compile_fp8_dual_gemm_silu_mxfp8(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 32,
    tile_n: int = None,
    tile_k: int = 32,
    k_unroll: int = None,
    group_m: int = 8,
):
    """Compile fused dual FP8 GEMM + silu_mul + MXFP8 quantize for RDNA4.

    N = K_out = inter_dim (half of fc1 output width).
    tile_n must be divisible by 32 (MXFP8 block size).

    Returns launcher(out_fp8, out_scale, a, b_gate, b_up,
                     scale_a, scale_b_gate, scale_b_up, stream, m).
    """
    if tile_n is None:
        tile_n = 256 if M >= 256 else 128
    if k_unroll is None:
        k_unroll = 1 if M >= 256 else 2

    assert tile_n % 32 == 0, f"tile_n={tile_n} must be divisible by 32 for MXFP8"

    WAVE_SIZE = 32
    assert tile_m % WMMA_M == 0
    assert tile_n % WMMA_N == 0
    assert tile_k % WMMA_K == 0
    assert M % tile_m == 0
    assert N % tile_n == 0
    assert K % tile_k == 0

    reg_m = tile_m // WMMA_M
    reg_n = tile_n // WMMA_N
    reg_k = tile_k // WMMA_K

    if tile_m >= 128 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 64 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_n >= 256:
        waves_m, waves_n = 1, 2
    elif tile_m >= 64:
        waves_m, waves_n = 2, 1
    elif tile_n >= 128:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m = reg_m // waves_m
    wave_reg_n = reg_n // waves_n

    num_k_tiles = K // tile_k
    grid_m = M // tile_m
    grid_n = N // tile_n

    K0_total = K // 16

    B_KPACK     = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0    = 2 * 16 * B_KPACK
    B_STRIDE_N0    = K0_total * B_STRIDE_K0

    # LDS layout:
    #   [0, tile_m * tile_n * 4)  — f32 act_val tile for MXFP8 quantization (Phase A)
    # LDS size: tile_m * tile_n * 4 bytes  (e.g. 32*256*4 = 32KB)
    N_BLOCKS_PER_ROW = tile_n // 32           # e.g. 8
    PAIRS_PER_THREAD = (tile_m * N_BLOCKS_PER_ROW) // THREADS_PER_BLOCK  # e.g. 4
    assert PAIRS_PER_THREAD > 0, "tile_m * (tile_n//32) must be >= THREADS_PER_BLOCK"

    arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=arch)
    tile_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = tile_offset + tile_m * tile_n * 4

    @flyc.kernel
    def kernel_dual_gemm_silu_mxfp8(
        arg_out_fp8: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b_gate: fx.Tensor,
        arg_b_up: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b_gate: fx.Tensor,
        arg_scale_b_up: fx.Tensor,
        arg_grid_m: fx.Int32,
    ):
        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        # L2 cache swizzle
        pid_i32      = fx.arith.index_cast(fx.T.i32(), pid)
        group_m_cst  = fx.arith.constant(group_m, type=fx.T.i32())
        grid_n_cst   = fx.arith.constant(grid_n,  type=fx.T.i32())
        eff_gm       = fx.arith.select(
            fx.arith.cmpi(fx.arith.CmpIPredicate.slt, arg_grid_m, group_m_cst),
            arg_grid_m, group_m_cst,
        )
        num_in_group = eff_gm * grid_n_cst
        group_id     = pid_i32 // num_in_group
        pid_in_group = pid_i32 % num_in_group
        bid_m_i32    = group_id * eff_gm + pid_in_group % eff_gm
        bid_n_i32    = pid_in_group // eff_gm
        bid_m = fx.arith.index_cast(fx.T.index(), bid_m_i32)
        bid_n = fx.arith.index_cast(fx.T.index(), bid_n_i32)

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        tile_m0     = bid_m * tile_m
        tile_n0     = bid_n * tile_n
        tile_m0_i32 = ArithValue(bid_m_i32) * tile_m
        tile_n0_i32 = ArithValue(bid_n_i32) * tile_n

        a_rsrc          = buffer_ops.create_buffer_resource(arg_a,          max_size=True)
        b_gate_rsrc     = buffer_ops.create_buffer_resource(arg_b_gate,     max_size=True)
        b_up_rsrc       = buffer_ops.create_buffer_resource(arg_b_up,       max_size=True)
        out_rsrc        = buffer_ops.create_buffer_resource(arg_out_fp8,    max_size=True)
        out_scale_rsrc  = buffer_ops.create_buffer_resource(arg_out_scale,  max_size=True)
        scale_a_rsrc    = buffer_ops.create_buffer_resource(arg_scale_a,    max_size=True)
        scale_b_gate_rsrc = buffer_ops.create_buffer_resource(arg_scale_b_gate, max_size=True)
        scale_b_up_rsrc   = buffer_ops.create_buffer_resource(arg_scale_b_up,   max_size=True)

        # LDS tile for Phase A → Phase B
        base_ptr = allocator.get_base()
        s_tile = SmemPtr(base_ptr, tile_offset, T.f32, shape=(tile_m * tile_n,))
        s_tile.get()

        def _load_a_tile(k_tile_idx):
            a_vecs = []
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                col_base = k_tile_idx * tile_k + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    byte_off  = row * K + col_base
                    dword_off = byte_off // 4
                    a_raw = buffer_ops.buffer_load(a_rsrc, dword_off, vec_width=2, dtype=fx.Int32)
                    rk_vecs.append(a_raw)
                a_vecs.append(rk_vecs)
            return a_vecs

        def _load_b_tile(k_tile_idx, b_rsrc):
            b_vecs = []
            n0_base = tile_n0 // 16 + wave_n * wave_reg_n
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                k0 = k_tile_idx * reg_k + rk
                for rn in range_constexpr(wave_reg_n):
                    n0       = n0_base + rn
                    byte_off = (
                        n0 * B_STRIDE_N0
                        + k0 * B_STRIDE_K0
                        + klane * B_STRIDE_KLANE
                        + lane16 * B_STRIDE_NLANE
                    )
                    dword_off = byte_off // 4
                    b_raw = buffer_ops.buffer_load(b_rsrc, dword_off, vec_width=2, dtype=fx.Int32)
                    rk_vecs.append(b_raw)
                b_vecs.append(rk_vecs)
            return b_vecs

        def _do_compute(accs_in, a_vecs, b_vecs):
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n):
                        idx = rm * wave_reg_n + rn
                        new_accs[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            new_accs[idx].type,
                            a_vecs[rk][rm],
                            b_vecs[rk][rn],
                            new_accs[idx],
                        ).result
            return new_accs

        zero_acc  = fx.full(8, 0.0, fx.Float32)
        gate_accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]
        up_accs   = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

        a_cur      = _load_a_tile(0)
        b_gate_cur = _load_b_tile(0, b_gate_rsrc)
        b_up_cur   = _load_b_tile(0, b_up_rsrc)

        full_outer_iters = (num_k_tiles - 1) // k_unroll
        remainder        = (num_k_tiles - 1) % k_unroll

        def _flatten_tile(tile):
            flat = []
            for rk_vecs in tile:
                flat.extend(rk_vecs)
            return flat

        def _unflatten_a(flat):
            out = []
            idx = 0
            for _rk in range_constexpr(reg_k):
                row = []
                for _rm in range_constexpr(wave_reg_m):
                    row.append(flat[idx])
                    idx += 1
                out.append(row)
            return out

        def _unflatten_b(flat):
            out = []
            idx = 0
            for _rk in range_constexpr(reg_k):
                row = []
                for _rn in range_constexpr(wave_reg_n):
                    row.append(flat[idx])
                    idx += 1
                out.append(row)
            return out

        n_a   = reg_k * wave_reg_m
        n_acc = wave_reg_m * wave_reg_n
        n_b   = reg_k * wave_reg_n

        init_state = (
            _flatten_tile(a_cur)
            + list(gate_accs)
            + list(up_accs)
            + _flatten_tile(b_gate_cur)
            + _flatten_tile(b_up_cur)
        )

        if const_expr(full_outer_iters > 0):
            for iv, state in range(0, full_outer_iters * k_unroll, k_unroll, init=init_state):
                s_a      = _unflatten_a(list(state[:n_a]))
                s_gate   = list(state[n_a            : n_a + n_acc])
                s_up     = list(state[n_a + n_acc    : n_a + 2 * n_acc])
                s_b_gate = _unflatten_b(list(state[n_a + 2 * n_acc         : n_a + 2 * n_acc + n_b]))
                s_b_up   = _unflatten_b(list(state[n_a + 2 * n_acc + n_b   :]))

                for j in range_constexpr(k_unroll):
                    next_kt     = iv + (j + 1)
                    a_next      = _load_a_tile(next_kt)
                    b_gate_next = _load_b_tile(next_kt, b_gate_rsrc)
                    b_up_next   = _load_b_tile(next_kt, b_up_rsrc)
                    s_gate = _do_compute(s_gate, s_a, s_b_gate)
                    s_up   = _do_compute(s_up,   s_a, s_b_up)
                    s_a      = _unflatten_a(_flatten_tile(a_next))
                    s_b_gate = _unflatten_b(_flatten_tile(b_gate_next))
                    s_b_up   = _unflatten_b(_flatten_tile(b_up_next))

                results = yield (
                    _flatten_tile(s_a)
                    + list(s_gate)
                    + list(s_up)
                    + _flatten_tile(s_b_gate)
                    + _flatten_tile(s_b_up)
                )

            a_cur      = _unflatten_a(list(results[:n_a]))
            gate_accs  = list(results[n_a            : n_a + n_acc])
            up_accs    = list(results[n_a + n_acc    : n_a + 2 * n_acc])
            b_gate_cur = _unflatten_b(list(results[n_a + 2 * n_acc         : n_a + 2 * n_acc + n_b]))
            b_up_cur   = _unflatten_b(list(results[n_a + 2 * n_acc + n_b   :]))

        if const_expr(remainder > 0):
            for j in range_constexpr(remainder):
                next_kt     = full_outer_iters * k_unroll + j + 1
                a_next      = _load_a_tile(next_kt)
                b_gate_next = _load_b_tile(next_kt, b_gate_rsrc)
                b_up_next   = _load_b_tile(next_kt, b_up_rsrc)
                gate_accs = _do_compute(gate_accs, a_cur, b_gate_cur)
                up_accs   = _do_compute(up_accs,   a_cur, b_up_cur)
                a_cur      = _unflatten_a(_flatten_tile(a_next))
                b_gate_cur = _unflatten_b(_flatten_tile(b_gate_next))
                b_up_cur   = _unflatten_b(_flatten_tile(b_up_next))

        gate_accs = _do_compute(gate_accs, a_cur, b_gate_cur)
        up_accs   = _do_compute(up_accs,   a_cur, b_up_cur)

        # ── Epilogue Phase A: silu_mul → f32 LDS tile ──────────────────────
        neg_log2e = arith.constant(-1.4426950408889634, type=fx.T.f32())
        c1_f32    = arith.constant(1.0, type=fx.T.f32())
        base8 = klane * 8

        sb_gate_cache = []
        sb_up_cache   = []
        for rn in range_constexpr(wave_reg_n):
            g_col = tile_n0 + wave_n * (wave_reg_n * WMMA_N) + 16 * rn + lane16
            sb_gate_cache.append(buffer_ops.buffer_load(scale_b_gate_rsrc, g_col, vec_width=1, dtype=fx.Float32))
            sb_up_cache.append(  buffer_ops.buffer_load(scale_b_up_rsrc,   g_col, vec_width=1, dtype=fx.Float32))

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m * (wave_reg_m * WMMA_M) + 16 * rm
            sa_cache = []
            for si in range_constexpr(8):
                g_row_si = tile_m0 + wmma_m_off + base8 + si
                sa_cache.append(buffer_ops.buffer_load(scale_a_rsrc, g_row_si, vec_width=1, dtype=fx.Float32))

            for rn in range_constexpr(wave_reg_n):
                idx        = rm * wave_reg_n + rn
                wmma_n_off = wave_n * (wave_reg_n * WMMA_N) + 16 * rn

                sb_gate = sb_gate_cache[rn]
                sb_up   = sb_up_cache[rn]

                for si in range_constexpr(8):
                    sa       = sa_cache[si]
                    gate_val = ArithValue(gate_accs[idx][si]) * sa * sb_gate
                    up_val   = ArithValue(up_accs[idx][si])   * sa * sb_up

                    # SiLU(gate) * up
                    emu     = ArithValue(rocdl.exp2(fx.T.f32(), gate_val * neg_log2e))
                    sig     = ArithValue(rocdl.rcp(fx.T.f32(), c1_f32 + emu))
                    act_val = gate_val * sig * up_val

                    # Write to LDS: [lds_row, lds_col] = [wmma_m_off + klane*8 + si, wmma_n_off + lane16]
                    lds_row = wmma_m_off + base8 + si   # 0..tile_m-1 (compile-time for si/rm/klane)
                    lds_col = wmma_n_off + lane16       # 0..tile_n-1 (lane16 is runtime)
                    lds_idx = lds_row * tile_n + lds_col
                    SmemPtr.store(s_tile, act_val, [lds_idx])

        gpu.barrier()

        # ── Epilogue Phase B: per-32-element MXFP8 quantize from LDS ───────
        # Each thread independently processes PAIRS_PER_THREAD row-block pairs.
        # pair_idx → row (0..tile_m-1), n_block (0..N_BLOCKS_PER_ROW-1)
        c0_i32  = arith.constant(0, type=T.i32)
        c0_f32  = arith.constant(0.0, type=T.f32)
        thread_id = ArithValue(tid)

        for pair_iter in range_constexpr(PAIRS_PER_THREAD):
            pair_idx = thread_id + pair_iter * THREADS_PER_BLOCK
            row      = pair_idx // N_BLOCKS_PER_ROW
            n_block  = pair_idx %  N_BLOCKS_PER_ROW

            # Load 32 f32 values from LDS and find local max
            local_max = c0_f32
            vals = []
            for e in range_constexpr(32):
                lds_idx = row * tile_n + n_block * 32 + e
                v = SmemPtr.load(s_tile, [lds_idx])
                vals.append(v)
                local_max = local_max.maximumf(fx_math.absf(v))

            # E8M0 scale — same logic as silu_and_mul_fq.py lines 267–275
            exp_field = ((local_max.bitcast(T.i32) + 0x200000) & 0xFF800000) >> 23
            e8m0      = arith.maxsi(
                exp_field - arith.constant(FP8_HEADROOM, type=T.i32),
                c0_i32,
            )
            quant_scale = ((arith.constant(254, type=T.i32) - e8m0) << 23).bitcast(T.f32)
            e8m0_byte   = arith.trunci(T.i8, e8m0)

            # Global output position
            g_row = tile_m0_i32 + row
            g_col = tile_n0_i32 + n_block * 32

            # Pack 32 f32 → 8 i32 (4 fp8 bytes each) via rocdl.cvt_pk_fp8_f32
            fp8_byte_off = g_row * N + g_col
            for wg in range_constexpr(8):
                base   = wg * 4
                packed = c0_i32
                packed = rocdl.cvt_pk_fp8_f32(T.i32, vals[base] * quant_scale,     vals[base + 1] * quant_scale, packed, 0)
                packed = rocdl.cvt_pk_fp8_f32(T.i32, vals[base + 2] * quant_scale, vals[base + 3] * quant_scale, packed, 1)
                buffer_ops.buffer_store(packed, out_rsrc, fp8_byte_off + wg * 4, offset_is_bytes=True)

            # E8M0 scale: row-major [M, N//32], uint8
            scale_col = tile_n0_i32 // 32 + n_block
            scale_off = g_row * (N // 32) + scale_col
            buffer_ops.buffer_store(e8m0_byte, out_scale_rsrc, scale_off, offset_is_bytes=True)

    @flyc.jit
    def launch_dual_gemm_silu_mxfp8(
        arg_out_fp8: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b_gate: fx.Tensor,
        arg_b_up: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b_gate: fx.Tensor,
        arg_scale_b_up: fx.Tensor,
        stream: fx.Stream,
        m: fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_n
        launcher = kernel_dual_gemm_silu_mxfp8(
            arg_out_fp8, arg_out_scale,
            arg_a, arg_b_gate, arg_b_up,
            arg_scale_a, arg_scale_b_gate, arg_scale_b_up,
            dyn_grid_m,
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_dual_gemm_silu_mxfp8


__all__ = [
    "compile_fp8_dual_gemm_silu_mxfp8",
    "preshuffle_b_fp8",
    "split_and_preshuffle_fc1",
]
