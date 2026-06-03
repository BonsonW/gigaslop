"""Fused dual FP8 GEMM + silu_mul for RDNA4 (gfx120x, wave32).

Replaces fc1 GEMM + separate silu_mul for GatedMLP.

  out[M, N] = silu(A[M,K] @ B_gate[K,N]) * (A[M,K] @ B_up[K,N])

Both A, B_gate, B_up are fp8_e4m3fn.  Output is fp16.
Scales: per-token scale_a[M] for A, per-channel scale_b_gate[N] / scale_b_up[N] for weights.

B_gate and B_up must each be preshuffled to [N0, K0, KLane=2, NLane=16, KPack=8] bytes
using preshuffle_b_fp8() (or split_and_preshuffle_fc1() for a combined fc1 weight).

Grid:  (grid_m * grid_n, 1, 1)
Block: (THREADS_PER_BLOCK, 1, 1)

Optimisations vs the naive dual-GEMM loop:
1. Interleaved WMMA — gate[i] and up[i] WMMAs are emitted back-to-back so the
   GPU's WMMA pipeline can overlap independent dependency chains and hide latency.
2. k_unroll=2 for all M — the dual-GEMM loop body is 2x bigger so the extra
   unrolling pays off for pipeline fill even at large M.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


# =============================================================================
# Host-side preshuffle / weight-prep helpers
# =============================================================================

def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K, N] fp8 for WMMA B operand layout.

    Layout: [N0, K0, KLane=2, NLane=16, KPack=8] bytes.
    """
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_view = B_kn.view(torch.uint8)
    B_reshaped = B_view.reshape(K // 16, 2, 8, N // 16, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()  # [N0, K0, 2, 16, 8]


def split_and_preshuffle_fc1(W_kn):
    """Split fc1 weight W[K_in, 2*N] into gate/up halves and preshuffle both.

    Returns (B_gate_shuf, B_up_shuf) ready for compile_fp8_dual_gemm_silu.
    """
    K, N2 = W_kn.shape
    N = N2 // 2
    return (
        preshuffle_b_fp8(W_kn[:, :N].contiguous()),
        preshuffle_b_fp8(W_kn[:, N:].contiguous()),
    )


# =============================================================================
# Kernel compiler
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_dual_gemm_silu(
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
    """Compile fused dual FP8 GEMM + silu_mul for RDNA4.

    N = K_out = inter_dim (half of fc1 output width).
    K = K_in  = model dim.

    Returns launcher(c, a, b_gate, b_up, scale_a, scale_b_gate, scale_b_up, stream, m).
    """
    if tile_n is None:
        tile_n = 128  # dual GEMM has 2× accumulator pressure; tile_n=256 overflows 256 VGPR limit
    if k_unroll is None:
        k_unroll = 2  # always 2 — dual GEMM loop body is big enough to benefit

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

    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0 = 2 * 16 * B_KPACK
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    @flyc.kernel
    def kernel_dual_gemm_silu(
        arg_c: fx.Tensor,
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
        lane = tid % 32
        lane16 = lane % 16
        klane = lane // 16

        pid_i32     = fx.arith.index_cast(fx.T.i32(), pid)
        group_m_cst = fx.arith.constant(group_m, type=fx.T.i32())
        grid_n_cst  = fx.arith.constant(grid_n,  type=fx.T.i32())
        eff_gm      = fx.arith.select(
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

        tile_m0 = bid_m * tile_m
        tile_n0 = bid_n * tile_n

        a_rsrc            = buffer_ops.create_buffer_resource(arg_a,          max_size=True)
        b_gate_rsrc       = buffer_ops.create_buffer_resource(arg_b_gate,     max_size=True)
        b_up_rsrc         = buffer_ops.create_buffer_resource(arg_b_up,       max_size=True)
        c_rsrc            = buffer_ops.create_buffer_resource(arg_c,          max_size=True)
        scale_a_rsrc      = buffer_ops.create_buffer_resource(arg_scale_a,    max_size=True)
        scale_b_gate_rsrc = buffer_ops.create_buffer_resource(arg_scale_b_gate, max_size=True)
        scale_b_up_rsrc   = buffer_ops.create_buffer_resource(arg_scale_b_up, max_size=True)

        def _load_a_tile(k_tile_idx):
            a_vecs = []
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                col_base = k_tile_idx * tile_k + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    byte_off = row * K + col_base
                    a_raw = buffer_ops.buffer_load(a_rsrc, byte_off // 4, vec_width=2, dtype=fx.Int32)
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
                    n0 = n0_base + rn
                    byte_off = (
                        n0 * B_STRIDE_N0
                        + k0 * B_STRIDE_K0
                        + klane * B_STRIDE_KLANE
                        + lane16 * B_STRIDE_NLANE
                    )
                    b_raw = buffer_ops.buffer_load(b_rsrc, byte_off // 4, vec_width=2, dtype=fx.Int32)
                    rk_vecs.append(b_raw)
                b_vecs.append(rk_vecs)
            return b_vecs

        def _do_compute_both(gate_accs_in, up_accs_in, a_vecs, b_gate_vecs, b_up_vecs):
            """Interleave gate[i] and up[i] WMMAs — hides WMMA pipeline latency."""
            new_gate = list(gate_accs_in)
            new_up   = list(up_accs_in)
            for rk in range_constexpr(reg_k):
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n):
                        idx = rm * wave_reg_n + rn
                        new_gate[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            new_gate[idx].type,
                            a_vecs[rk][rm], b_gate_vecs[rk][rn],
                            new_gate[idx],
                        ).result
                        new_up[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            new_up[idx].type,
                            a_vecs[rk][rm], b_up_vecs[rk][rn],
                            new_up[idx],
                        ).result
            return new_gate, new_up

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
            out, idx = [], 0
            for _rk in range_constexpr(reg_k):
                row = []
                for _rm in range_constexpr(wave_reg_m):
                    row.append(flat[idx]); idx += 1
                out.append(row)
            return out

        def _unflatten_b(flat):
            out, idx = [], 0
            for _rk in range_constexpr(reg_k):
                row = []
                for _rn in range_constexpr(wave_reg_n):
                    row.append(flat[idx]); idx += 1
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
                s_gate   = list(state[n_a              : n_a + n_acc])
                s_up     = list(state[n_a + n_acc      : n_a + 2 * n_acc])
                s_b_gate = _unflatten_b(list(state[n_a + 2 * n_acc         : n_a + 2 * n_acc + n_b]))
                s_b_up   = _unflatten_b(list(state[n_a + 2 * n_acc + n_b   :]))

                for j in range_constexpr(k_unroll):
                    next_kt     = iv + (j + 1)
                    a_next      = _load_a_tile(next_kt)
                    b_gate_next = _load_b_tile(next_kt, b_gate_rsrc)
                    b_up_next   = _load_b_tile(next_kt, b_up_rsrc)
                    s_gate, s_up = _do_compute_both(s_gate, s_up, s_a, s_b_gate, s_b_up)
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
            gate_accs  = list(results[n_a              : n_a + n_acc])
            up_accs    = list(results[n_a + n_acc      : n_a + 2 * n_acc])
            b_gate_cur = _unflatten_b(list(results[n_a + 2 * n_acc         : n_a + 2 * n_acc + n_b]))
            b_up_cur   = _unflatten_b(list(results[n_a + 2 * n_acc + n_b   :]))

        if const_expr(remainder > 0):
            for j in range_constexpr(remainder):
                next_kt     = full_outer_iters * k_unroll + j + 1
                a_next      = _load_a_tile(next_kt)
                b_gate_next = _load_b_tile(next_kt, b_gate_rsrc)
                b_up_next   = _load_b_tile(next_kt, b_up_rsrc)
                gate_accs, up_accs = _do_compute_both(gate_accs, up_accs, a_cur, b_gate_cur, b_up_cur)
                a_cur      = _unflatten_a(_flatten_tile(a_next))
                b_gate_cur = _unflatten_b(_flatten_tile(b_gate_next))
                b_up_cur   = _unflatten_b(_flatten_tile(b_up_next))

        gate_accs, up_accs = _do_compute_both(gate_accs, up_accs, a_cur, b_gate_cur, b_up_cur)

        # Epilogue: silu(gate) * up → fp16
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
                sb_gate    = sb_gate_cache[rn]
                sb_up      = sb_up_cache[rn]

                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + si
                    g_col = tile_n0 + wmma_n_off + lane16
                    sa    = sa_cache[si]

                    gate_val = ArithValue(gate_accs[idx][si]) * sa * sb_gate
                    up_val   = ArithValue(up_accs[idx][si])   * sa * sb_up

                    emu     = ArithValue(rocdl.exp2(fx.T.f32(), gate_val * neg_log2e))
                    sig     = ArithValue(rocdl.rcp(fx.T.f32(), c1_f32 + emu))
                    act_val = gate_val * sig * up_val

                    val_fp16 = arith.truncf(fx.T.f16(), act_val)
                    elem_off = g_row * N + g_col
                    buffer_ops.buffer_store(val_fp16, c_rsrc, elem_off)

    @flyc.jit
    def launch_dual_gemm_silu(
        arg_c: fx.Tensor,
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
        launcher = kernel_dual_gemm_silu(
            arg_c, arg_a, arg_b_gate, arg_b_up,
            arg_scale_a, arg_scale_b_gate, arg_scale_b_up,
            dyn_grid_m,
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_dual_gemm_silu


__all__ = [
    "compile_fp8_dual_gemm_silu",
    "preshuffle_b_fp8",
    "split_and_preshuffle_fc1",
]
