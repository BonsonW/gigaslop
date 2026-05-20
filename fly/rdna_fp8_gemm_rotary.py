"""Fused fp8 GEMM + rotary embedding for RDNA4 (gfx120x, wave32).

Same compute as rdna_fp8_preshuffle_gemm but the epilogue applies rotary
embedding in-register before writing C, eliminating the DRAM round-trip.

  C[M,N] = rotary_embed(A[M,K] @ B[K,N])

Layout: N = 3 * nhead * head_dim  (QKV concatenated)
  cols [0,             nhead*head_dim): Q  — rotary applied to first rotary_dim cols per head
  cols [nhead*head_dim, 2*nhead*head_dim): K  — same
  cols [2*nhead*head_dim, N):            V  — written unchanged

sin/cos: [seqlen, rotary_half] fp32.  seqlen must be a power of 2.
Row m corresponds to sequence position  seq = m & (seqlen - 1).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


# =============================================================================
# Host-side helpers (identical to rdna_fp8_preshuffle_gemm)
# =============================================================================

def preshuffle_b_fp8(B_kn):
    import torch
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0, K0 = N // 16, K // 16
    B_view = B_kn.view(torch.uint8)
    B_reshaped = B_view.reshape(K0, 2, 8, N0, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()


def fp8_quantize_per_token(x_f32):
    import torch
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    x_scaled = (x_f32 / scale).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def fp8_quantize_per_channel(x_f32):
    import torch
    amax = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    x_scaled = (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale


# =============================================================================
# Kernel compiler
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_gemm_rotary(
    *,
    M: int,
    N: int,
    K: int,
    nhead: int,
    head_dim: int,
    rotary_dim: int,
    tile_m: int = None,
    tile_n: int = None,
    tile_k: int = 32,
    k_unroll: int = None,
    group_m: int = 8,
):
    rotary_half = rotary_dim // 2
    assert N == 3 * nhead * head_dim, f"N={N} must equal 3*nhead*head_dim={3*nhead*head_dim}"

    if tile_m is None:
        tile_m = 64 if M >= 256 else 32
    if tile_n is None:
        tile_n = 256 if M >= 256 else 128
    if k_unroll is None:
        k_unroll = 1 if M >= 256 else 2

    assert tile_n % head_dim == 0, f"tile_n={tile_n} must be multiple of head_dim={head_dim}"
    assert tile_n % (nhead * head_dim) == 0 or (nhead * head_dim) % tile_n == 0, \
        "tile_n must align with QKV chunk boundaries"

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
    def kernel_gemm_rotary(
        arg_c:            fx.Tensor,
        arg_a:            fx.Tensor,
        arg_b:            fx.Tensor,
        arg_scale_a:      fx.Tensor,
        arg_scale_b:      fx.Tensor,
        arg_sin:          fx.Tensor,
        arg_cos:          fx.Tensor,
        arg_grid_m:   fx.Int32,
        arg_seqlen:   fx.Int32,
    ):
        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

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

        a_rsrc       = buffer_ops.create_buffer_resource(arg_a,       max_size=True)
        b_rsrc       = buffer_ops.create_buffer_resource(arg_b,       max_size=True)
        c_rsrc       = buffer_ops.create_buffer_resource(arg_c,       max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)
        sin_rsrc     = buffer_ops.create_buffer_resource(arg_sin,     max_size=True)
        cos_rsrc     = buffer_ops.create_buffer_resource(arg_cos,     max_size=True)

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

        def _load_b_tile(k_tile_idx):
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

        zero_acc = fx.full(8, 0.0, fx.Float32)
        accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

        a_cur = _load_a_tile(0)
        b_cur = _load_b_tile(0)

        full_outer_iters = (num_k_tiles - 1) // k_unroll
        remainder = (num_k_tiles - 1) % k_unroll

        def _flatten_tile(tile):
            flat = []
            for rk_vecs in tile:
                flat.extend(rk_vecs)
            return flat

        def _unflatten_a(flat):
            out = []
            idx = 0
            for rk in range_constexpr(reg_k):
                row = []
                for rm in range_constexpr(wave_reg_m):
                    row.append(flat[idx]); idx += 1
                out.append(row)
            return out

        def _unflatten_b(flat):
            out = []
            idx = 0
            for rk in range_constexpr(reg_k):
                row = []
                for rn in range_constexpr(wave_reg_n):
                    row.append(flat[idx]); idx += 1
                out.append(row)
            return out

        n_a   = reg_k * wave_reg_m
        n_acc = wave_reg_m * wave_reg_n

        init_state = _flatten_tile(a_cur) + list(accs) + _flatten_tile(b_cur)

        if const_expr(full_outer_iters > 0):
            for iv, state in range(0, full_outer_iters * k_unroll, k_unroll, init=init_state):
                s_a    = _unflatten_a(list(state[:n_a]))
                s_accs = list(state[n_a : n_a + n_acc])
                s_b    = _unflatten_b(list(state[n_a + n_acc:]))
                for j in range_constexpr(k_unroll):
                    next_kt = iv + (j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    s_accs = _do_compute(s_accs, s_a, s_b)
                    s_a = _unflatten_a(_flatten_tile(a_next))
                    s_b = _unflatten_b(_flatten_tile(b_next))
                results = yield _flatten_tile(s_a) + list(s_accs) + _flatten_tile(s_b)

            a_cur = _unflatten_a(list(results[:n_a]))
            accs  = list(results[n_a : n_a + n_acc])
            b_cur = _unflatten_b(list(results[n_a + n_acc:]))

        if const_expr(remainder > 0):
            for j in range_constexpr(remainder):
                next_kt = full_outer_iters * k_unroll + j + 1
                a_next  = _load_a_tile(next_kt)
                b_next  = _load_b_tile(next_kt)
                accs    = _do_compute(accs, a_cur, b_cur)
                a_cur   = _unflatten_a(_flatten_tile(a_next))
                b_cur   = _unflatten_b(_flatten_tile(b_next))

        accs = _do_compute(accs, a_cur, b_cur)

        # ── Epilogue: scale, fused rotary, fp16 store ─────────────────────────
        # Rotary layout (matches openfish rotary_emb_hip):
        #   rotary_half = rotary_dim // 2
        #   x0 = head cols [0,          rotary_half)  — first half
        #   x1 = head cols [rotary_half, rotary_dim)  — second half
        #   x0_out[k] = x0[k]*cos[k] - x1[k]*sin[k]
        #   x1_out[k] = x0[k]*sin[k] + x1[k]*cos[k]
        #   sin/cos buffer stride = rotary_half (floats per seq row)
        #
        # WMMA blocks of WMMA_N=16 cols map to rotary as:
        #   companion_step = rotary_half // WMMA_N  (blocks between x0 and x1)
        #   For each rn: pos_in_head = (WMMA_N * rn) % head_dim  (compile-time)
        #   sin/cos rot index: rot_base = pos_in_head % rotary_half  (compile-time)
        #                      rotary_off = seq * rotary_half + rot_base + lane16

        base8 = klane * 8
        companion_step = rotary_half // WMMA_N   # compile-time, e.g. 32//16 = 2
        num_rot_blocks = rotary_half // WMMA_N   # unique sin/cos loads per (rm,si)

        # Pre-load scale_b for all N-columns this wave writes (constant per wave)
        sb_cache = []
        for rn in range_constexpr(wave_reg_n):
            g_col = tile_n0 + wave_n * (wave_reg_n * WMMA_N) + 16 * rn + lane16
            sb_cache.append(
                buffer_ops.buffer_load(scale_b_rsrc, g_col, vec_width=1, dtype=fx.Float32)
            )

        # Determine if this wave writes Q or K (cols < 2*nhead*head_dim) vs V
        wave_col_base     = tile_n0 + wave_n * (wave_reg_n * WMMA_N)
        wave_col_base_i32 = fx.arith.index_cast(fx.T.i32(), wave_col_base)
        qk_limit_i32      = fx.arith.constant(2 * nhead * head_dim, type=fx.T.i32())
        in_qk = fx.arith.cmpi(fx.arith.CmpIPredicate.slt, wave_col_base_i32, qk_limit_i32)

        rotary_half_cst = fx.arith.constant(rotary_half, type=fx.T.i32())
        lane16_i32      = fx.arith.index_cast(fx.T.i32(), lane16)

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m * (wave_reg_m * WMMA_M) + 16 * rm

            for si in range_constexpr(8):
                g_row = tile_m0 + wmma_m_off + base8 + si

                sa_val = buffer_ops.buffer_load(
                    scale_a_rsrc, g_row, vec_width=1, dtype=fx.Float32
                )

                g_row_i32 = fx.arith.index_cast(fx.T.i32(), g_row)
                seq_i32   = g_row_i32 % arg_seqlen

                # Preload sin/cos for each unique rot_base block (compile-time unrolled).
                # rot_base for block rb_idx = rb_idx * WMMA_N.
                sin_sc = []
                cos_sc = []
                for rb_idx in range_constexpr(num_rot_blocks):
                    rot_base_cst = fx.arith.constant(rb_idx * WMMA_N, type=fx.T.i32())
                    rotary_off   = seq_i32 * rotary_half_cst + rot_base_cst + lane16_i32
                    sin_sc.append(buffer_ops.buffer_load(sin_rsrc, rotary_off, vec_width=1, dtype=fx.Float32))
                    cos_sc.append(buffer_ops.buffer_load(cos_rsrc, rotary_off, vec_width=1, dtype=fx.Float32))

                for rn in range_constexpr(wave_reg_n):
                    idx   = rm * wave_reg_n + rn
                    g_col = tile_n0 + wave_n * (wave_reg_n * WMMA_N) + 16 * rn + lane16

                    val = accs[idx][si] * sa_val * sb_cache[rn]

                    pos_in_head = (WMMA_N * rn) % head_dim  # compile-time

                    if const_expr(pos_in_head < rotary_half):
                        # x0: first-half of rotary span
                        rb_idx   = (pos_in_head % rotary_half) // WMMA_N  # compile-time
                        sin_val  = sin_sc[rb_idx]
                        cos_val  = cos_sc[rb_idx]
                        companion = rm * wave_reg_n + rn + companion_step
                        x0 = val
                        x1 = accs[companion][si] * sa_val * sb_cache[rn + companion_step]
                        rotated  = x0 * cos_val - x1 * sin_val
                        val_fp16 = arith.truncf(fx.T.f16(), fx.arith.select(in_qk, rotated, x0))

                    elif const_expr(rotary_half <= pos_in_head < rotary_dim):
                        # x1: second-half of rotary span
                        rb_idx   = (pos_in_head - rotary_half) // WMMA_N  # compile-time
                        sin_val  = sin_sc[rb_idx]
                        cos_val  = cos_sc[rb_idx]
                        companion = rm * wave_reg_n + rn - companion_step
                        x0 = accs[companion][si] * sa_val * sb_cache[rn - companion_step]
                        x1 = val
                        rotated  = x0 * sin_val + x1 * cos_val
                        val_fp16 = arith.truncf(fx.T.f16(), fx.arith.select(in_qk, rotated, x1))

                    else:
                        # Non-rotary (only reachable when rotary_dim < head_dim)
                        val_fp16 = val.to(fx.Float16)

                    elem_off = g_row * N + g_col
                    buffer_ops.buffer_store(val_fp16, c_rsrc, elem_off)

    # ── Host launcher ─────────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_gemm_rotary(
        arg_c:        fx.Tensor,
        arg_a:        fx.Tensor,
        arg_b:        fx.Tensor,
        arg_scale_a:  fx.Tensor,
        arg_scale_b:  fx.Tensor,
        arg_sin:      fx.Tensor,
        arg_cos:      fx.Tensor,
        stream:   fx.Stream,
        m:        fx.Int32,
        seqlen:   fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_n
        launcher = kernel_gemm_rotary(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_sin, arg_cos,
            dyn_grid_m, seqlen,
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_fp8_gemm_rotary


__all__ = [
    "compile_fp8_gemm_rotary",
    "preshuffle_b_fp8",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
