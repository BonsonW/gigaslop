"""FP8 Unfactored LSTM GEMM Kernel for RDNA4 (gfx1201, wave32).

Single GEMM fused with LSTM epilogue.  Avoids the factored two-phase approach
and the LDS intermediate quantize step:

  gates = hh_fp8[B,H] @ W_fused_fp8[H,4H]^T  +  up_bias[4H] + ih_t[B,4H]
  W_fused[4H,H] = up_weight[4H,R] @ dn_weight[R,H]   (precomputed offline)

  i,f,g,o = chunk4(gates, dim=H)
  i = clamp(0.2*i+0.5, 0, 1); f = same; g = clamp(g,-1,1); o = same
  c_new   = f*c + i*g
  h_new   = o*tanh(c_new)  → fp8 e4m3 output (fixed scale 1/448)

h_new is provably bounded to [-1,1] (o∈[0,1], tanh∈[-1,1]), so a fixed scale of
1/448 maps it exactly onto e4m3's [-448,448].  The per-token quantize kernel is
thus fused away: each thread writes one fp8 byte directly from the epilogue, and
the recurrent scale_hh is a host-side constant (1/448).

No LDS used.  4 accumulator sets (i,f,g,o) run in parallel over K=H.

Grid: (B/tile_m * H/tile_n_h, 1, 1)
Block: (64 threads = 2 waves x 32 lanes)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

WAVE_SIZE = 32
FP8_MAX   = 448.0


# =============================================================================
# Host-side helpers
# =============================================================================

def preshuffle_b_fp8(B_kn):
    """Preshuffle fp8 B[K,N] → [N0, K0, KLane=2, NLane=16, KPack=8]."""
    import torch
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_view = B_kn.view(torch.uint8)
    B_r = B_view.reshape(K // 16, 2, 8, N // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()   # [N0, K0, 2, 16, 8]


def fp8_quantize_per_token(x_f32):
    import torch
    amax  = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    return (x_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale.squeeze(-1)


def fp8_quantize_per_channel(x_f32):
    import torch
    amax  = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    return (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def make_ih_t_interleaved(ih_t_b_fh):
    """Permute ih_t from [B, 4H] to [B, H, 4] for vectorized epilogue loads.

    The kernel expects arg_ih_t in [B, H, 4] layout so all 4 gate values at a
    given (batch_row, h_col) are contiguous and can be loaded with one vec4_f16
    instruction instead of 4 strided scalar loads.
    """
    import torch
    B, FH = ih_t_b_fh.shape
    H = FH // 4
    return ih_t_b_fh.reshape(B, 4, H).permute(0, 2, 1).contiguous()  # [B, H, 4]


def make_w_fused(dn_weight_rh, up_weight_4hr):
    """Compute and quantize the fused weight W_fused = up_weight @ dn_weight.

    dn_weight_rh:  [R, H]  f32 — down-projection (R rows, H cols)
    up_weight_4hr: [4H, R] f32 — up-projection

    Returns: (w_fused_shuf, scale_wf) where w_fused_shuf is preshuffled fp8 [H, 4H]
             and scale_wf is per-channel f32 [4H].
    """
    import torch
    # W_fused[4H, H] = up_weight[4H, R] @ dn_weight[R, H]
    w_fused_4h_h = up_weight_4hr @ dn_weight_rh   # [4H, H]
    # transpose to [H, 4H] for preshuffle (K=H, N=4H)
    w_fused_h_4h = w_fused_4h_h.t().contiguous()  # [H, 4H]
    w_fp8, scale_wf = fp8_quantize_per_channel(w_fused_h_4h)
    return preshuffle_b_fp8(w_fp8), scale_wf


# =============================================================================
# Kernel compiler
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_unfactored_lstm_gemm(
    *,
    B: int,
    H: int,
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k: int = 32,
    k_unroll: int = 2,
    group_m: int = 1,
):
    """Compile unfactored LSTM GEMM for gfx1201.

    W_fused[H,4H] must be precomputed (up @ dn), quantized per-channel, and
    preshuffled with make_w_fused / preshuffle_b_fp8.

    Returns launcher(h_fp16_out, c_inout, hh, scale_hh,
                      w_fused_shuf, scale_wf, bias, ih_t, stream, m).
    """
    FH = 4 * H

    assert B % tile_m == 0
    assert H % tile_n_h == 0
    assert H % tile_k == 0
    assert tile_n_h % WMMA_N == 0
    assert tile_k % WMMA_K == 0

    reg_m      = tile_m   // WMMA_M        # 2
    reg_k      = tile_k   // WMMA_K        # 2
    num_k_tiles = H // tile_k              # 32

    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE   # 64
    wave_reg_m        = reg_m   // waves_m  # 2
    wave_reg_n_h      = (tile_n_h // WMMA_N) // waves_n   # 1

    grid_m = B // tile_m
    grid_h = H // tile_n_h

    # ── B-strides: W_fused fp8 [H, 4H] preshuffled as [N0, K0, KLane=2, NLane=16, KPack=8] ─
    # K=H, N=4H  →  K0=H//16, N0=4H//16
    K0_H            = H // 16
    B_KPACK         = 8
    B_STRIDE_NLANE  = B_KPACK           # 8
    B_STRIDE_KLANE  = 16 * B_KPACK      # 128
    B_STRIDE_K0     = 2 * 16 * B_KPACK  # 256
    B_STRIDE_N0     = K0_H * B_STRIDE_K0   # 64 * 256 = 16384
    GATE_N0_STRIDE  = H // 16           # 64

    @flyc.kernel
    def kernel_unfactored_lstm(
        arg_h_fp8_out:  fx.Tensor,   # [B, H]   fp8 (uint8) — fixed scale 1/448
        arg_c_inout:    fx.Tensor,   # [B, H]   f32
        arg_hh:         fx.Tensor,   # [B, H]   fp8
        arg_scale_hh:   fx.Tensor,   # [B]      f32
        arg_w_fused:    fx.Tensor,   # preshuffled fp8 [H, 4H]
        arg_scale_wf:   fx.Tensor,   # [4H]     f32
        arg_bias:       fx.Tensor,   # [4H]     f32
        arg_ih_t:       fx.Tensor,   # [B, 4H]  f16
        arg_grid_m:     fx.Int32,
    ):
        # ── Thread / block indices ───────────────────────────────────────────
        tid     = gpu.thread_id("x")
        pid     = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        pid_i32   = fx.arith.index_cast(fx.T.i32(), pid)
        grid_h_c  = fx.arith.constant(grid_h, type=fx.T.i32())
        group_m_c = fx.arith.constant(group_m, type=fx.T.i32())
        eff_gm    = fx.arith.select(
            fx.arith.cmpi(fx.arith.CmpIPredicate.slt, arg_grid_m, group_m_c),
            arg_grid_m, group_m_c,
        )
        num_in_group = eff_gm * grid_h_c
        group_id     = pid_i32 // num_in_group
        pid_in_group = pid_i32 % num_in_group
        bid_m_i32    = group_id * eff_gm + pid_in_group % eff_gm
        bid_h_i32    = pid_in_group // eff_gm
        bid_m        = fx.arith.index_cast(fx.T.index(), bid_m_i32)
        bid_h        = fx.arith.index_cast(fx.T.index(), bid_h_i32)

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        tile_m0  = bid_m * tile_m
        tile_nh0 = bid_h * tile_n_h

        # ── Buffer resources ─────────────────────────────────────────────────
        hh_rsrc  = buffer_ops.create_buffer_resource(arg_hh,        max_size=True)
        wf_rsrc  = buffer_ops.create_buffer_resource(arg_w_fused,   max_size=True)
        c_rsrc   = buffer_ops.create_buffer_resource(arg_c_inout,   max_size=True)
        h_rsrc   = buffer_ops.create_buffer_resource(arg_h_fp8_out, max_size=True)
        shh_rsrc = buffer_ops.create_buffer_resource(arg_scale_hh,  max_size=True)
        swf_rsrc = buffer_ops.create_buffer_resource(arg_scale_wf,  max_size=True)
        bias_rsrc= buffer_ops.create_buffer_resource(arg_bias,      max_size=True)
        ih_rsrc  = buffer_ops.create_buffer_resource(arg_ih_t,      max_size=True)

        # ── GEMM: 4 accumulator sets over K=H ────────────────────────────────
        wave_m_off = wave_m * (wave_reg_m * WMMA_M)  # = 0 (waves_m=1)
        base8      = klane * 8

        def _load_a(kt):
            """Load hh_fp8 A-fragments as v2i32."""
            vecs = []
            for rk in range_constexpr(reg_k):
                rv = []
                col = kt * tile_k + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row      = tile_m0 + wave_m_off + 16 * rm + lane16
                    byte_off = row * H + col
                    rv.append(buffer_ops.buffer_load(hh_rsrc, byte_off // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _load_b(kt, gate_n0_base):
            """Load W_fused B-tile as v2i32."""
            vecs = []
            for rk in range_constexpr(reg_k):
                rv = []
                k0 = kt * reg_k + rk
                for rn in range_constexpr(wave_reg_n_h):
                    n0       = gate_n0_base + wave_n * wave_reg_n_h + rn
                    byte_off = (n0 * B_STRIDE_N0 + k0 * B_STRIDE_K0
                                + klane * B_STRIDE_KLANE + lane16 * B_STRIDE_NLANE)
                    rv.append(buffer_ops.buffer_load(wf_rsrc, byte_off // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _compute4(ia, fa, ga, oa, a_vecs, bi, bf, bg, bo):
            ni = list(ia); nf = list(fa); ng = list(ga); no = list(oa)
            for rk in range_constexpr(reg_k):
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n_h):
                        idx = rm * wave_reg_n_h + rn
                        ni[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            ni[idx].type, a_vecs[rk][rm], bi[rk][rn], ni[idx]).result
                        nf[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            nf[idx].type, a_vecs[rk][rm], bf[rk][rn], nf[idx]).result
                        ng[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            ng[idx].type, a_vecs[rk][rm], bg[rk][rn], ng[idx]).result
                        no[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            no[idx].type, a_vecs[rk][rm], bo[rk][rn], no[idx]).result
            return ni, nf, ng, no

        # ── Gate N0 offsets ────────────────────────────────────────────────
        h_tile_n0 = bid_h * (tile_n_h // 16)
        gate_n0_i = h_tile_n0 + 0 * GATE_N0_STRIDE
        gate_n0_f = h_tile_n0 + 1 * GATE_N0_STRIDE
        gate_n0_g = h_tile_n0 + 2 * GATE_N0_STRIDE
        gate_n0_o = h_tile_n0 + 3 * GATE_N0_STRIDE

        # ── Software-pipelined K-loop ─────────────────────────────────────
        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_ac = wave_reg_m * wave_reg_n_h
        i_ac = [zero_acc for _ in range_constexpr(n_ac)]
        f_ac = [zero_acc for _ in range_constexpr(n_ac)]
        g_ac = [zero_acc for _ in range_constexpr(n_ac)]
        o_ac = [zero_acc for _ in range_constexpr(n_ac)]

        a_cur  = _load_a(0)
        bi_cur = _load_b(0, gate_n0_i)
        bf_cur = _load_b(0, gate_n0_f)
        bg_cur = _load_b(0, gate_n0_g)
        bo_cur = _load_b(0, gate_n0_o)

        def _flat(t):
            f = []
            for r in t: f.extend(r)
            return f

        def _unflat_a(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k):
                r = []
                for _ in range_constexpr(wave_reg_m):
                    r.append(f[i]); i += 1
                out.append(r)
            return out

        def _unflat_b(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k):
                r = []
                for _ in range_constexpr(wave_reg_n_h):
                    r.append(f[i]); i += 1
                out.append(r)
            return out

        n_a  = reg_k * wave_reg_m
        n_b  = reg_k * wave_reg_n_h
        n_ac4 = n_ac * 4   # packed i,f,g,o flat

        full_out = (num_k_tiles - 1) // k_unroll
        rem      = (num_k_tiles - 1) % k_unroll

        # Pack 4 accumulator lists into one flat list for the loop carry
        def _pack_accs(ia, fa, ga, oa):
            return list(ia) + list(fa) + list(ga) + list(oa)
        def _unpack_accs(flat):
            ia = flat[:n_ac]; fa = flat[n_ac:2*n_ac]
            ga = flat[2*n_ac:3*n_ac]; oa = flat[3*n_ac:]
            return ia, fa, ga, oa

        init = _flat(a_cur) + _pack_accs(i_ac, f_ac, g_ac, o_ac) + _flat(bi_cur) + _flat(bf_cur) + _flat(bg_cur) + _flat(bo_cur)

        if const_expr(full_out > 0):
            for iv, st in range(0, full_out * k_unroll, k_unroll, init=init):
                s_a  = _unflat_a(list(st[:n_a]))
                s_ac = list(st[n_a : n_a + n_ac4])
                rest = list(st[n_a + n_ac4:])
                s_bi = _unflat_b(rest[:n_b])
                s_bf = _unflat_b(rest[n_b:2*n_b])
                s_bg = _unflat_b(rest[2*n_b:3*n_b])
                s_bo = _unflat_b(rest[3*n_b:])
                s_ia, s_fa, s_ga, s_oa = _unpack_accs(s_ac)
                for j in range_constexpr(k_unroll):
                    nkt     = iv + j + 1
                    a_nxt   = _load_a(nkt)
                    bi_nxt  = _load_b(nkt, gate_n0_i)
                    bf_nxt  = _load_b(nkt, gate_n0_f)
                    bg_nxt  = _load_b(nkt, gate_n0_g)
                    bo_nxt  = _load_b(nkt, gate_n0_o)
                    s_ia, s_fa, s_ga, s_oa = _compute4(s_ia, s_fa, s_ga, s_oa, s_a, s_bi, s_bf, s_bg, s_bo)
                    s_a  = _unflat_a(_flat(a_nxt))
                    s_bi = _unflat_b(_flat(bi_nxt))
                    s_bf = _unflat_b(_flat(bf_nxt))
                    s_bg = _unflat_b(_flat(bg_nxt))
                    s_bo = _unflat_b(_flat(bo_nxt))
                res = yield _flat(s_a) + _pack_accs(s_ia, s_fa, s_ga, s_oa) + _flat(s_bi) + _flat(s_bf) + _flat(s_bg) + _flat(s_bo)
            s_a  = _unflat_a(list(res[:n_a]))
            s_ac = list(res[n_a : n_a + n_ac4])
            rest = list(res[n_a + n_ac4:])
            a_cur  = s_a
            bi_cur = _unflat_b(rest[:n_b]); bf_cur = _unflat_b(rest[n_b:2*n_b])
            bg_cur = _unflat_b(rest[2*n_b:3*n_b]); bo_cur = _unflat_b(rest[3*n_b:])
            i_ac, f_ac, g_ac, o_ac = _unpack_accs(s_ac)

        if const_expr(rem > 0):
            for j in range_constexpr(rem):
                nkt     = full_out * k_unroll + j + 1
                a_nxt   = _load_a(nkt)
                bi_nxt  = _load_b(nkt, gate_n0_i)
                bf_nxt  = _load_b(nkt, gate_n0_f)
                bg_nxt  = _load_b(nkt, gate_n0_g)
                bo_nxt  = _load_b(nkt, gate_n0_o)
                i_ac, f_ac, g_ac, o_ac = _compute4(i_ac, f_ac, g_ac, o_ac, a_cur, bi_cur, bf_cur, bg_cur, bo_cur)
                a_cur   = _unflat_a(_flat(a_nxt))
                bi_cur  = _unflat_b(_flat(bi_nxt)); bf_cur = _unflat_b(_flat(bf_nxt))
                bg_cur  = _unflat_b(_flat(bg_nxt)); bo_cur = _unflat_b(_flat(bo_nxt))

        i_ac, f_ac, g_ac, o_ac = _compute4(i_ac, f_ac, g_ac, o_ac, a_cur, bi_cur, bf_cur, bg_cur, bo_cur)

        # ── Epilogue ──────────────────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        # Padé [3/2] tanh: x*(27+x²)/(27+9x²) — replaces exp2+rcp, ~3% max error
        c27   = arith.constant(27.0, type=fx.T.f32())
        c9    = arith.constant(9.0,  type=fx.T.f32())
        # Fixed-scale fp8 output: h_new ∈ [-1,1] maps onto e4m3 [-448,448] exactly.
        c448     = arith.constant(FP8_MAX, type=fx.T.f32())
        zero_i32 = arith.constant(0, type=fx.T.i32())

        wave_nh0 = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

        def _bias(g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(
                bias_rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        def _swf(g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(
                swf_rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        bias_i = [_bias(0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_f = [_bias(1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_g = [_bias(2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_o = [_bias(3, rn) for rn in range_constexpr(wave_reg_n_h)]

        swf_i = [_swf(0, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_f = [_swf(1, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_g = [_swf(2, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_o = [_swf(3, rn) for rn in range_constexpr(wave_reg_n_h)]

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m_off + 16 * rm
            # Two vec4 f32 loads cover all 8 scale_hh values (max HW load = 128 bits).
            # Build a Python list of 8 pre-extracted MLIR values before the si loop
            # to avoid Python if/else inside range_constexpr.
            shh_base = tile_m0 + wmma_m_off + base8
            shh_lo   = buffer_ops.buffer_load(shh_rsrc, shh_base,     vec_width=4, dtype=fx.Float32)
            shh_hi   = buffer_ops.buffer_load(shh_rsrc, shh_base + 4, vec_width=4, dtype=fx.Float32)
            shh_vals = (
                [mlir_vector.extract(shh_lo, static_position=[i], dynamic_position=[]) for i in range(4)] +
                [mlir_vector.extract(shh_hi, static_position=[i], dynamic_position=[]) for i in range(4)]
            )

            for rn in range_constexpr(wave_reg_n_h):
                idx   = rm * wave_reg_n_h + rn
                h_col = wave_nh0 + 16 * rn + lane16

                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + si
                    s_hh  = ArithValue(shh_vals[si])

                    # arg_ih_t is [B, H, 4] layout: stride H*4 per row, 4 per h_col.
                    # Load all 4 gate values as one vec4_f16 (64-bit) instruction.
                    ih_base = g_row * (H * 4) + h_col * 4
                    ih_vec  = buffer_ops.buffer_load(
                        ih_rsrc, ih_base, vec_width=4, dtype=fx.Float16)
                    def _ih(gi):
                        return ArithValue(mlir_vector.extract(
                            ih_vec, static_position=[gi], dynamic_position=[])).extf(fx.T.f32())

                    # Descale: acc * scale_hh[row] * scale_wf[col] + bias + ih_t
                    i_raw = ArithValue(i_ac[idx][si]) * s_hh * swf_i[rn] + bias_i[rn] + _ih(0)
                    f_raw = ArithValue(f_ac[idx][si]) * s_hh * swf_f[rn] + bias_f[rn] + _ih(1)
                    g_raw = ArithValue(g_ac[idx][si]) * s_hh * swf_g[rn] + bias_g[rn] + _ih(2)
                    o_raw = ArithValue(o_ac[idx][si]) * s_hh * swf_o[rn] + bias_o[rn] + _ih(3)

                    def sighard(x):
                        return ArithValue(arith.minimumf(
                            (x * c0_2 + c0_5).maximumf(c0_0), c1_0))

                    i_a = sighard(i_raw)
                    f_a = sighard(f_raw)
                    g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                    o_a = sighard(o_raw)

                    c_off = g_row * H + h_col
                    c_val = ArithValue(buffer_ops.buffer_load(
                        c_rsrc, c_off, vec_width=1, dtype=fx.Float32))
                    c_new = f_a * c_val + i_a * g_a
                    buffer_ops.buffer_store(c_new, c_rsrc, c_off)

                    # Padé [3/2] tanh: x*(27+x²)/(27+9x²), ~3% max error, no exp2
                    x2     = c_new * c_new
                    numer  = c_new * (ArithValue(c27) + x2)
                    denom  = ArithValue(c27) + ArithValue(c9) * x2
                    tanh_c = ArithValue(arith.minimumf(
                        (numer * ArithValue(rocdl.rcp(fx.T.f32(), denom))).maximumf(cm1_0), c1_0))

                    h_new  = o_a * tanh_c
                    # Fixed-scale fp8: multiply by 448, convert to e4m3, store low byte.
                    h_scaled = h_new * ArithValue(c448)
                    packed   = rocdl.cvt_pk_fp8_f32(
                        fx.T.i32(), h_scaled, h_scaled, zero_i32, 0)
                    h_fp8_b  = arith.trunci(fx.T.i8(), packed)
                    buffer_ops.buffer_store(
                        h_fp8_b, h_rsrc, g_row * H + h_col, offset_is_bytes=True)

    # ── Host launcher ─────────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_unfactored_lstm(
        arg_h_fp8_out:  fx.Tensor,
        arg_c_inout:    fx.Tensor,
        arg_hh:         fx.Tensor,
        arg_scale_hh:   fx.Tensor,
        arg_w_fused:    fx.Tensor,
        arg_scale_wf:   fx.Tensor,
        arg_bias:       fx.Tensor,
        arg_ih_t:       fx.Tensor,
        stream:         fx.Stream,
        m:              fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_h
        launcher = kernel_unfactored_lstm(
            arg_h_fp8_out, arg_c_inout, arg_hh, arg_scale_hh,
            arg_w_fused, arg_scale_wf, arg_bias, arg_ih_t, dyn_grid_m,
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_fp8_unfactored_lstm


__all__ = [
    "compile_fp8_unfactored_lstm_gemm",
    "make_w_fused",
    "make_ih_t_interleaved",
    "preshuffle_b_fp8",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
