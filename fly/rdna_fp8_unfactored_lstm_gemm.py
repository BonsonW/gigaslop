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
    group_m: int = 8,   # block-schedule grouping for W_fused L2 reuse (~11% under graph replay)
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

                def sighard(x):
                    return ArithValue(arith.minimumf(
                        (x * c0_2 + c0_5).maximumf(c0_0), c1_0))

                # ── Phase A: issue ALL epilogue loads up front (memory-level parallelism).
                # ih_t [B,H,4]: one vec4_f16 per row covers all 4 gates; c f32 read.
                g_rows  = [tile_m0 + wmma_m_off + base8 + si for si in range_constexpr(8)]
                ih_vecs = [buffer_ops.buffer_load(
                    ih_rsrc, g_rows[si] * (H * 4) + h_col * 4, vec_width=4, dtype=fx.Float16)
                    for si in range_constexpr(8)]
                c_vals  = [ArithValue(buffer_ops.buffer_load(
                    c_rsrc, g_rows[si] * H + h_col, vec_width=1, dtype=fx.Float32))
                    for si in range_constexpr(8)]

                # ── Phase B: compute + store (loads above pipeline behind this work).
                for si in range_constexpr(8):
                    g_row = g_rows[si]
                    s_hh  = ArithValue(shh_vals[si])
                    iv    = ih_vecs[si]
                    def _ih(gi, _iv=iv):
                        return ArithValue(mlir_vector.extract(
                            _iv, static_position=[gi], dynamic_position=[])).extf(fx.T.f32())

                    # Descale: acc * scale_hh[row] * scale_wf[col] + bias + ih_t
                    i_raw = ArithValue(i_ac[idx][si]) * s_hh * swf_i[rn] + bias_i[rn] + _ih(0)
                    f_raw = ArithValue(f_ac[idx][si]) * s_hh * swf_f[rn] + bias_f[rn] + _ih(1)
                    g_raw = ArithValue(g_ac[idx][si]) * s_hh * swf_g[rn] + bias_g[rn] + _ih(2)
                    o_raw = ArithValue(o_ac[idx][si]) * s_hh * swf_o[rn] + bias_o[rn] + _ih(3)

                    i_a = sighard(i_raw)
                    f_a = sighard(f_raw)
                    g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                    o_a = sighard(o_raw)

                    c_off = g_row * H + h_col
                    c_new = f_a * c_vals[si] + i_a * g_a
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


def preshuffle_b_f16(B_kn):
    """Preshuffle B[K,N] f16 → [N0, K0, KLane=2, NLane=16, KPack=8] f16."""
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_r = B_kn.reshape(K // 16, 2, 8, N // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()


# =============================================================================
# Fused LSTM: hidden (hh @ W_fused, fp8) + input (x_down @ up_ih, f16) in one kernel.
# Folds the input projection's up-GEMM into the gate accumulation so the [B,4H] ih
# tensor is never materialized — only the tiny x_down[B,R] is read.
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_fused_lstm(
    *,
    B: int,
    H: int,
    R: int,
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k: int = 32,
    k_unroll: int = 2,
    tile_k2: int = 16,   # f16 WMMA K for the x_down@up pass
    group_m: int = 8,
    waves_m: int = 1,
    waves_n: int = 2,
    _abl: str = "",   # timing-only ablation: comma list of {phase2,cload,cstore,hstore}
):
    _ABL = set(s for s in _abl.split(",") if s)
    """Fused LSTM step:
        gates = hh_fp8 @ W_fused_fp8 (descaled) + x_down_f16 @ up_ih_f16 + bias_hh + bias_ih
        i,f,o = sighard; g = tanh_hard; c = f*c + i*g; h = o*tanh(c) -> fp8 (1/448)

    Replaces (factored GEMM -> ih[B,4H]) + (unfactored LSTM reading ih). x_down[B,R]
    (= x @ dn, precomputed) is read instead of ih[B,4H] (32x smaller). c updated in place.
    """
    FH = 4 * H
    assert B % tile_m == 0
    assert H % tile_n_h == 0 and H % tile_k == 0
    assert R % WMMA_N == 0 and R % tile_k2 == 0
    assert tile_n_h % WMMA_N == 0 and tile_k % WMMA_K == 0

    reg_m       = tile_m // WMMA_M
    reg_k       = tile_k // WMMA_K
    num_k_tiles = H // tile_k

    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m        = reg_m // waves_m
    wave_reg_n_h      = (tile_n_h // WMMA_N) // waves_n
    assert wave_reg_m >= 1 and wave_reg_n_h >= 1, "wave layout leaves a wave with no work"

    grid_m = B // tile_m
    grid_h = H // tile_n_h

    # W_fused fp8 strides [4H//16, H//16, 2, 16, 8]
    K0_H            = H // 16
    B_STRIDE_NLANE  = 8
    B_STRIDE_KLANE  = 16 * 8
    B_STRIDE_K0     = 2 * 16 * 8
    B_STRIDE_N0     = K0_H * B_STRIDE_K0
    GATE_N0_STRIDE  = H // 16

    # up_ih f16 strides [4H//16, R//16, 2, 16, 8] (f16 element units)
    K0_R            = R // 16
    B2_STRIDE_NLANE = 8
    B2_STRIDE_KLANE = 16 * 8
    B2_STRIDE_K0    = 2 * 16 * 8
    B2_STRIDE_N0    = K0_R * B2_STRIDE_K0
    reg_k2          = tile_k2 // WMMA_K
    num_k2_tiles    = R // tile_k2

    @flyc.kernel
    def kernel_fused_lstm(
        arg_h_fp8_out: fx.Tensor,   # [B, H]   fp8 (uint8), fixed scale 1/448
        arg_c_inout:   fx.Tensor,   # [B, H]   f32
        arg_hh:        fx.Tensor,   # [B, H]   fp8
        arg_scale_hh:  fx.Tensor,   # [B]      f32
        arg_w_fused:   fx.Tensor,   # preshuffled fp8 [H, 4H]
        arg_scale_wf:  fx.Tensor,   # [4H]     f32
        arg_bias_hh:   fx.Tensor,   # [4H]     f32
        arg_x_down:    fx.Tensor,   # [B, R]   f16  (= x @ dn, precomputed)
        arg_up_ih:     fx.Tensor,   # preshuffled f16 [R, 4H]
        arg_bias_ih:   fx.Tensor,   # [4H]     f32
        arg_grid_m:    fx.Int32,
    ):
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
            arg_grid_m, group_m_c)
        num_in_group = eff_gm * grid_h_c
        group_id     = pid_i32 // num_in_group
        pid_in_group = pid_i32 % num_in_group
        bid_m        = fx.arith.index_cast(fx.T.index(), group_id * eff_gm + pid_in_group % eff_gm)
        bid_h        = fx.arith.index_cast(fx.T.index(), pid_in_group // eff_gm)

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        tile_m0  = bid_m * tile_m
        tile_nh0 = bid_h * tile_n_h

        hh_rsrc  = buffer_ops.create_buffer_resource(arg_hh,        max_size=True)
        wf_rsrc  = buffer_ops.create_buffer_resource(arg_w_fused,   max_size=True)
        c_rsrc   = buffer_ops.create_buffer_resource(arg_c_inout,   max_size=True)
        h_rsrc   = buffer_ops.create_buffer_resource(arg_h_fp8_out, max_size=True)
        shh_rsrc = buffer_ops.create_buffer_resource(arg_scale_hh,  max_size=True)
        swf_rsrc = buffer_ops.create_buffer_resource(arg_scale_wf,  max_size=True)
        bhh_rsrc = buffer_ops.create_buffer_resource(arg_bias_hh,   max_size=True)
        xd_rsrc  = buffer_ops.create_buffer_resource(arg_x_down,    max_size=True)
        up_rsrc  = buffer_ops.create_buffer_resource(arg_up_ih,     max_size=True)
        bih_rsrc = buffer_ops.create_buffer_resource(arg_bias_ih,   max_size=True)

        wave_m_off = wave_m * (wave_reg_m * WMMA_M)
        base8      = klane * 8

        # ── Phase 1 loaders: hh @ W_fused (fp8) over K=H ─────────────────────
        def _load_a(kt):
            vecs = []
            for rk in range_constexpr(reg_k):
                rv = []
                col = kt * tile_k + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m_off + 16 * rm + lane16
                    rv.append(buffer_ops.buffer_load(hh_rsrc, (row * H + col) // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _load_b(kt, gate_n0_base):
            vecs = []
            for rk in range_constexpr(reg_k):
                rv = []
                k0 = kt * reg_k + rk
                for rn in range_constexpr(wave_reg_n_h):
                    n0 = gate_n0_base + wave_n * wave_reg_n_h + rn
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
                        ni[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(ni[idx].type, a_vecs[rk][rm], bi[rk][rn], ni[idx]).result
                        nf[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(nf[idx].type, a_vecs[rk][rm], bf[rk][rn], nf[idx]).result
                        ng[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(ng[idx].type, a_vecs[rk][rm], bg[rk][rn], ng[idx]).result
                        no[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(no[idx].type, a_vecs[rk][rm], bo[rk][rn], no[idx]).result
            return ni, nf, ng, no

        h_tile_n0 = bid_h * (tile_n_h // 16)
        gate_n0_i = h_tile_n0 + 0 * GATE_N0_STRIDE
        gate_n0_f = h_tile_n0 + 1 * GATE_N0_STRIDE
        gate_n0_g = h_tile_n0 + 2 * GATE_N0_STRIDE
        gate_n0_o = h_tile_n0 + 3 * GATE_N0_STRIDE

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_ac = wave_reg_m * wave_reg_n_h
        i_ac = [zero_acc for _ in range_constexpr(n_ac)]
        f_ac = [zero_acc for _ in range_constexpr(n_ac)]
        g_ac = [zero_acc for _ in range_constexpr(n_ac)]
        o_ac = [zero_acc for _ in range_constexpr(n_ac)]

        a_cur  = _load_a(0)
        bi_cur = _load_b(0, gate_n0_i); bf_cur = _load_b(0, gate_n0_f)
        bg_cur = _load_b(0, gate_n0_g); bo_cur = _load_b(0, gate_n0_o)

        def _flat(t):
            f = []
            for r in t: f.extend(r)
            return f
        def _unflat_a(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k):
                r = []
                for _ in range_constexpr(wave_reg_m): r.append(f[i]); i += 1
                out.append(r)
            return out
        def _unflat_b(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k):
                r = []
                for _ in range_constexpr(wave_reg_n_h): r.append(f[i]); i += 1
                out.append(r)
            return out

        n_a  = reg_k * wave_reg_m
        n_b  = reg_k * wave_reg_n_h
        n_ac4 = n_ac * 4
        full_out = (num_k_tiles - 1) // k_unroll
        rem      = (num_k_tiles - 1) % k_unroll

        def _pack_accs(ia, fa, ga, oa):
            return list(ia) + list(fa) + list(ga) + list(oa)
        def _unpack_accs(flat):
            return flat[:n_ac], flat[n_ac:2*n_ac], flat[2*n_ac:3*n_ac], flat[3*n_ac:]

        init = _flat(a_cur) + _pack_accs(i_ac, f_ac, g_ac, o_ac) + _flat(bi_cur) + _flat(bf_cur) + _flat(bg_cur) + _flat(bo_cur)

        if const_expr(full_out > 0):
            for iv, st in range(0, full_out * k_unroll, k_unroll, init=init):
                s_a  = _unflat_a(list(st[:n_a]))
                s_ac = list(st[n_a : n_a + n_ac4])
                rest = list(st[n_a + n_ac4:])
                s_bi = _unflat_b(rest[:n_b]); s_bf = _unflat_b(rest[n_b:2*n_b])
                s_bg = _unflat_b(rest[2*n_b:3*n_b]); s_bo = _unflat_b(rest[3*n_b:])
                s_ia, s_fa, s_ga, s_oa = _unpack_accs(s_ac)
                for j in range_constexpr(k_unroll):
                    nkt = iv + j + 1
                    a_nxt  = _load_a(nkt)
                    bi_nxt = _load_b(nkt, gate_n0_i); bf_nxt = _load_b(nkt, gate_n0_f)
                    bg_nxt = _load_b(nkt, gate_n0_g); bo_nxt = _load_b(nkt, gate_n0_o)
                    s_ia, s_fa, s_ga, s_oa = _compute4(s_ia, s_fa, s_ga, s_oa, s_a, s_bi, s_bf, s_bg, s_bo)
                    s_a  = _unflat_a(_flat(a_nxt))
                    s_bi = _unflat_b(_flat(bi_nxt)); s_bf = _unflat_b(_flat(bf_nxt))
                    s_bg = _unflat_b(_flat(bg_nxt)); s_bo = _unflat_b(_flat(bo_nxt))
                res = yield _flat(s_a) + _pack_accs(s_ia, s_fa, s_ga, s_oa) + _flat(s_bi) + _flat(s_bf) + _flat(s_bg) + _flat(s_bo)
            a_cur  = _unflat_a(list(res[:n_a]))
            rest   = list(res[n_a + n_ac4:])
            bi_cur = _unflat_b(rest[:n_b]); bf_cur = _unflat_b(rest[n_b:2*n_b])
            bg_cur = _unflat_b(rest[2*n_b:3*n_b]); bo_cur = _unflat_b(rest[3*n_b:])
            i_ac, f_ac, g_ac, o_ac = _unpack_accs(list(res[n_a : n_a + n_ac4]))

        if const_expr(rem > 0):
            for j in range_constexpr(rem):
                nkt = full_out * k_unroll + j + 1
                a_nxt  = _load_a(nkt)
                bi_nxt = _load_b(nkt, gate_n0_i); bf_nxt = _load_b(nkt, gate_n0_f)
                bg_nxt = _load_b(nkt, gate_n0_g); bo_nxt = _load_b(nkt, gate_n0_o)
                i_ac, f_ac, g_ac, o_ac = _compute4(i_ac, f_ac, g_ac, o_ac, a_cur, bi_cur, bf_cur, bg_cur, bo_cur)
                a_cur  = _unflat_a(_flat(a_nxt))
                bi_cur = _unflat_b(_flat(bi_nxt)); bf_cur = _unflat_b(_flat(bf_nxt))
                bg_cur = _unflat_b(_flat(bg_nxt)); bo_cur = _unflat_b(_flat(bo_nxt))

        i_ac, f_ac, g_ac, o_ac = _compute4(i_ac, f_ac, g_ac, o_ac, a_cur, bi_cur, bf_cur, bg_cur, bo_cur)

        # ── Phase 2: x_down @ up_ih (f16) over K=R → separate xu accumulators ──
        v8f16_ty = ir.VectorType.get([8], ir.F16Type.get())

        def _load_a2(kt):
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row = tile_m0 + wave_m_off + 16 * rm + lane16
                k_elem = kt * WMMA_K + klane * 8
                frags.append(buffer_ops.buffer_load(xd_rsrc, row * R + k_elem,
                                                    vec_width=8, dtype=fx.Float16))
            return frags

        def _load_b2(kt, gate_n0_base):
            vecs = []
            for rn in range_constexpr(wave_reg_n_h):
                n0 = gate_n0_base + wave_n * wave_reg_n_h + rn
                f16_off = (n0 * B2_STRIDE_N0 + kt * B2_STRIDE_K0
                           + klane * B2_STRIDE_KLANE + lane16 * B2_STRIDE_NLANE)
                vecs.append(buffer_ops.buffer_load(up_rsrc, f16_off, vec_width=8, dtype=fx.Float16))
            return vecs

        def _compute2(ia, fa, ga, oa, a_v, bi, bf, bg, bo):
            ni = list(ia); nf = list(fa); ng = list(ga); no = list(oa)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n_h):
                    idx = rm * wave_reg_n_h + rn
                    ni[idx] = rocdl.wmma_f32_16x16x16_f16(ni[idx].type, a_v[rm], bi[rn], ni[idx]).result
                    nf[idx] = rocdl.wmma_f32_16x16x16_f16(nf[idx].type, a_v[rm], bf[rn], nf[idx]).result
                    ng[idx] = rocdl.wmma_f32_16x16x16_f16(ng[idx].type, a_v[rm], bg[rn], ng[idx]).result
                    no[idx] = rocdl.wmma_f32_16x16x16_f16(no[idx].type, a_v[rm], bo[rn], no[idx]).result
            return ni, nf, ng, no

        # Descale-in-place: fold Phase 2 into the Phase-1 accumulators instead of
        # using a second set xi/xf/xg/xo (which maxed VGPR at 256 + spills). Descale
        # the raw fp8 hidden sums by scale_hh*scale_wf, then let the f16 input WMMA
        # accumulate on top — 4 accumulator sets total instead of 8.
        v8f32_ty = ir.VectorType.get([8], ir.F32Type.get())
        wave_nh0 = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

        def _ld(rsrc, g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        swf_i = [_ld(swf_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_f = [_ld(swf_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_g = [_ld(swf_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_o = [_ld(swf_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m_off + 16 * rm
            shh_base = tile_m0 + wmma_m_off + base8
            shh_lo = buffer_ops.buffer_load(shh_rsrc, shh_base,     vec_width=4, dtype=fx.Float32)
            shh_hi = buffer_ops.buffer_load(shh_rsrc, shh_base + 4, vec_width=4, dtype=fx.Float32)
            shh_vals = ([mlir_vector.extract(shh_lo, static_position=[i], dynamic_position=[]) for i in range(4)] +
                        [mlir_vector.extract(shh_hi, static_position=[i], dynamic_position=[]) for i in range(4)])
            for rn in range_constexpr(wave_reg_n_h):
                idx = rm * wave_reg_n_h + rn
                def _descale(acc, swf):
                    lanes = [ArithValue(acc[idx][si]) * ArithValue(shh_vals[si]) * swf[rn]
                             for si in range_constexpr(8)]
                    return mlir_vector.from_elements(v8f32_ty, lanes)
                i_ac[idx] = _descale(i_ac, swf_i)
                f_ac[idx] = _descale(f_ac, swf_f)
                g_ac[idx] = _descale(g_ac, swf_g)
                o_ac[idx] = _descale(o_ac, swf_o)

        if const_expr("phase2" not in _ABL):
            for kt2 in range_constexpr(num_k2_tiles):
                a2 = _load_a2(kt2)
                bi2 = _load_b2(kt2, gate_n0_i); bf2 = _load_b2(kt2, gate_n0_f)
                bg2 = _load_b2(kt2, gate_n0_g); bo2 = _load_b2(kt2, gate_n0_o)
                i_ac, f_ac, g_ac, o_ac = _compute2(i_ac, f_ac, g_ac, o_ac, a2, bi2, bf2, bg2, bo2)

        # ── Epilogue ──────────────────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        c27   = arith.constant(27.0, type=fx.T.f32())
        c9    = arith.constant(9.0,  type=fx.T.f32())
        c448  = arith.constant(FP8_MAX, type=fx.T.f32())
        zero_i32 = arith.constant(0, type=fx.T.i32())

        bias_i = [_ld(bhh_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_f = [_ld(bhh_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_g = [_ld(bhh_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_o = [_ld(bhh_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_i  = [_ld(bih_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_f  = [_ld(bih_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_g  = [_ld(bih_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_o  = [_ld(bih_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m_off + 16 * rm

            for rn in range_constexpr(wave_reg_n_h):
                idx   = rm * wave_reg_n_h + rn
                h_col = wave_nh0 + 16 * rn + lane16

                def sighard(x):
                    return ArithValue(arith.minimumf((x * c0_2 + c0_5).maximumf(c0_0), c1_0))

                g_rows = [tile_m0 + wmma_m_off + base8 + si for si in range_constexpr(8)]
                if const_expr("cload" in _ABL):
                    c_vals = [ArithValue(c0_0) for si in range_constexpr(8)]
                else:
                    c_vals = [ArithValue(buffer_ops.buffer_load(
                        c_rsrc, g_rows[si] * H + h_col, vec_width=1, dtype=fx.Float32))
                        for si in range_constexpr(8)]

                for si in range_constexpr(8):
                    g_row = g_rows[si]
                    # accumulators already hold descaled hidden + input(x_down@up);
                    # just add the biases.
                    i_raw = ArithValue(i_ac[idx][si]) + bias_i[rn] + bih_i[rn]
                    f_raw = ArithValue(f_ac[idx][si]) + bias_f[rn] + bih_f[rn]
                    g_raw = ArithValue(g_ac[idx][si]) + bias_g[rn] + bih_g[rn]
                    o_raw = ArithValue(o_ac[idx][si]) + bias_o[rn] + bih_o[rn]

                    i_a = sighard(i_raw)
                    f_a = sighard(f_raw)
                    g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                    o_a = sighard(o_raw)

                    c_off = g_row * H + h_col
                    c_new = f_a * c_vals[si] + i_a * g_a
                    if const_expr("cstore" not in _ABL):
                        buffer_ops.buffer_store(c_new, c_rsrc, c_off)

                    x2     = c_new * c_new
                    numer  = c_new * (ArithValue(c27) + x2)
                    denom  = ArithValue(c27) + ArithValue(c9) * x2
                    tanh_c = ArithValue(arith.minimumf(
                        (numer * ArithValue(rocdl.rcp(fx.T.f32(), denom))).maximumf(cm1_0), c1_0))

                    h_new    = o_a * tanh_c
                    h_scaled = h_new * ArithValue(c448)
                    packed   = rocdl.cvt_pk_fp8_f32(fx.T.i32(), h_scaled, h_scaled, zero_i32, 0)
                    h_fp8_b  = arith.trunci(fx.T.i8(), packed)
                    if const_expr("hstore" not in _ABL):
                        buffer_ops.buffer_store(h_fp8_b, h_rsrc, g_row * H + h_col, offset_is_bytes=True)

    @flyc.jit
    def launch_fp8_fused_lstm(
        arg_h_fp8_out: fx.Tensor,
        arg_c_inout:   fx.Tensor,
        arg_hh:        fx.Tensor,
        arg_scale_hh:  fx.Tensor,
        arg_w_fused:   fx.Tensor,
        arg_scale_wf:  fx.Tensor,
        arg_bias_hh:   fx.Tensor,
        arg_x_down:    fx.Tensor,
        arg_up_ih:     fx.Tensor,
        arg_bias_ih:   fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_h
        launcher = kernel_fused_lstm(
            arg_h_fp8_out, arg_c_inout, arg_hh, arg_scale_hh,
            arg_w_fused, arg_scale_wf, arg_bias_hh,
            arg_x_down, arg_up_ih, arg_bias_ih, dyn_grid_m,
        )
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_fp8_fused_lstm


# =============================================================================
# Factored LSTM: BOTH hidden and input projections kept low-rank (no K=H GEMM).
#   gates = hh_down @ up_hh (f16, K=K_hh) + x_down @ up_ih (f16, K=R) + bias_hh + bias_ih
# hh_down = h @ dn_hh (recurrent, computed per-step by compile_fp8_down_proj) replaces
# the expensive K=H=1024 hh@W_fused Phase 1 of compile_fp8_fused_lstm. Two cheap K=128
# f16 up-projections, no fp8 descale.
# =============================================================================
def compile_fp8_factored_lstm(
    *,
    B: int,
    H: int,
    K_hh: int = 128,     # hidden rank (h @ dn_hh -> hh_down[B, K_hh])
    R: int = 128,        # input rank  (x @ dn_ih -> x_down[B, R])
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k2: int = 16,   # f16 WMMA K-tile for both up-projections
    group_m: int = 8,
    waves_m: int = 1,
    waves_n: int = 2,
    _abl: str = "",      # timing-only ablation: comma list of {hh,ih,cload,cstore,hstore,tanh}
):
    """Factored LSTM step (no K=H GEMM):
        gates = hh_down @ up_hh (f16, K=K_hh) + x_down @ up_ih (f16, K=R) + bias_hh + bias_ih
        i,f,o = sighard; g = tanh_hard; c = f*c + i*g; h = o*tanh(c) -> fp8 (1/448)

    hh_down (= h @ dn_hh, recurrent) is produced per-step by compile_fp8_down_proj, and
    x_down (= x @ dn_ih) is precomputed. Both up-projections are f16, K=128 -> cheap.
    """
    _ABL = set(s for s in _abl.split(",") if s)
    FH = 4 * H
    assert B % tile_m == 0
    assert H % tile_n_h == 0
    assert K_hh % WMMA_N == 0 and K_hh % tile_k2 == 0
    assert R % WMMA_N == 0 and R % tile_k2 == 0
    assert tile_n_h % WMMA_N == 0 and tile_k2 == WMMA_K

    reg_m             = tile_m // WMMA_M
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m        = reg_m // waves_m
    wave_reg_n_h      = (tile_n_h // WMMA_N) // waves_n
    assert wave_reg_m >= 1 and wave_reg_n_h >= 1, "wave layout leaves a wave with no work"

    grid_m = B // tile_m
    grid_h = H // tile_n_h

    # Output is 4H gates in both up-projections -> gates stride by H//16 in N0.
    GATE_N0_STRIDE = H // 16
    # Shared f16 B-tile strides (K-independent); N0 stride differs by rank.
    B_STRIDE_NLANE = 8
    B_STRIDE_KLANE = 16 * 8
    B_STRIDE_K0    = 2 * 16 * 8
    BHH_STRIDE_N0  = (K_hh // 16) * B_STRIDE_K0   # up_hh [4H//16, K_hh//16, 2, 16, 8]
    BIH_STRIDE_N0  = (R // 16) * B_STRIDE_K0      # up_ih [4H//16, R//16,    2, 16, 8]
    num_k_tiles_hh = K_hh // tile_k2
    num_k_tiles_ih = R // tile_k2

    @flyc.kernel
    def kernel_factored_lstm(
        arg_h_fp8_out: fx.Tensor,   # [B, H]   fp8 (uint8), fixed scale 1/448
        arg_c_inout:   fx.Tensor,   # [B, H]   f32
        arg_hh_down:   fx.Tensor,   # [B, K_hh] f16  (= h @ dn_hh, recurrent)
        arg_up_hh:     fx.Tensor,   # preshuffled f16 [4H//16, K_hh//16, 2, 16, 8]
        arg_bias_hh:   fx.Tensor,   # [4H]     f32
        arg_x_down:    fx.Tensor,   # [B, R]   f16  (= x @ dn_ih, precomputed)
        arg_up_ih:     fx.Tensor,   # preshuffled f16 [4H//16, R//16, 2, 16, 8]
        arg_bias_ih:   fx.Tensor,   # [4H]     f32
        arg_grid_m:    fx.Int32,
    ):
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
            arg_grid_m, group_m_c)
        num_in_group = eff_gm * grid_h_c
        group_id     = pid_i32 // num_in_group
        pid_in_group = pid_i32 % num_in_group
        bid_m        = fx.arith.index_cast(fx.T.index(), group_id * eff_gm + pid_in_group % eff_gm)
        bid_h        = fx.arith.index_cast(fx.T.index(), pid_in_group // eff_gm)

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        tile_m0  = bid_m * tile_m
        tile_nh0 = bid_h * tile_n_h

        c_rsrc   = buffer_ops.create_buffer_resource(arg_c_inout,   max_size=True)
        h_rsrc   = buffer_ops.create_buffer_resource(arg_h_fp8_out, max_size=True)
        hd_rsrc  = buffer_ops.create_buffer_resource(arg_hh_down,   max_size=True)
        uhh_rsrc = buffer_ops.create_buffer_resource(arg_up_hh,     max_size=True)
        bhh_rsrc = buffer_ops.create_buffer_resource(arg_bias_hh,   max_size=True)
        xd_rsrc  = buffer_ops.create_buffer_resource(arg_x_down,    max_size=True)
        up_rsrc  = buffer_ops.create_buffer_resource(arg_up_ih,     max_size=True)
        bih_rsrc = buffer_ops.create_buffer_resource(arg_bias_ih,   max_size=True)

        wave_m_off = wave_m * (wave_reg_m * WMMA_M)
        base8      = klane * 8

        h_tile_n0 = bid_h * (tile_n_h // 16)
        gate_n0_i = h_tile_n0 + 0 * GATE_N0_STRIDE
        gate_n0_f = h_tile_n0 + 1 * GATE_N0_STRIDE
        gate_n0_g = h_tile_n0 + 2 * GATE_N0_STRIDE
        gate_n0_o = h_tile_n0 + 3 * GATE_N0_STRIDE

        # ── f16 up-projection loaders (shared by hidden & input paths) ──────────
        def _load_a_f16(kt, rsrc, Kdim):
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row = tile_m0 + wave_m_off + 16 * rm + lane16
                k_elem = kt * WMMA_K + klane * 8
                frags.append(buffer_ops.buffer_load(rsrc, row * Kdim + k_elem,
                                                    vec_width=8, dtype=fx.Float16))
            return frags

        def _load_b_f16(kt, gate_n0_base, rsrc, stride_n0):
            vecs = []
            for rn in range_constexpr(wave_reg_n_h):
                n0 = gate_n0_base + wave_n * wave_reg_n_h + rn
                f16_off = (n0 * stride_n0 + kt * B_STRIDE_K0
                           + klane * B_STRIDE_KLANE + lane16 * B_STRIDE_NLANE)
                vecs.append(buffer_ops.buffer_load(rsrc, f16_off, vec_width=8, dtype=fx.Float16))
            return vecs

        def _compute2(ia, fa, ga, oa, a_v, bi, bf, bg, bo):
            ni = list(ia); nf = list(fa); ng = list(ga); no = list(oa)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n_h):
                    idx = rm * wave_reg_n_h + rn
                    ni[idx] = rocdl.wmma_f32_16x16x16_f16(ni[idx].type, a_v[rm], bi[rn], ni[idx]).result
                    nf[idx] = rocdl.wmma_f32_16x16x16_f16(nf[idx].type, a_v[rm], bf[rn], nf[idx]).result
                    ng[idx] = rocdl.wmma_f32_16x16x16_f16(ng[idx].type, a_v[rm], bg[rn], ng[idx]).result
                    no[idx] = rocdl.wmma_f32_16x16x16_f16(no[idx].type, a_v[rm], bo[rn], no[idx]).result
            return ni, nf, ng, no

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_ac = wave_reg_m * wave_reg_n_h
        i_ac = [zero_acc for _ in range_constexpr(n_ac)]
        f_ac = [zero_acc for _ in range_constexpr(n_ac)]
        g_ac = [zero_acc for _ in range_constexpr(n_ac)]
        o_ac = [zero_acc for _ in range_constexpr(n_ac)]

        # ── Hidden up-projection: hh_down @ up_hh (f16) over K=K_hh ─────────────
        if const_expr("hh" not in _ABL):
            for kt in range_constexpr(num_k_tiles_hh):
                a  = _load_a_f16(kt, hd_rsrc, K_hh)
                bi = _load_b_f16(kt, gate_n0_i, uhh_rsrc, BHH_STRIDE_N0)
                bf = _load_b_f16(kt, gate_n0_f, uhh_rsrc, BHH_STRIDE_N0)
                bg = _load_b_f16(kt, gate_n0_g, uhh_rsrc, BHH_STRIDE_N0)
                bo = _load_b_f16(kt, gate_n0_o, uhh_rsrc, BHH_STRIDE_N0)
                i_ac, f_ac, g_ac, o_ac = _compute2(i_ac, f_ac, g_ac, o_ac, a, bi, bf, bg, bo)

        # ── Input up-projection: x_down @ up_ih (f16) over K=R ──────────────────
        if const_expr("ih" not in _ABL):
            for kt in range_constexpr(num_k_tiles_ih):
                a  = _load_a_f16(kt, xd_rsrc, R)
                bi = _load_b_f16(kt, gate_n0_i, up_rsrc, BIH_STRIDE_N0)
                bf = _load_b_f16(kt, gate_n0_f, up_rsrc, BIH_STRIDE_N0)
                bg = _load_b_f16(kt, gate_n0_g, up_rsrc, BIH_STRIDE_N0)
                bo = _load_b_f16(kt, gate_n0_o, up_rsrc, BIH_STRIDE_N0)
                i_ac, f_ac, g_ac, o_ac = _compute2(i_ac, f_ac, g_ac, o_ac, a, bi, bf, bg, bo)

        # ── Epilogue ────────────────────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        c27   = arith.constant(27.0, type=fx.T.f32())
        c9    = arith.constant(9.0,  type=fx.T.f32())
        c448  = arith.constant(FP8_MAX, type=fx.T.f32())
        zero_i32 = arith.constant(0, type=fx.T.i32())

        wave_nh0 = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

        def _ld(rsrc, g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        bias_i = [_ld(bhh_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_f = [_ld(bhh_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_g = [_ld(bhh_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_o = [_ld(bhh_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_i  = [_ld(bih_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_f  = [_ld(bih_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_g  = [_ld(bih_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bih_o  = [_ld(bih_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m_off + 16 * rm

            for rn in range_constexpr(wave_reg_n_h):
                idx   = rm * wave_reg_n_h + rn
                h_col = wave_nh0 + 16 * rn + lane16

                def sighard(x):
                    return ArithValue(arith.minimumf((x * c0_2 + c0_5).maximumf(c0_0), c1_0))

                g_rows = [tile_m0 + wmma_m_off + base8 + si for si in range_constexpr(8)]
                if const_expr("cload" in _ABL):
                    c_vals = [ArithValue(c0_0) for si in range_constexpr(8)]
                else:
                    c_vals = [ArithValue(buffer_ops.buffer_load(
                        c_rsrc, g_rows[si] * H + h_col, vec_width=1, dtype=fx.Float32))
                        for si in range_constexpr(8)]

                for si in range_constexpr(8):
                    g_row = g_rows[si]
                    # accumulators already hold hidden(hh_down@up_hh) + input(x_down@up_ih);
                    # just add the biases.
                    i_raw = ArithValue(i_ac[idx][si]) + bias_i[rn] + bih_i[rn]
                    f_raw = ArithValue(f_ac[idx][si]) + bias_f[rn] + bih_f[rn]
                    g_raw = ArithValue(g_ac[idx][si]) + bias_g[rn] + bih_g[rn]
                    o_raw = ArithValue(o_ac[idx][si]) + bias_o[rn] + bih_o[rn]

                    i_a = sighard(i_raw)
                    f_a = sighard(f_raw)
                    g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                    o_a = sighard(o_raw)

                    c_off = g_row * H + h_col
                    c_new = f_a * c_vals[si] + i_a * g_a
                    if const_expr("cstore" not in _ABL):
                        buffer_ops.buffer_store(c_new, c_rsrc, c_off)

                    if const_expr("tanh" in _ABL):
                        tanh_c = c_new
                    else:
                        x2     = c_new * c_new
                        numer  = c_new * (ArithValue(c27) + x2)
                        denom  = ArithValue(c27) + ArithValue(c9) * x2
                        tanh_c = ArithValue(arith.minimumf(
                            (numer * ArithValue(rocdl.rcp(fx.T.f32(), denom))).maximumf(cm1_0), c1_0))

                    h_new    = o_a * tanh_c
                    h_scaled = h_new * ArithValue(c448)
                    packed   = rocdl.cvt_pk_fp8_f32(fx.T.i32(), h_scaled, h_scaled, zero_i32, 0)
                    h_fp8_b  = arith.trunci(fx.T.i8(), packed)
                    if const_expr("hstore" not in _ABL):
                        buffer_ops.buffer_store(h_fp8_b, h_rsrc, g_row * H + h_col, offset_is_bytes=True)

    @flyc.jit
    def launch_fp8_factored_lstm(
        arg_h_fp8_out: fx.Tensor,
        arg_c_inout:   fx.Tensor,
        arg_hh_down:   fx.Tensor,
        arg_up_hh:     fx.Tensor,
        arg_bias_hh:   fx.Tensor,
        arg_x_down:    fx.Tensor,
        arg_up_ih:     fx.Tensor,
        arg_bias_ih:   fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_h
        launcher = kernel_factored_lstm(
            arg_h_fp8_out, arg_c_inout,
            arg_hh_down, arg_up_hh, arg_bias_hh,
            arg_x_down, arg_up_ih, arg_bias_ih, dyn_grid_m,
        )
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_fp8_factored_lstm


# =============================================================================
# Option A: single-kernel factored LSTM — fold the recurrent down-projection
# hh_down = h @ dn_hh INTO the LSTM kernel (compute once per block, stage in LDS),
# then both up-projections + epilogue in the same launch. One launch floor instead
# of two (down_proj + factored_lstm). grid = grid_m * h_split (occupancy knob).
# =============================================================================
def compile_fp8_factored_lstm_fused(
    *,
    B: int,
    H: int,
    K_hh: int = 128,
    R: int = 128,
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k1: int = 32,   # Phase-A (h@dn_hh) fp8 K-tile
    k_unroll: int = 2,   # (reserved; Phase A uses a simple K-loop for now)
    tile_k2: int = 16,   # up-projection f16 WMMA K-tile
    h_split: int = 4,    # split H output across this many blocks per M-tile (occupancy)
    waves_m: int = 1,
    waves_n: int = 2,
    _abl: str = "",      # timing-only ablation: {phaseA,hh,ih,cstore,hstore,tanh}
):
    """Single-kernel factored LSTM (Option A):
        hh_down = h_prev_fp8 @ dn_hh_fp8 (K=H, computed in-kernel, staged in LDS)
        gates   = hh_down @ up_hh (K=K_hh) + x_down @ up_ih (K=R) + bias_hh + bias_ih
        i,f,o = sighard; g = tanh_hard; c = f*c + i*g; h = o*tanh(c) -> fp8 (1/448)
    h_prev is fp8 with the fixed 1/448 output scale. One launch instead of
    down_proj + factored_lstm (removes one launch/grid floor + the hh_down round-trip).
    """
    _ABL = set(s for s in _abl.split(",") if s)
    FH = 4 * H
    assert B % tile_m == 0
    assert H % tile_n_h == 0 and H % tile_k1 == 0
    assert H % h_split == 0 and (H // h_split) % tile_n_h == 0
    assert K_hh % WMMA_N == 0 and K_hh % tile_k2 == 0
    assert R % WMMA_N == 0 and R % tile_k2 == 0
    assert tile_n_h % WMMA_N == 0 and tile_k1 % WMMA_K == 0 and tile_k2 == WMMA_K

    # Phase A dims (h_prev[tile_m,H] @ dn_hh[H,K_hh] -> hh_down[tile_m,K_hh])
    reg_m        = tile_m // WMMA_M
    reg_n_y      = K_hh // WMMA_N
    reg_k1       = tile_k1 // WMMA_K
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m   = reg_m // waves_m
    wave_reg_n_y = reg_n_y // waves_n
    wave_reg_n_h = (tile_n_h // WMMA_N) // waves_n
    assert wave_reg_m >= 1 and wave_reg_n_y >= 1 and wave_reg_n_h >= 1
    num_k1_tiles = H // tile_k1

    grid_m       = B // tile_m
    nt_per_block = (H // h_split) // tile_n_h   # h-tiles each block walks

    # Phase A B-strides: dn_hh fp8 preshuffled [K_hh//16, H//16, 2, 16, 8]
    K0_H            = H // 16
    B1_STRIDE_NLANE = 8
    B1_STRIDE_KLANE = 16 * 8
    B1_STRIDE_K0    = 2 * 16 * 8
    B1_STRIDE_N0    = K0_H * B1_STRIDE_K0

    # Up-projection (Phase B/2) shared f16 strides; N0 stride differs by rank.
    GATE_N0_STRIDE = H // 16
    B_STRIDE_NLANE = 8
    B_STRIDE_KLANE = 16 * 8
    B_STRIDE_K0    = 2 * 16 * 8
    BHH_STRIDE_N0  = (K_hh // 16) * B_STRIDE_K0
    BIH_STRIDE_N0  = (R // 16) * B_STRIDE_K0
    num_k_tiles_hh = K_hh // tile_k2
    num_k_tiles_ih = R // tile_k2

    # LDS: hh_down f16 [tile_m, K_hh]
    lds_alloc    = SmemAllocator(None, global_sym_name="smem_factored_lstm_fused")
    hh_byte_off  = lds_alloc._align(lds_alloc.ptr, 32)
    hh_size      = tile_m * K_hh * 2
    lds_alloc.ptr = hh_byte_off + hh_size

    @flyc.kernel
    def kernel_factored_lstm_fused(
        arg_h_fp8_out:  fx.Tensor,   # [B, H]    fp8 (uint8), fixed scale 1/448
        arg_c_inout:    fx.Tensor,   # [B, H]    f32
        arg_h_prev_fp8: fx.Tensor,   # [B, H]    fp8 (uint8), scale 1/448 (recurrent input)
        arg_scale_dn:   fx.Tensor,   # [1]       f32  scalar dn_hh scale
        arg_dn_hh:      fx.Tensor,   # preshuffled fp8 [K_hh//16, H//16, 2, 16, 8]
        arg_up_hh:      fx.Tensor,   # preshuffled f16 [4H//16, K_hh//16, 2, 16, 8]
        arg_bias_hh:    fx.Tensor,   # [4H]      f32
        arg_x_down:     fx.Tensor,   # [B, R]    f16  (= x @ dn_ih, precomputed)
        arg_up_ih:      fx.Tensor,   # preshuffled f16 [4H//16, R//16, 2, 16, 8]
        arg_bias_ih:    fx.Tensor,   # [4H]      f32
    ):
        tid     = gpu.thread_id("x")
        pid     = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        bid_m   = pid // h_split
        bid_hs  = pid - bid_m * h_split        # pid % h_split (index)
        wave_m  = wave_id // waves_n
        wave_n  = wave_id % waves_n
        tile_m0 = bid_m * tile_m

        hp_rsrc  = buffer_ops.create_buffer_resource(arg_h_prev_fp8, max_size=True)
        dn_rsrc  = buffer_ops.create_buffer_resource(arg_dn_hh,      max_size=True)
        sdn_rsrc = buffer_ops.create_buffer_resource(arg_scale_dn,   max_size=True)
        c_rsrc   = buffer_ops.create_buffer_resource(arg_c_inout,    max_size=True)
        h_rsrc   = buffer_ops.create_buffer_resource(arg_h_fp8_out,  max_size=True)
        uhh_rsrc = buffer_ops.create_buffer_resource(arg_up_hh,      max_size=True)
        bhh_rsrc = buffer_ops.create_buffer_resource(arg_bias_hh,    max_size=True)
        xd_rsrc  = buffer_ops.create_buffer_resource(arg_x_down,     max_size=True)
        up_rsrc  = buffer_ops.create_buffer_resource(arg_up_ih,      max_size=True)
        bih_rsrc = buffer_ops.create_buffer_resource(arg_bias_ih,    max_size=True)

        base_ptr = lds_alloc.get_base()
        s_hh     = SmemPtr(base_ptr, hh_byte_off, fx.T.f16(), shape=(tile_m * K_hh,))
        s_hh.get()

        wave_m_off = wave_m * (wave_reg_m * WMMA_M)
        base8      = klane * 8
        zero_acc   = fx.full(8, 0.0, fx.Float32)

        # ── Phase A: hh_down = h_prev_fp8 @ dn_hh_fp8 (K=H) ─────────────────────
        def _load_a1(kt):
            vecs = []
            for rk in range_constexpr(reg_k1):
                rv = []
                col = kt * tile_k1 + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m_off + 16 * rm + lane16
                    rv.append(buffer_ops.buffer_load(hp_rsrc, (row * H + col) // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _load_b1(kt):
            vecs = []
            n0_base = wave_n * wave_reg_n_y
            for rk in range_constexpr(reg_k1):
                rv = []
                k0 = kt * reg_k1 + rk
                for rn in range_constexpr(wave_reg_n_y):
                    n0 = n0_base + rn
                    byte_off = (n0 * B1_STRIDE_N0 + k0 * B1_STRIDE_K0
                                + klane * B1_STRIDE_KLANE + lane16 * B1_STRIDE_NLANE)
                    rv.append(buffer_ops.buffer_load(dn_rsrc, byte_off // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _compute1(acc, a, b):
            new = list(acc)
            for rk in range_constexpr(reg_k1):
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n_y):
                        idx = rm * wave_reg_n_y + rn
                        new[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            new[idx].type, a[rk][rm], b[rk][rn], new[idx]).result
            return new

        accs1 = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n_y)]
        if const_expr("phaseA" not in _ABL):
            for kt in range_constexpr(num_k1_tiles):
                a1 = _load_a1(kt); b1 = _load_b1(kt)
                accs1 = _compute1(accs1, a1, b1)

        # ── Phase A.5: descale (1/448 * scale_dn) → f16 LDS ─────────────────────
        inv448   = arith.constant(1.0 / FP8_MAX, type=fx.T.f32())
        scale_dn = ArithValue(buffer_ops.buffer_load(sdn_rsrc, 0, vec_width=1, dtype=fx.Float32))
        wave_n_r0 = wave_n * (wave_reg_n_y * WMMA_N)
        for rm in range_constexpr(wave_reg_m):
            for rn in range_constexpr(wave_reg_n_y):
                idx   = rm * wave_reg_n_y + rn
                r_col = wave_n_r0 + 16 * rn + lane16
                for si in range_constexpr(8):
                    row_in_tile = wave_m_off + 16 * rm + base8 + si
                    lin_idx     = row_in_tile * K_hh + r_col
                    y_f32 = ArithValue(accs1[idx][si]) * inv448 * scale_dn
                    SmemPtr.store(s_hh, arith.truncf(fx.T.f16(), y_f32), [lin_idx])
        gpu.barrier()

        # ── Up-projection loaders (Phase B hidden from LDS; Phase 2 input global) ─
        v8f16_ty = ir.VectorType.get([8], ir.F16Type.get())

        def _load_a_hh(kt):
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row_in_tile = wave_m_off + 16 * rm + lane16
                k_elem      = kt * WMMA_K + klane * 8
                frags.append(mlir_vector.load(v8f16_ty, s_hh.get(), [row_in_tile * K_hh + k_elem]))
            return frags

        def _load_a_ih(kt):
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row    = tile_m0 + wave_m_off + 16 * rm + lane16
                k_elem = kt * WMMA_K + klane * 8
                frags.append(buffer_ops.buffer_load(xd_rsrc, row * R + k_elem,
                                                    vec_width=8, dtype=fx.Float16))
            return frags

        def _load_b_f16(kt, gate_n0_base, rsrc, stride_n0):
            vecs = []
            for rn in range_constexpr(wave_reg_n_h):
                n0 = gate_n0_base + wave_n * wave_reg_n_h + rn
                f16_off = (n0 * stride_n0 + kt * B_STRIDE_K0
                           + klane * B_STRIDE_KLANE + lane16 * B_STRIDE_NLANE)
                vecs.append(buffer_ops.buffer_load(rsrc, f16_off, vec_width=8, dtype=fx.Float16))
            return vecs

        def _compute2(ia, fa, ga, oa, a_v, bi, bf, bg, bo):
            ni = list(ia); nf = list(fa); ng = list(ga); no = list(oa)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n_h):
                    idx = rm * wave_reg_n_h + rn
                    ni[idx] = rocdl.wmma_f32_16x16x16_f16(ni[idx].type, a_v[rm], bi[rn], ni[idx]).result
                    nf[idx] = rocdl.wmma_f32_16x16x16_f16(nf[idx].type, a_v[rm], bf[rn], nf[idx]).result
                    ng[idx] = rocdl.wmma_f32_16x16x16_f16(ng[idx].type, a_v[rm], bg[rn], ng[idx]).result
                    no[idx] = rocdl.wmma_f32_16x16x16_f16(no[idx].type, a_v[rm], bo[rn], no[idx]).result
            return ni, nf, ng, no

        # ── Epilogue constants ──────────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        c27   = arith.constant(27.0, type=fx.T.f32())
        c9    = arith.constant(9.0,  type=fx.T.f32())
        c448  = arith.constant(FP8_MAX, type=fx.T.f32())
        zero_i32 = arith.constant(0, type=fx.T.i32())
        n_ac = wave_reg_m * wave_reg_n_h

        def sighard(x):
            return ArithValue(arith.minimumf((x * c0_2 + c0_5).maximumf(c0_0), c1_0))

        # ── H-tile loop: this block walks nt_per_block hidden-column tiles ──────
        for nt_local in range_constexpr(nt_per_block):
            nt       = bid_hs * nt_per_block + nt_local   # absolute h-tile index (index)
            h_n0     = nt * (tile_n_h // 16)
            tile_nh0 = nt * tile_n_h
            gate_n0_i = h_n0 + 0 * GATE_N0_STRIDE
            gate_n0_f = h_n0 + 1 * GATE_N0_STRIDE
            gate_n0_g = h_n0 + 2 * GATE_N0_STRIDE
            gate_n0_o = h_n0 + 3 * GATE_N0_STRIDE

            i_ac = [zero_acc for _ in range_constexpr(n_ac)]
            f_ac = [zero_acc for _ in range_constexpr(n_ac)]
            g_ac = [zero_acc for _ in range_constexpr(n_ac)]
            o_ac = [zero_acc for _ in range_constexpr(n_ac)]

            if const_expr("hh" not in _ABL):
                for kt in range_constexpr(num_k_tiles_hh):
                    a  = _load_a_hh(kt)
                    bi = _load_b_f16(kt, gate_n0_i, uhh_rsrc, BHH_STRIDE_N0)
                    bf = _load_b_f16(kt, gate_n0_f, uhh_rsrc, BHH_STRIDE_N0)
                    bg = _load_b_f16(kt, gate_n0_g, uhh_rsrc, BHH_STRIDE_N0)
                    bo = _load_b_f16(kt, gate_n0_o, uhh_rsrc, BHH_STRIDE_N0)
                    i_ac, f_ac, g_ac, o_ac = _compute2(i_ac, f_ac, g_ac, o_ac, a, bi, bf, bg, bo)

            if const_expr("ih" not in _ABL):
                for kt in range_constexpr(num_k_tiles_ih):
                    a  = _load_a_ih(kt)
                    bi = _load_b_f16(kt, gate_n0_i, up_rsrc, BIH_STRIDE_N0)
                    bf = _load_b_f16(kt, gate_n0_f, up_rsrc, BIH_STRIDE_N0)
                    bg = _load_b_f16(kt, gate_n0_g, up_rsrc, BIH_STRIDE_N0)
                    bo = _load_b_f16(kt, gate_n0_o, up_rsrc, BIH_STRIDE_N0)
                    i_ac, f_ac, g_ac, o_ac = _compute2(i_ac, f_ac, g_ac, o_ac, a, bi, bf, bg, bo)

            wave_nh0 = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

            def _ld(rsrc, g, rn):
                h_col = wave_nh0 + 16 * rn + lane16
                return ArithValue(buffer_ops.buffer_load(rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

            bias_i = [_ld(bhh_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
            bias_f = [_ld(bhh_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
            bias_g = [_ld(bhh_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
            bias_o = [_ld(bhh_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]
            bih_i  = [_ld(bih_rsrc, 0, rn) for rn in range_constexpr(wave_reg_n_h)]
            bih_f  = [_ld(bih_rsrc, 1, rn) for rn in range_constexpr(wave_reg_n_h)]
            bih_g  = [_ld(bih_rsrc, 2, rn) for rn in range_constexpr(wave_reg_n_h)]
            bih_o  = [_ld(bih_rsrc, 3, rn) for rn in range_constexpr(wave_reg_n_h)]

            for rm in range_constexpr(wave_reg_m):
                wmma_m_off = wave_m_off + 16 * rm
                for rn in range_constexpr(wave_reg_n_h):
                    idx   = rm * wave_reg_n_h + rn
                    h_col = wave_nh0 + 16 * rn + lane16
                    g_rows = [tile_m0 + wmma_m_off + base8 + si for si in range_constexpr(8)]
                    if const_expr("cload" in _ABL):
                        c_vals = [ArithValue(c0_0) for si in range_constexpr(8)]
                    else:
                        c_vals = [ArithValue(buffer_ops.buffer_load(
                            c_rsrc, g_rows[si] * H + h_col, vec_width=1, dtype=fx.Float32))
                            for si in range_constexpr(8)]

                    for si in range_constexpr(8):
                        g_row = g_rows[si]
                        i_raw = ArithValue(i_ac[idx][si]) + bias_i[rn] + bih_i[rn]
                        f_raw = ArithValue(f_ac[idx][si]) + bias_f[rn] + bih_f[rn]
                        g_raw = ArithValue(g_ac[idx][si]) + bias_g[rn] + bih_g[rn]
                        o_raw = ArithValue(o_ac[idx][si]) + bias_o[rn] + bih_o[rn]

                        i_a = sighard(i_raw)
                        f_a = sighard(f_raw)
                        g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                        o_a = sighard(o_raw)

                        c_off = g_row * H + h_col
                        c_new = f_a * c_vals[si] + i_a * g_a
                        if const_expr("cstore" not in _ABL):
                            buffer_ops.buffer_store(c_new, c_rsrc, c_off)

                        if const_expr("tanh" in _ABL):
                            tanh_c = c_new
                        else:
                            x2     = c_new * c_new
                            numer  = c_new * (ArithValue(c27) + x2)
                            denom  = ArithValue(c27) + ArithValue(c9) * x2
                            tanh_c = ArithValue(arith.minimumf(
                                (numer * ArithValue(rocdl.rcp(fx.T.f32(), denom))).maximumf(cm1_0), c1_0))

                        h_new    = o_a * tanh_c
                        h_scaled = h_new * ArithValue(c448)
                        packed   = rocdl.cvt_pk_fp8_f32(fx.T.i32(), h_scaled, h_scaled, zero_i32, 0)
                        h_fp8_b  = arith.trunci(fx.T.i8(), packed)
                        if const_expr("hstore" not in _ABL):
                            buffer_ops.buffer_store(h_fp8_b, h_rsrc, g_row * H + h_col, offset_is_bytes=True)

    @flyc.jit
    def launch_fp8_factored_lstm_fused(
        arg_h_fp8_out:  fx.Tensor,
        arg_c_inout:    fx.Tensor,
        arg_h_prev_fp8: fx.Tensor,
        arg_scale_dn:   fx.Tensor,
        arg_dn_hh:      fx.Tensor,
        arg_up_hh:      fx.Tensor,
        arg_bias_hh:    fx.Tensor,
        arg_x_down:     fx.Tensor,
        arg_up_ih:      fx.Tensor,
        arg_bias_ih:    fx.Tensor,
        stream:         fx.Stream,
        m:              fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * h_split
        launcher = kernel_factored_lstm_fused(
            arg_h_fp8_out, arg_c_inout, arg_h_prev_fp8, arg_scale_dn,
            arg_dn_hh, arg_up_hh, arg_bias_hh, arg_x_down, arg_up_ih, arg_bias_ih,
        )
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            lds_alloc.finalized = False
            lds_alloc.finalize()
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_fp8_factored_lstm_fused


# =============================================================================
# Persistent kernel: all T timesteps in a single launch
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_unfactored_lstm_gemm_persistent(
    *,
    B: int,
    H: int,
    T: int,
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k: int = 32,
    k_unroll: int = 2,
    group_m: int = 8,
):
    """Persistent unfactored LSTM GEMM: all T timesteps in a single kernel launch.

    W_fused (4 MB) stays L2-warm across all T iterations instead of being
    re-fetched from HBM on each of T separate launches.

    arg_h_fp8_all: [T+1, B, H] fp8 uint8 — h[0]=initial h, h[1..T]=outputs written in-place
    arg_ih_all:    [T,   B, H, 4] f16    — all T precomputed ih values

    Grid/block layout: identical to single-step (compile_fp8_unfactored_lstm_gemm).
    T is a compile-time constant baked into the outer loop trip count.
    """
    FH = 4 * H

    assert B % tile_m == 0
    assert H % tile_n_h == 0
    assert H % tile_k == 0
    assert tile_n_h % WMMA_N == 0
    assert tile_k % WMMA_K == 0

    reg_m       = tile_m   // WMMA_M
    reg_k       = tile_k   // WMMA_K
    num_k_tiles = H // tile_k

    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m        = reg_m   // waves_m
    wave_reg_n_h      = (tile_n_h // WMMA_N) // waves_n

    grid_m = B // tile_m
    grid_h = H // tile_n_h

    K0_H            = H // 16
    B_KPACK         = 8
    B_STRIDE_NLANE  = B_KPACK
    B_STRIDE_KLANE  = 16 * B_KPACK
    B_STRIDE_K0     = 2 * 16 * B_KPACK
    B_STRIDE_N0     = K0_H * B_STRIDE_K0
    GATE_N0_STRIDE  = H // 16

    BH  = B * H        # fp8 bytes per timestep in h_fp8_all
    BH4 = B * H * 4    # f16 elements per timestep in ih_all

    @flyc.kernel
    def kernel_unfactored_lstm_persistent(
        arg_h_fp8_all: fx.Tensor,   # [T+1, B, H] fp8 uint8
        arg_c_inout:   fx.Tensor,   # [B, H]      f32
        arg_scale_hh:  fx.Tensor,   # [B]         f32
        arg_w_fused:   fx.Tensor,   # preshuffled fp8 [H, 4H]
        arg_scale_wf:  fx.Tensor,   # [4H]        f32
        arg_bias:      fx.Tensor,   # [4H]        f32
        arg_ih_all:    fx.Tensor,   # [T, B, H, 4] f16
        arg_grid_m:    fx.Int32,
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
        h_all_rsrc  = buffer_ops.create_buffer_resource(arg_h_fp8_all, max_size=True)
        wf_rsrc     = buffer_ops.create_buffer_resource(arg_w_fused,   max_size=True)
        c_rsrc      = buffer_ops.create_buffer_resource(arg_c_inout,   max_size=True)
        shh_rsrc    = buffer_ops.create_buffer_resource(arg_scale_hh,  max_size=True)
        swf_rsrc    = buffer_ops.create_buffer_resource(arg_scale_wf,  max_size=True)
        bias_rsrc   = buffer_ops.create_buffer_resource(arg_bias,      max_size=True)
        ih_all_rsrc = buffer_ops.create_buffer_resource(arg_ih_all,    max_size=True)

        # ── Gate N0 offsets ──────────────────────────────────────────────────
        h_tile_n0 = bid_h * (tile_n_h // 16)
        gate_n0_i = h_tile_n0 + 0 * GATE_N0_STRIDE
        gate_n0_f = h_tile_n0 + 1 * GATE_N0_STRIDE
        gate_n0_g = h_tile_n0 + 2 * GATE_N0_STRIDE
        gate_n0_o = h_tile_n0 + 3 * GATE_N0_STRIDE

        wave_m_off = wave_m * (wave_reg_m * WMMA_M)
        base8      = klane * 8
        wave_nh0   = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

        # ── W_fused B-tile loader — addresses unchanged across all T iters ──
        def _load_b(kt, gate_n0_base):
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

        def _flat(t_):
            f = []
            for r in t_: f.extend(r)
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

        n_a   = reg_k * wave_reg_m
        n_b   = reg_k * wave_reg_n_h
        n_ac  = wave_reg_m * wave_reg_n_h
        n_ac4 = n_ac * 4

        full_out = (num_k_tiles - 1) // k_unroll
        rem      = (num_k_tiles - 1) % k_unroll

        def _pack_accs(ia, fa, ga, oa):
            return list(ia) + list(fa) + list(ga) + list(oa)
        def _unpack_accs(flat):
            ia = flat[:n_ac]; fa = flat[n_ac:2*n_ac]
            ga = flat[2*n_ac:3*n_ac]; oa = flat[3*n_ac:]
            return ia, fa, ga, oa

        # ── Epilogue constants ────────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        c27   = arith.constant(27.0, type=fx.T.f32())
        c9    = arith.constant(9.0,  type=fx.T.f32())
        c448  = arith.constant(FP8_MAX, type=fx.T.f32())
        zero_i32 = arith.constant(0, type=fx.T.i32())

        def _bias(g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(
                bias_rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        def _swf(g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(
                swf_rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        # Hoist loop-invariant loads outside T loop (same value every timestep)
        bias_i = [_bias(0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_f = [_bias(1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_g = [_bias(2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_o = [_bias(3, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_i  = [_swf(0, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_f  = [_swf(1, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_g  = [_swf(2, rn) for rn in range_constexpr(wave_reg_n_h)]
        swf_o  = [_swf(3, rn) for rn in range_constexpr(wave_reg_n_h)]

        # scale_hh [B] f32 is constant across T — hoist 8-value loads per rm
        all_shh_vals = []
        for rm in range_constexpr(wave_reg_m):
            wmma_m_off_rm = wave_m_off + 16 * rm
            shh_base_rm   = tile_m0 + wmma_m_off_rm + base8
            shh_lo = buffer_ops.buffer_load(shh_rsrc, shh_base_rm,     vec_width=4, dtype=fx.Float32)
            shh_hi = buffer_ops.buffer_load(shh_rsrc, shh_base_rm + 4, vec_width=4, dtype=fx.Float32)
            all_shh_vals.append(
                [mlir_vector.extract(shh_lo, static_position=[i], dynamic_position=[]) for i in range(4)] +
                [mlir_vector.extract(shh_hi, static_position=[i], dynamic_position=[]) for i in range(4)]
            )

        # ── T timestep loop ──────────────────────────────────────────────────
        # Carry: hh_t_base (byte offset into h_fp8_all) and
        #        ih_t_base (f16-element offset into ih_all).
        # BH and BH4 must be computed from the runtime grid_m (= B/tile_m),
        # NOT from the compile-time B, because B may differ at runtime.
        zero_idx  = arith.constant(0, type=fx.T.index())
        gm_idx    = fx.arith.index_cast(fx.T.index(), arg_grid_m)
        BH_idx    = gm_idx * arith.constant(tile_m * H,     type=fx.T.index())
        BH4_idx   = gm_idx * arith.constant(tile_m * H * 4, type=fx.T.index())

        for _tv, t_st in range(0, T, 1, init=[zero_idx, zero_idx]):
            hh_t_base  = t_st[0]               # byte offset to h_fp8_all[t, 0, 0]
            ih_t_base  = t_st[1]               # f16-elem offset to ih_all[t, 0, 0, 0]
            h_out_base = hh_t_base + BH_idx    # byte offset to h_fp8_all[t+1, 0, 0]

            # A-tile loader (closes over hh_t_base to read h[t])
            def _load_a(kt):
                vecs = []
                for rk in range_constexpr(reg_k):
                    rv = []
                    col = kt * tile_k + 16 * rk + klane * 8
                    for rm in range_constexpr(wave_reg_m):
                        row      = tile_m0 + wave_m_off + 16 * rm + lane16
                        byte_off = hh_t_base + row * H + col
                        rv.append(buffer_ops.buffer_load(h_all_rsrc, byte_off // 4,
                                                         vec_width=2, dtype=fx.Int32))
                    vecs.append(rv)
                return vecs

            # Reset accumulators for this timestep
            zero_acc = fx.full(8, 0.0, fx.Float32)
            i_ac = [zero_acc for _ in range_constexpr(n_ac)]
            f_ac = [zero_acc for _ in range_constexpr(n_ac)]
            g_ac = [zero_acc for _ in range_constexpr(n_ac)]
            o_ac = [zero_acc for _ in range_constexpr(n_ac)]

            # Prefetch first A and B tiles
            a_cur  = _load_a(0)
            bi_cur = _load_b(0, gate_n0_i)
            bf_cur = _load_b(0, gate_n0_f)
            bg_cur = _load_b(0, gate_n0_g)
            bo_cur = _load_b(0, gate_n0_o)

            init_k = (_flat(a_cur) + _pack_accs(i_ac, f_ac, g_ac, o_ac)
                      + _flat(bi_cur) + _flat(bf_cur) + _flat(bg_cur) + _flat(bo_cur))

            # ── Software-pipelined K-loop (identical structure to single-step) ─
            if const_expr(full_out > 0):
                for iv, st in range(0, full_out * k_unroll, k_unroll, init=init_k):
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
                    res = yield (_flat(s_a) + _pack_accs(s_ia, s_fa, s_ga, s_oa)
                                 + _flat(s_bi) + _flat(s_bf) + _flat(s_bg) + _flat(s_bo))
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
                    a_cur  = _unflat_a(_flat(a_nxt))
                    bi_cur = _unflat_b(_flat(bi_nxt)); bf_cur = _unflat_b(_flat(bf_nxt))
                    bg_cur = _unflat_b(_flat(bg_nxt)); bo_cur = _unflat_b(_flat(bo_nxt))

            i_ac, f_ac, g_ac, o_ac = _compute4(i_ac, f_ac, g_ac, o_ac, a_cur, bi_cur, bf_cur, bg_cur, bo_cur)

            # ── Epilogue ─────────────────────────────────────────────────────
            for rm in range_constexpr(wave_reg_m):
                wmma_m_off = wave_m_off + 16 * rm
                shh_vals   = all_shh_vals[rm]

                for rn in range_constexpr(wave_reg_n_h):
                    idx   = rm * wave_reg_n_h + rn
                    h_col = wave_nh0 + 16 * rn + lane16

                    def sighard(x):
                        return ArithValue(arith.minimumf(
                            (x * c0_2 + c0_5).maximumf(c0_0), c1_0))

                    g_rows  = [tile_m0 + wmma_m_off + base8 + si for si in range_constexpr(8)]
                    # Load ih for this timestep (with ih_t_base)
                    ih_vecs = [buffer_ops.buffer_load(
                        ih_all_rsrc,
                        ih_t_base + g_rows[si] * (H * 4) + h_col * 4,
                        vec_width=4, dtype=fx.Float16)
                        for si in range_constexpr(8)]
                    # c is updated each step so must be loaded inside the T loop
                    c_vals  = [ArithValue(buffer_ops.buffer_load(
                        c_rsrc, g_rows[si] * H + h_col, vec_width=1, dtype=fx.Float32))
                        for si in range_constexpr(8)]

                    for si in range_constexpr(8):
                        g_row = g_rows[si]
                        s_hh  = ArithValue(shh_vals[si])
                        iv2   = ih_vecs[si]
                        def _ih(gi, _iv=iv2):
                            return ArithValue(mlir_vector.extract(
                                _iv, static_position=[gi], dynamic_position=[])).extf(fx.T.f32())

                        i_raw = ArithValue(i_ac[idx][si]) * s_hh * swf_i[rn] + bias_i[rn] + _ih(0)
                        f_raw = ArithValue(f_ac[idx][si]) * s_hh * swf_f[rn] + bias_f[rn] + _ih(1)
                        g_raw = ArithValue(g_ac[idx][si]) * s_hh * swf_g[rn] + bias_g[rn] + _ih(2)
                        o_raw = ArithValue(o_ac[idx][si]) * s_hh * swf_o[rn] + bias_o[rn] + _ih(3)

                        i_a = sighard(i_raw)
                        f_a = sighard(f_raw)
                        g_a = ArithValue(arith.minimumf(g_raw.maximumf(cm1_0), c1_0))
                        o_a = sighard(o_raw)

                        c_new = f_a * c_vals[si] + i_a * g_a
                        buffer_ops.buffer_store(c_new, c_rsrc, g_row * H + h_col)

                        x2     = c_new * c_new
                        numer  = c_new * (ArithValue(c27) + x2)
                        denom  = ArithValue(c27) + ArithValue(c9) * x2
                        tanh_c = ArithValue(arith.minimumf(
                            (numer * ArithValue(rocdl.rcp(fx.T.f32(), denom))).maximumf(cm1_0), c1_0))

                        h_new    = o_a * tanh_c
                        h_scaled = h_new * ArithValue(c448)
                        packed   = rocdl.cvt_pk_fp8_f32(
                            fx.T.i32(), h_scaled, h_scaled, zero_i32, 0)
                        h_fp8_b  = arith.trunci(fx.T.i8(), packed)
                        # Write h[t+1] using h_out_base byte offset into h_fp8_all
                        buffer_ops.buffer_store(
                            h_fp8_b, h_all_rsrc,
                            h_out_base + g_row * H + h_col,
                            offset_is_bytes=True)

            # Advance T-loop carried byte offsets
            _ = yield [hh_t_base + BH_idx, ih_t_base + BH4_idx]

    # ── Host launcher ──────────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_unfactored_lstm_persistent(
        arg_h_fp8_all: fx.Tensor,
        arg_c_inout:   fx.Tensor,
        arg_scale_hh:  fx.Tensor,
        arg_w_fused:   fx.Tensor,
        arg_scale_wf:  fx.Tensor,
        arg_bias:      fx.Tensor,
        arg_ih_all:    fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_h
        launcher = kernel_unfactored_lstm_persistent(
            arg_h_fp8_all, arg_c_inout, arg_scale_hh,
            arg_w_fused, arg_scale_wf, arg_bias, arg_ih_all, dyn_grid_m,
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_fp8_unfactored_lstm_persistent


__all__ = [
    "compile_fp8_unfactored_lstm_gemm",
    "compile_fp8_unfactored_lstm_gemm_persistent",
    "compile_fp8_fused_lstm",
    "compile_fp8_factored_lstm",
    "compile_fp8_factored_lstm_fused",
    "make_w_fused",
    "make_ih_t_interleaved",
    "preshuffle_b_fp8",
    "preshuffle_b_f16",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
