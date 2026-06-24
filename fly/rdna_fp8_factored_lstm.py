"""FP8 Factored LSTM step for RDNA4 (gfx1201, wave32).

`compile_fp8_factored_lstm`: the LSTM gate update with BOTH hidden and input
projections kept low-rank — no K=H GEMM. The expensive recurrent hh@W_fused
(K=H=1024) is replaced by two cheap K=128 f16 up-projections:

  gates = hh_down @ up_hh (f16, K=K_hh) + x_down @ up_ih (f16, K=R) + bias_hh + bias_ih
  i,f,o = clamp(0.2*x+0.5, 0, 1); g = clamp(x, -1, 1)
  c_new = f*c + i*g
  h_new = o*tanh(c_new)  → fp8 e4m3 output (fixed scale 1/448)

hh_down (= h @ dn_hh, recurrent) is produced per-step by compile_fp8_down_proj
(rdna_fp8_factored_gemm.py); x_down (= x @ dn_ih) is precomputed once. h_new is
provably bounded to [-1,1], so a fixed 1/448 scale maps it exactly onto e4m3 and
the quantize is fused into the epilogue (one fp8 byte per thread).

Two f16 WMMA up-projections over K=128, no LDS. Shared helpers (preshuffle_b_fp8/
_f16, fp8_quantize_per_token/_per_channel) live here too.

Grid: (B/tile_m * H/tile_n_h, 1, 1)   Block: 64 threads = 2 waves x 32 lanes
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


def preshuffle_b_f16(B_kn):
    """Preshuffle B[K,N] f16 → [N0, K0, KLane=2, NLane=16, KPack=8] f16."""
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_r = B_kn.reshape(K // 16, 2, 8, N // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()



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


__all__ = [
    "compile_fp8_factored_lstm",
    "preshuffle_b_fp8",
    "preshuffle_b_f16",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
