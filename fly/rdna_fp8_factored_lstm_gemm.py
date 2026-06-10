"""FP8 Factored LSTM GEMM Kernel for RDNA4 (gfx1201, wave32).

Phase 1 uses wmma_f32_16x16x16_fp8_fp8; Phase 2 uses wmma_f32_16x16x16_f16.
The intermediate y is stored as f16 in LDS (no fp8 requantization step).

  y       = hh_fp8[B,H] @ dn_weight_fp8[H,R]^T          Phase 1 (K=H=1024)
  gates   = y_f16[B,R]  @ up_weight_f16[R,4H]^T          Phase 2 (K=R=128)
           + up_bias[4H] + ih_t[B,4H]
  i,f,g,o = chunk4(gates, dim=H)
  i = clamp(0.2*i+0.5, 0, 1); f = same; g = clamp(g,-1,1); o = same
  c_new   = f*c + i*g
  h_new   = o*tanh(c_new)  -> f16 output

LDS layout  (static, via SmemAllocator):
  [0, tile_m*R*2)     f16  y  (8 KB for tile_m=32, R=128)

Grid: (B/tile_m * H/tile_n_h, 1, 1)
Block: (64 threads = 2 waves x 32 lanes)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16   # fp8 WMMA K-size (only variant on gfx1201)

WAVE_SIZE = 32
SHUFFLE_DISTS = [1, 2, 4, 8, 16]


# =============================================================================
# Host-side helpers
# =============================================================================

def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K,N] fp8 → [N0, K0, KLane=2, NLane=16, KPack=8] bytes."""
    import torch
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_view = B_kn.view(torch.uint8)
    B_r = B_view.reshape(K // 16, 2, 8, N // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()   # [N0, K0, 2, 16, 8]


def preshuffle_b_f16(B_kn):
    """Preshuffle B[K,N] f16 → [N0, K0, KLane=2, NLane=16, KPack=8] f16."""
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_r = B_kn.reshape(K // 16, 2, 8, N // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()   # [N0, K0, 2, 16, 8]


def fp8_quantize_per_token(x_f32):
    """Quantize f32 → fp8_e4m3fn with per-token scale. Returns (fp8, scale[M])."""
    import torch
    amax  = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    x_fp8 = (x_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def fp8_quantize_per_channel(x_f32):
    """Quantize f32 → fp8_e4m3fn with per-channel scale. Returns (fp8, scale[N])."""
    import torch
    amax  = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    x_fp8 = (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale


# =============================================================================
# Kernel compiler
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_factored_lstm_gemm(
    *,
    B: int,
    H: int,
    R: int,
    tile_m: int = 32,
    tile_n_h: int = 32,
    tile_k1: int = 32,
    k_unroll: int = 2,
    tile_k2: int = 16,   # fp8 WMMA K=16
    group_m: int = 8,
):
    """Compile fused factored LSTM GEMM for gfx1201.

    Phase 1: wmma_fp8_fp8  (dn_weight must be preshuffled fp8 + scale_dn)
    Phase 2: wmma_f16      (up_weight must be preshuffled f16, no scale needed)
    y intermediate is stored as f16 in LDS — no fp8 requantization.

    Returns launcher(h_fp16_out, c_inout, hh, scale_hh,
                      dn_weight, scale_dn, up_weight,
                      bias, ih_t, stream, m).
    """
    assert tile_k2 == WMMA_K, f"WMMA K=16 required, got tile_k2={WMMA_K}"
    assert B % tile_m == 0
    assert H % tile_n_h == 0
    assert H % tile_k1 == 0
    assert R % tile_k2 == 0
    assert tile_n_h % WMMA_N == 0
    assert tile_k1 % WMMA_K == 0

    # ── Phase 1 tile dimensions ──────────────────────────────────────────────
    tile_n_y   = R
    reg_m      = tile_m   // WMMA_M        # 2
    reg_n_y    = tile_n_y // WMMA_N        # 8
    reg_k1     = tile_k1  // WMMA_K        # 2

    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n   # 2
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE   # 64
    wave_reg_m        = reg_m   // waves_m  # 2
    wave_reg_n_y      = reg_n_y // waves_n  # 4

    num_k1_tiles = H // tile_k1
    grid_m       = B // tile_m
    grid_h       = H // tile_n_h

    # ── Phase 2 tile dimensions ──────────────────────────────────────────────
    reg_k2       = tile_k2 // WMMA_K   # 1
    num_k2_tiles = R // tile_k2        # 8  (R=128 / 16)
    wave_reg_n_h = (tile_n_h // WMMA_N) // waves_n   # 1

    # ── Phase 1 B-strides: dn_weight fp8 [N0, K0, KLane=2, NLane=16, KPack=8] ─
    K0_H            = H // 16
    B1_KPACK        = 8
    B1_STRIDE_NLANE = B1_KPACK
    B1_STRIDE_KLANE = 16 * B1_KPACK       # 128
    B1_STRIDE_K0    = 2 * 16 * B1_KPACK   # 256
    B1_STRIDE_N0    = K0_H * B1_STRIDE_K0

    # ── Phase 2 B-strides: up_weight f16, same preshuffle shape as fp8 ───────
    # All strides in f16 ELEMENT units (buffer_load with dtype=Float16 uses elem offsets)
    K0_R            = R // 16              # 8
    B2_KPACK        = 8
    B2_STRIDE_NLANE = B2_KPACK             # 8 f16 per NLane slot
    B2_STRIDE_KLANE = 16 * B2_KPACK        # 128 f16
    B2_STRIDE_K0    = 2 * 16 * B2_KPACK    # 256 f16
    B2_STRIDE_N0    = K0_R * B2_STRIDE_K0  # 2048 f16
    GATE2_N0_STRIDE = H // 16              # 64 (in N0 units)

    # ── LDS allocation ───────────────────────────────────────────────────────
    # s_y_f16  : tile_m * R f16  (2 bytes/element = 8 KB for tile_m=32, R=128)
    lds_alloc    = SmemAllocator(None, global_sym_name="smem_factored_lstm")
    f16_byte_off = lds_alloc._align(lds_alloc.ptr, 32)
    f16_size     = tile_m * R * 2                        # 8192 bytes
    lds_alloc.ptr = f16_byte_off + f16_size

    @flyc.kernel
    def kernel_factored_lstm(
        arg_h_fp16_out: fx.Tensor,   # [B, H]   f16  — output h[t+1]
        arg_c_inout:    fx.Tensor,   # [B, H]   f32  — cell state in-place
        arg_hh:         fx.Tensor,   # [B, H]   fp8  — input h[t]
        arg_scale_hh:   fx.Tensor,   # [B]      f32  — per-token scale for hh
        arg_dn_weight:  fx.Tensor,   # preshuffled fp8 dn_weight [H, R]
        arg_scale_dn:   fx.Tensor,   # [R]      f32  — per-channel scale dn_weight
        arg_up_weight:  fx.Tensor,   # preshuffled f16 up_weight [R, 4H]
        arg_bias:       fx.Tensor,   # [4H]     f32
        arg_ih_t:       fx.Tensor,   # [B, 4H]  f16
        arg_grid_m:     fx.Int32,
    ):
        # ── Thread / block indices ───────────────────────────────────────────
        tid    = gpu.thread_id("x")
        pid    = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        pid_i32   = fx.arith.index_cast(fx.T.i32(), pid)
        grid_h_c  = fx.arith.constant(grid_h,  type=fx.T.i32())
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
        hh_rsrc   = buffer_ops.create_buffer_resource(arg_hh,        max_size=True)
        dn_rsrc   = buffer_ops.create_buffer_resource(arg_dn_weight, max_size=True)
        up_rsrc   = buffer_ops.create_buffer_resource(arg_up_weight, max_size=True)
        c_rsrc    = buffer_ops.create_buffer_resource(arg_c_inout,   max_size=True)
        h_rsrc    = buffer_ops.create_buffer_resource(arg_h_fp16_out,max_size=True)
        shh_rsrc  = buffer_ops.create_buffer_resource(arg_scale_hh,  max_size=True)
        sdn_rsrc  = buffer_ops.create_buffer_resource(arg_scale_dn,  max_size=True)
        bias_rsrc = buffer_ops.create_buffer_resource(arg_bias,      max_size=True)
        ih_rsrc   = buffer_ops.create_buffer_resource(arg_ih_t,      max_size=True)

        # ── LDS view ─────────────────────────────────────────────────────────
        base_ptr = lds_alloc.get_base()
        s_y_f16  = SmemPtr(base_ptr, f16_byte_off, fx.T.f16(), shape=(tile_m * R,))
        s_y_f16.get()

        # ── Phase 1 sub-functions: FP8 × FP8 → F32 over K=H ─────────────────

        def _load_a1(kt):
            vecs = []
            for rk in range_constexpr(reg_k1):
                rv = []
                col = kt * tile_k1 + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row     = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    byte_off = row * H + col
                    rv.append(buffer_ops.buffer_load(hh_rsrc, byte_off // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _load_b1(kt):
            vecs   = []
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

        # ── Phase 1: software-pipelined K-loop over H ────────────────────────
        zero_acc = fx.full(8, 0.0, fx.Float32)
        accs1    = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n_y)]
        a1_cur   = _load_a1(0)
        b1_cur   = _load_b1(0)

        full_out = (num_k1_tiles - 1) // k_unroll
        rem      = (num_k1_tiles - 1) % k_unroll

        def _flat(tile):
            f = []
            for row in tile: f.extend(row)
            return f

        def _unflat_a1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_m):
                    r.append(f[i]); i += 1
                out.append(r)
            return out

        def _unflat_b1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_n_y):
                    r.append(f[i]); i += 1
                out.append(r)
            return out

        n_a1  = reg_k1 * wave_reg_m
        n_ac1 = wave_reg_m * wave_reg_n_y
        init  = _flat(a1_cur) + list(accs1) + _flat(b1_cur)

        if const_expr(full_out > 0):
            for iv, st in range(0, full_out * k_unroll, k_unroll, init=init):
                s_a  = _unflat_a1(list(st[:n_a1]))
                s_ac = list(st[n_a1 : n_a1 + n_ac1])
                s_b  = _unflat_b1(list(st[n_a1 + n_ac1:]))
                for j in range_constexpr(k_unroll):
                    nkt   = iv + j + 1
                    a_nxt = _load_a1(nkt)
                    b_nxt = _load_b1(nkt)
                    s_ac  = _compute1(s_ac, s_a, s_b)
                    s_a   = _unflat_a1(_flat(a_nxt))
                    s_b   = _unflat_b1(_flat(b_nxt))
                res = yield _flat(s_a) + list(s_ac) + _flat(s_b)
            a1_cur = _unflat_a1(list(res[:n_a1]))
            accs1  = list(res[n_a1 : n_a1 + n_ac1])
            b1_cur = _unflat_b1(list(res[n_a1 + n_ac1:]))

        if const_expr(rem > 0):
            for j in range_constexpr(rem):
                nkt    = full_out * k_unroll + j + 1
                a_nxt  = _load_a1(nkt)
                b_nxt  = _load_b1(nkt)
                accs1  = _compute1(accs1, a1_cur, b1_cur)
                a1_cur = _unflat_a1(_flat(a_nxt))
                b1_cur = _unflat_b1(_flat(b_nxt))

        accs1 = _compute1(accs1, a1_cur, b1_cur)

        # ── Phase 1.5: descale y → f16, store to LDS ────────────────────────
        wave_n_r0  = wave_n * (wave_reg_n_y * WMMA_N)
        wave_m_off = wave_m * (wave_reg_m  * WMMA_M)   # = 0  (waves_m=1)
        base8      = klane * 8

        # Cache scale_dn per rn
        sdn_cache = []
        for rn in range_constexpr(wave_reg_n_y):
            r_col = wave_n_r0 + 16 * rn + lane16
            sdn_cache.append(ArithValue(
                buffer_ops.buffer_load(sdn_rsrc, r_col, vec_width=1, dtype=fx.Float32)))

        # Cache scale_hh per (rm, si)
        shh_all = []
        for rm in range_constexpr(wave_reg_m):
            shh_rm = []
            for si in range_constexpr(8):
                m_row = tile_m0 + wave_m_off + 16 * rm + base8 + si
                shh_rm.append(ArithValue(
                    buffer_ops.buffer_load(shh_rsrc, m_row, vec_width=1, dtype=fx.Float32)))
            shh_all.append(shh_rm)

        for rm in range_constexpr(wave_reg_m):
            for rn in range_constexpr(wave_reg_n_y):
                idx   = rm * wave_reg_n_y + rn
                r_col = wave_n_r0 + 16 * rn + lane16
                for si in range_constexpr(8):
                    row_in_tile = wave_m_off + 16 * rm + base8 + si
                    lin_idx     = row_in_tile * R + r_col
                    y_f32 = ArithValue(accs1[idx][si]) * shh_all[rm][si] * sdn_cache[rn]
                    SmemPtr.store(s_y_f16, arith.truncf(fx.T.f16(), y_f32), [lin_idx])

        gpu.barrier()

        # ── Phase 2: F16 × F16 → F32 over K=R (8 tiles of 16) ───────────────
        v8f16_ty = ir.VectorType.get([8], ir.F16Type.get())

        def _load_a2(kt):
            """Load y_f16 A-fragments from LDS as v8f16."""
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row_in_tile = wave_m_off + 16 * rm + lane16   # M-row index
                k_elem      = kt * WMMA_K + klane * 8          # f16 element offset
                f16_off     = row_in_tile * R + k_elem
                frags.append(mlir_vector.load(v8f16_ty, s_y_f16.get(), [f16_off]))
            return frags

        def _load_b2(kt, gate_n0_base):
            """Load up_weight_f16 B-tile as v8f16 (f16 element offsets)."""
            vecs = []
            for rn in range_constexpr(wave_reg_n_h):
                n0      = gate_n0_base + wave_n * wave_reg_n_h + rn
                f16_off = (n0 * B2_STRIDE_N0 + kt * B2_STRIDE_K0
                           + klane * B2_STRIDE_KLANE + lane16 * B2_STRIDE_NLANE)
                vecs.append(buffer_ops.buffer_load(up_rsrc, f16_off,
                                                   vec_width=8, dtype=fx.Float16))
            return vecs

        def _compute2(ia, fa, ga, oa, a_v, bi, bf, bg, bo):
            ni = list(ia); nf = list(fa); ng = list(ga); no = list(oa)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n_h):
                    idx = rm * wave_reg_n_h + rn
                    ni[idx] = rocdl.wmma_f32_16x16x16_f16(
                        ni[idx].type, a_v[rm], bi[rn], ni[idx]).result
                    nf[idx] = rocdl.wmma_f32_16x16x16_f16(
                        nf[idx].type, a_v[rm], bf[rn], nf[idx]).result
                    ng[idx] = rocdl.wmma_f32_16x16x16_f16(
                        ng[idx].type, a_v[rm], bg[rn], ng[idx]).result
                    no[idx] = rocdl.wmma_f32_16x16x16_f16(
                        no[idx].type, a_v[rm], bo[rn], no[idx]).result
            return ni, nf, ng, no

        n_ac2 = wave_reg_m * wave_reg_n_h
        i_ac  = [zero_acc for _ in range_constexpr(n_ac2)]
        f_ac  = [zero_acc for _ in range_constexpr(n_ac2)]
        g_ac  = [zero_acc for _ in range_constexpr(n_ac2)]
        o_ac  = [zero_acc for _ in range_constexpr(n_ac2)]

        h_tile_n0 = bid_h * (tile_n_h // 16)
        gate_n0_i = h_tile_n0 + 0 * GATE2_N0_STRIDE
        gate_n0_f = h_tile_n0 + 1 * GATE2_N0_STRIDE
        gate_n0_g = h_tile_n0 + 2 * GATE2_N0_STRIDE
        gate_n0_o = h_tile_n0 + 3 * GATE2_N0_STRIDE

        for kt2 in range_constexpr(num_k2_tiles):
            a2   = _load_a2(kt2)
            bi_v = _load_b2(kt2, gate_n0_i)
            bf_v = _load_b2(kt2, gate_n0_f)
            bg_v = _load_b2(kt2, gate_n0_g)
            bo_v = _load_b2(kt2, gate_n0_o)
            i_ac, f_ac, g_ac, o_ac = _compute2(
                i_ac, f_ac, g_ac, o_ac, a2, bi_v, bf_v, bg_v, bo_v)

        # ── Phase 3: LSTM epilogue ────────────────────────────────────────────
        c0_2  = arith.constant(0.2,  type=fx.T.f32())
        c0_5  = arith.constant(0.5,  type=fx.T.f32())
        c0_0  = arith.constant(0.0,  type=fx.T.f32())
        c1_0  = arith.constant(1.0,  type=fx.T.f32())
        cm1_0 = arith.constant(-1.0, type=fx.T.f32())
        c2_0  = arith.constant(2.0,  type=fx.T.f32())
        nl2e  = arith.constant(-1.4426950408889634, type=fx.T.f32())

        wave_nh0 = tile_nh0 + wave_n * (wave_reg_n_h * WMMA_N)

        def _bias(g, rn):
            h_col = wave_nh0 + 16 * rn + lane16
            return ArithValue(buffer_ops.buffer_load(
                bias_rsrc, g * H + h_col, vec_width=1, dtype=fx.Float32))

        bias_i = [_bias(0, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_f = [_bias(1, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_g = [_bias(2, rn) for rn in range_constexpr(wave_reg_n_h)]
        bias_o = [_bias(3, rn) for rn in range_constexpr(wave_reg_n_h)]

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m_off + 16 * rm

            for rn in range_constexpr(wave_reg_n_h):
                idx   = rm * wave_reg_n_h + rn
                h_col = wave_nh0 + 16 * rn + lane16

                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + si

                    ih_row_base = g_row * (4 * H) + h_col
                    def _ih_f32(gate_off):
                        return ArithValue(buffer_ops.buffer_load(
                            ih_rsrc, ih_row_base + gate_off,
                            vec_width=1, dtype=fx.Float16)).extf(fx.T.f32())

                    # Phase 2 uses actual f16 weights → no descaling needed
                    i_raw = ArithValue(i_ac[idx][si]) + bias_i[rn] + _ih_f32(0 * H)
                    f_raw = ArithValue(f_ac[idx][si]) + bias_f[rn] + _ih_f32(1 * H)
                    g_raw = ArithValue(g_ac[idx][si]) + bias_g[rn] + _ih_f32(2 * H)
                    o_raw = ArithValue(o_ac[idx][si]) + bias_o[rn] + _ih_f32(3 * H)

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

                    emu    = ArithValue(rocdl.exp2(fx.T.f32(), c_new * c2_0 * nl2e))
                    sig2   = ArithValue(rocdl.rcp(fx.T.f32(), c1_0 + emu))
                    tanh_c = c2_0 * sig2 - c1_0

                    h_new  = o_a * tanh_c
                    h_fp16 = arith.truncf(fx.T.f16(), h_new)
                    buffer_ops.buffer_store(h_fp16, h_rsrc, g_row * H + h_col)

    # ── Host launcher ─────────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_factored_lstm(
        arg_h_fp16_out: fx.Tensor,
        arg_c_inout:    fx.Tensor,
        arg_hh:         fx.Tensor,
        arg_scale_hh:   fx.Tensor,
        arg_dn_weight:  fx.Tensor,
        arg_scale_dn:   fx.Tensor,
        arg_up_weight:  fx.Tensor,
        arg_bias:       fx.Tensor,
        arg_ih_t:       fx.Tensor,
        stream:         fx.Stream,
        m:              fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_h
        launcher = kernel_factored_lstm(
            arg_h_fp16_out, arg_c_inout, arg_hh, arg_scale_hh,
            arg_dn_weight, arg_scale_dn, arg_up_weight,
            arg_bias, arg_ih_t, dyn_grid_m,
        )
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            lds_alloc.finalized = False
            lds_alloc.finalize()
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_fp8_factored_lstm


__all__ = [
    "compile_fp8_factored_lstm_gemm",
    "preshuffle_b_fp8",
    "preshuffle_b_f16",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
