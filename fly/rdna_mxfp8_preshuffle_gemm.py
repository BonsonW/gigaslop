"""MXFP8 Preshuffle GEMM for RDNA4 (gfx120x, wave32).

  C[M,N] = A[M,K] @ B[K,N]

A is fp8_e4m3fn [M,K] bytes with MXFP8 per-32-element E8M0 block scales
along K: A_scale[M, K//32] uint8.  B uses per-channel f32 scale.

Output is fp16.  Accumulation in f32.

B must be preshuffled to [N//16, K//16, 2, 16, 8] bytes (same layout as
rdna_fp8_preshuffle_gemm).

MXFP8 dequantization per K-block kb (32 elements):
  a_scale_f32 = (A_scale[m, kb].uint8 << 23).bitcast(f32)   # E8M0 → f32

K-loop is fully unrolled (range_constexpr) — one iteration per 32-element
MXFP8 block.  Running accumulators are updated after each block.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
TILE_K = 32   # fixed = MXFP8 block size


# =============================================================================
# Host-side helpers
# =============================================================================


def mxfp8_quantize_a(x_f32, block_size: int = 32):
    """Quantize f32 [M,K] → fp8_e4m3fn bytes + E8M0 uint8 block scales [M, K//32].

    Dequantize with:  A_f32 ≈ A_fp8.float() * (A_scale.int() << 23).view(float32)
    """
    import torch

    M, K = x_f32.shape
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"
    x_blocks = x_f32.reshape(M, K // block_size, block_size)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)          # [M, K//32]
    amax_i32 = amax.view(torch.int32)
    exp_field = ((amax_i32 + 0x200000) & 0xFF800000) >> 23
    e8m0 = torch.clamp(exp_field - 8, min=0).to(torch.uint8)     # [M, K//32]
    quant_scale = ((254 - e8m0.int()) << 23).view(torch.float32) # [M, K//32]
    x_fp8 = (x_blocks * quant_scale.unsqueeze(-1)).reshape(M, K).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return x_fp8, e8m0


def fp8_quantize_per_channel(x_f32):
    """Quantize [K,N] → fp8_e4m3fn + per-channel f32 scale [N]."""
    import torch

    amax = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    x_fp8 = (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale


def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K,N] fp8 → [N//16, K//16, 2, 16, 8] bytes."""
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_view = B_kn.view(torch.uint8)
    B_reshaped = B_view.reshape(K // 16, 2, 8, N // 16, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_mxfp8_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 32,
    tile_n: int = None,
    group_m: int = 8,
):
    """Compile MXFP8 GEMM for RDNA4.

    A is raw fp8 [M,K] + E8M0 block scales [M, K//32].
    B is preshuffled fp8 + per-channel f32 scale [N].
    Output C is fp16 [M,N].

    tile_k is fixed at 32 (= MXFP8 block size).
    K-loop is fully unrolled — one iteration per K-block.
    """
    if tile_n is None:
        tile_n = 256 if M >= 256 else 128

    WAVE_SIZE = 32
    assert tile_m % WMMA_M == 0
    assert tile_n % WMMA_N == 0
    assert TILE_K % WMMA_K == 0
    assert M % tile_m == 0
    assert N % tile_n == 0
    assert K % TILE_K == 0

    reg_m = tile_m // WMMA_M
    reg_n = tile_n // WMMA_N
    reg_k = TILE_K // WMMA_K   # always 2

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

    num_k_blocks = K // TILE_K   # number of MXFP8 blocks along K
    grid_n = N // tile_n

    K0_total = K // 16
    B_KPACK       = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0    = 2 * 16 * B_KPACK
    B_STRIDE_N0    = K0_total * B_STRIDE_K0

    n_acc = wave_reg_m * wave_reg_n

    @flyc.kernel
    def kernel_mxfp8_gemm(
        arg_c:       fx.Tensor,
        arg_a:       fx.Tensor,
        arg_a_scale: fx.Tensor,   # uint8 [M, K//32] E8M0 block scales
        arg_b:       fx.Tensor,
        arg_scale_b: fx.Tensor,   # f32 [N] per-channel weight scale
        arg_grid_m:  fx.Int32,
    ):
        tid = fx.gpu.thread_id("x")
        pid = fx.gpu.block_id("x")

        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        # L2 swizzle
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
        a_scale_rsrc = buffer_ops.create_buffer_resource(arg_a_scale, max_size=True)
        b_rsrc       = buffer_ops.create_buffer_resource(arg_b,       max_size=True)
        c_rsrc       = buffer_ops.create_buffer_resource(arg_c,       max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        def _load_a_tile(kb):
            """Load A fp8 tile for K-block kb. Returns [reg_k][wave_reg_m] v2i32."""
            a_vecs = []
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                col_base = kb * TILE_K + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    byte_off = row * K + col_base
                    dword_off = byte_off // 4
                    a_raw = buffer_ops.buffer_load(a_rsrc, dword_off, vec_width=2, dtype=fx.Int32)
                    rk_vecs.append(a_raw)
                a_vecs.append(rk_vecs)
            return a_vecs

        def _load_b_tile(kb):
            """Load B fp8 tile for K-block kb. Returns [reg_k][wave_reg_n] v2i32."""
            b_vecs = []
            n0_base = tile_n0 // 16 + wave_n * wave_reg_n
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                k0 = kb * reg_k + rk
                for rn in range_constexpr(wave_reg_n):
                    n0 = n0_base + rn
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

        # Running accumulators: one Vector(v8f32) per (rm, rn) pair
        zero_acc     = fx.full(8, 0.0, fx.Float32)
        running_accs = [zero_acc for _ in range_constexpr(n_acc)]

        # SCF K-loop — one MXFP8 block (32 K-elements) per iteration.
        # Carrying running_accs as loop state avoids 16x unrolling of heavy code.
        init_state = list(running_accs)
        for iv, state in range(0, num_k_blocks, 1, init=init_state):
            s_accs = list(state)

            a_vecs = _load_a_tile(iv)
            b_vecs = _load_b_tile(iv)

            zero_block = fx.full(8, 0.0, fx.Float32)
            block_accs = _do_compute([zero_block] * n_acc, a_vecs, b_vecs)

            # E8M0 scale for this K-block: A_scale[row, iv]
            base8  = klane * 8
            iv_i32 = ArithValue(fx.arith.index_cast(fx.T.i32(), iv))
            new_s_accs = []
            for rm in range_constexpr(wave_reg_m):
                wmma_m_off = wave_m * (wave_reg_m * WMMA_M) + 16 * rm
                a_sc_cache = []
                for si in range_constexpr(8):
                    g_row_si  = tile_m0 + wmma_m_off + base8 + si
                    g_row_i32 = ArithValue(fx.arith.index_cast(fx.T.i32(), g_row_si))
                    byte_off  = g_row_i32 * (K // TILE_K) + iv_i32
                    sc_byte   = buffer_ops.buffer_load(
                        a_scale_rsrc, byte_off, vec_width=1, dtype=T.i8
                    )
                    sc_i32    = ArithValue(ArithValue(sc_byte).extui(T.i32))
                    sc_f32    = ArithValue((sc_i32 << 23).bitcast(T.f32))
                    a_sc_cache.append(sc_f32)

                for rn in range_constexpr(wave_reg_n):
                    idx = rm * wave_reg_n + rn
                    new_vals = []
                    for si in range_constexpr(8):
                        r_val = s_accs[idx][si]
                        b_val = ArithValue(block_accs[idx][si])
                        new_vals.append(r_val + b_val * a_sc_cache[si])
                    new_s_accs.append(Vector.from_elements(new_vals, fx.Float32))

            results = yield new_s_accs

        running_accs = list(results)

        # Epilogue: apply scale_b and store fp16
        base8 = klane * 8
        sb_cache = []
        for rn in range_constexpr(wave_reg_n):
            g_col = tile_n0 + wave_n * (wave_reg_n * WMMA_N) + 16 * rn + lane16
            sb_cache.append(
                buffer_ops.buffer_load(scale_b_rsrc, g_col, vec_width=1, dtype=fx.Float32)
            )

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m * (wave_reg_m * WMMA_M) + 16 * rm
            for rn in range_constexpr(wave_reg_n):
                idx = rm * wave_reg_n + rn
                sb_val = sb_cache[rn]
                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + si
                    g_col = tile_n0 + wave_n * (wave_reg_n * WMMA_N) + 16 * rn + lane16
                    val = running_accs[idx][si] * sb_val
                    buffer_ops.buffer_store(val.to(fx.Float16), c_rsrc, g_row * N + g_col)

    @flyc.jit
    def launch_mxfp8_gemm(
        arg_c:       fx.Tensor,
        arg_a:       fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b:       fx.Tensor,
        arg_scale_b: fx.Tensor,
        stream:      fx.Stream,
        m:           fx.Int32,
    ):
        c1           = 1
        dyn_grid_m   = m // tile_m
        total_blocks = dyn_grid_m * grid_n
        launcher = kernel_mxfp8_gemm(arg_c, arg_a, arg_a_scale, arg_b, arg_scale_b, dyn_grid_m)
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_mxfp8_gemm


__all__ = [
    "compile_mxfp8_gemm",
    "mxfp8_quantize_a",
    "fp8_quantize_per_channel",
    "preshuffle_b_fp8",
]
