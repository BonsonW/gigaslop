"""Fused FP8 low-rank ("factored") GEMM for RDNA4 (gfx1201, wave32).

Computes a chained two-GEMM input projection in ONE dispatch, keeping the rank-R
intermediate in LDS (no global round-trip):

  x_down = x_fp8[M,C] @ dn_fp8[C,R]^T            Phase 1 (K=C),  wmma fp8_fp8
         = (descale: x per-token scale × dn scalar scale) → f16 in LDS
  ih     = x_down_f16[M,R] @ up_f16[R,N]^T + bias[N]   Phase 2 (K=R), wmma f16
         → f16 output ih[M, N]

This is the factored-LSTM kernel without the gate/cell/tanh epilogue: it produces
the precomputed input contribution `ih` that the LSTM step adds to its gates.
`up_weight` columns are assumed pre-permuted to whatever output column order the
consumer wants (e.g. interleaved [.,H,4]); this kernel just writes ih[M,N] row-major.

Phase 2 uses f16 (x_down cast f32→f16 in LDS, up stored f16) — no fp8 requant; this
matches the proven factored-LSTM LDS path. up is tiny (R×N) so fp8 saves nothing.

Grid:  (M / tile_m, 1, 1)   — one block owns tile_m rows and loops over all N.
Block: (64 threads = 2 waves × 32 lanes)
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
WMMA_K = 16
WAVE_SIZE = 32
FP8_MAX = 448.0


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
    return B_r.permute(3, 0, 1, 4, 2).contiguous()


def fp8_quantize_per_token(x_f32):
    """Quantize f32 → fp8_e4m3fn with per-token scale. Returns (fp8, scale[M])."""
    import torch
    amax  = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    x_fp8 = (x_f32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def fp8_quantize_scalar(x_f32):
    """Quantize f32 → fp8_e4m3fn with a single scalar scale. Returns (fp8, scale[1])."""
    import torch
    amax  = x_f32.abs().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).reshape(1)
    x_fp8 = (x_f32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale


# =============================================================================
# Kernel compiler
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_factored_gemm(
    *,
    C: int,            # GEMM-1 contraction dim (input feature dim, e.g. 1024)
    R: int,            # rank / intermediate dim (e.g. 128)
    N: int,            # output dim (e.g. 4096)
    tile_m: int = 32,
    tile_k1: int = 32,
    k_unroll: int = 2,
    tile_n2: int = 128,  # Phase-2 N-tile width (128 best: fewer/wider N-tiles)
    tile_k2: int = 16,   # f16 WMMA K-size
    group_m: int = 8,
    n_split: int = 1,    # split the N output across this many blocks (more blocks at small M)
):
    """Fused fp8 down-proj + f16 up-proj GEMM → f16 output. M is runtime.

    n_split: at small M (few M-tiles) the kernel is latency-bound because one block
    serially walks all N. Splitting N across n_split blocks (grid = ceil(M/tile_m)*n_split)
    raises occupancy at the cost of recomputing Phase 1 (x@dn) n_split times per M-tile.
    """
    assert C % tile_k1 == 0
    assert R % WMMA_N == 0 and R % tile_k2 == 0
    assert N % tile_n2 == 0
    assert tile_m % WMMA_M == 0
    assert tile_n2 % WMMA_N == 0
    assert (N // tile_n2) % n_split == 0, "n_split must divide num_n_tiles"

    # ── Phase 1 tile dims (x[M,C] @ dn[C,R] → x_down[M,R]) ────────────────────
    tile_n_y   = R
    reg_m      = tile_m   // WMMA_M
    reg_n_y    = tile_n_y // WMMA_N
    reg_k1     = tile_k1  // WMMA_K

    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m        = reg_m   // waves_m
    wave_reg_n_y      = reg_n_y // waves_n

    num_k1_tiles = C // tile_k1

    # ── Phase 2 tile dims (x_down[M,R] @ up[R,N] → ih[M,N]) ───────────────────
    reg_n2       = tile_n2 // WMMA_N
    wave_reg_n2  = reg_n2 // waves_n
    num_k2_tiles = R // tile_k2
    num_n_tiles  = N // tile_n2
    nt_per_block = num_n_tiles // n_split   # N-tiles each block computes

    # ── Phase 1 B-strides: dn fp8 preshuffled [R//16, C//16, 2, 16, 8] ───────
    K0_C            = C // 16
    B1_KPACK        = 8
    B1_STRIDE_NLANE = B1_KPACK
    B1_STRIDE_KLANE = 16 * B1_KPACK       # 128
    B1_STRIDE_K0    = 2 * 16 * B1_KPACK   # 256
    B1_STRIDE_N0    = K0_C * B1_STRIDE_K0

    # ── Phase 2 B-strides: up f16 preshuffled [N//16, R//16, 2, 16, 8] ───────
    K0_R            = R // 16
    B2_STRIDE_NLANE = 8
    B2_STRIDE_KLANE = 16 * 8              # 128 f16
    B2_STRIDE_K0    = 2 * 16 * 8          # 256 f16
    B2_STRIDE_N0    = K0_R * B2_STRIDE_K0

    # ── LDS: x_down f16 [tile_m, R] ──────────────────────────────────────────
    lds_alloc    = SmemAllocator(None, global_sym_name="smem_factored_gemm")
    f16_byte_off = lds_alloc._align(lds_alloc.ptr, 32)
    f16_size     = tile_m * R * 2
    lds_alloc.ptr = f16_byte_off + f16_size

    @flyc.kernel
    def kernel_factored_gemm(
        arg_ih_out:    fx.Tensor,   # [M, N]   f16  — output
        arg_x:         fx.Tensor,   # [M, C]   fp8  — input
        arg_scale_x:   fx.Tensor,   # [M]      f32  — per-token scale for x
        arg_dn_weight: fx.Tensor,   # preshuffled fp8 dn [C, R]
        arg_scale_dn:  fx.Tensor,   # [1]      f32  — scalar scale for dn
        arg_up_weight: fx.Tensor,   # preshuffled f16 up [R, N]
        arg_bias:      fx.Tensor,   # [N]      f32
        arg_m:         fx.Int32,    # actual row count M (may not be a multiple of tile_m)
    ):
        tid     = gpu.thread_id("x")
        pid     = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        # Grid = ceil(M/tile_m) * n_split blocks. pid (index) decodes to (bid_m, bid_n):
        #   bid_m = pid // n_split   → which M-tile (rows)
        #   bid_n = pid %  n_split   → which N-chunk (output columns)
        bid_m   = pid // n_split
        bid_n   = pid - bid_m * n_split         # pid % n_split  (index)
        m_i32   = arg_m                         # for tail masking when M % tile_m != 0

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        tile_m0 = bid_m * tile_m
        nt0     = bid_n * nt_per_block          # first N-tile this block owns (index)

        x_rsrc    = buffer_ops.create_buffer_resource(arg_x,         max_size=True)
        dn_rsrc   = buffer_ops.create_buffer_resource(arg_dn_weight, max_size=True)
        up_rsrc   = buffer_ops.create_buffer_resource(arg_up_weight, max_size=True)
        # Output resource base-offset to this block's first row, so the per-thread
        # buffer offset (local_row*N + col) stays within i32. Without this, the
        # absolute offset g_row*N overflows i32 once M*N approaches 2^31 (e.g. M>262144
        # at N=4096) → corrupt writes. base_byte_offset is 64-bit pointer arithmetic.
        ih_base_bytes = tile_m0 * (N * 2)   # f16 output = 2 bytes/elem
        ih_rsrc   = buffer_ops.create_buffer_resource(
            arg_ih_out, max_size=True, base_byte_offset=ih_base_bytes)
        sx_rsrc   = buffer_ops.create_buffer_resource(arg_scale_x,   max_size=True)
        sdn_rsrc  = buffer_ops.create_buffer_resource(arg_scale_dn,  max_size=True)
        bias_rsrc = buffer_ops.create_buffer_resource(arg_bias,      max_size=True)

        base_ptr = lds_alloc.get_base()
        s_y_f16  = SmemPtr(base_ptr, f16_byte_off, fx.T.f16(), shape=(tile_m * R,))
        s_y_f16.get()

        # ── Phase 1: x_fp8 @ dn_fp8 → x_down (fp8×fp8), K-loop over C ─────────
        def _load_a1(kt):
            vecs = []
            for rk in range_constexpr(reg_k1):
                rv = []
                col = kt * tile_k1 + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row      = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    byte_off = row * C + col
                    rv.append(buffer_ops.buffer_load(x_rsrc, byte_off // 4,
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

        # ── Phase 1.5: descale (x per-token × dn scalar) → f16 LDS ───────────
        wave_n_r0  = wave_n * (wave_reg_n_y * WMMA_N)
        wave_m_off = wave_m * (wave_reg_m  * WMMA_M)
        base8      = klane * 8

        scale_dn = ArithValue(buffer_ops.buffer_load(
            sdn_rsrc, 0, vec_width=1, dtype=fx.Float32))

        sx_all = []
        for rm in range_constexpr(wave_reg_m):
            sx_rm = []
            for si in range_constexpr(8):
                m_row = tile_m0 + wave_m_off + 16 * rm + base8 + si
                sx_rm.append(ArithValue(
                    buffer_ops.buffer_load(sx_rsrc, m_row, vec_width=1, dtype=fx.Float32)))
            sx_all.append(sx_rm)

        for rm in range_constexpr(wave_reg_m):
            for rn in range_constexpr(wave_reg_n_y):
                idx   = rm * wave_reg_n_y + rn
                r_col = wave_n_r0 + 16 * rn + lane16
                for si in range_constexpr(8):
                    row_in_tile = wave_m_off + 16 * rm + base8 + si
                    lin_idx     = row_in_tile * R + r_col
                    y_f32 = ArithValue(accs1[idx][si]) * sx_all[rm][si] * scale_dn
                    SmemPtr.store(s_y_f16, arith.truncf(fx.T.f16(), y_f32), [lin_idx])

        gpu.barrier()

        # ── Phase 2: x_down_f16 @ up_f16 → ih, looped over all N tiles ───────
        v8f16_ty = ir.VectorType.get([8], ir.F16Type.get())

        def _load_a2(kt):
            frags = []
            for rm in range_constexpr(wave_reg_m):
                row_in_tile = wave_m_off + 16 * rm + lane16
                k_elem      = kt * WMMA_K + klane * 8
                f16_off     = row_in_tile * R + k_elem
                frags.append(mlir_vector.load(v8f16_ty, s_y_f16.get(), [f16_off]))
            return frags

        def _load_b2(kt, n0_base):
            vecs = []
            for rn in range_constexpr(wave_reg_n2):
                n0      = n0_base + wave_n * wave_reg_n2 + rn
                f16_off = (n0 * B2_STRIDE_N0 + kt * B2_STRIDE_K0
                           + klane * B2_STRIDE_KLANE + lane16 * B2_STRIDE_NLANE)
                vecs.append(buffer_ops.buffer_load(up_rsrc, f16_off,
                                                   vec_width=8, dtype=fx.Float16))
            return vecs

        def _compute2(acc, a_v, b_v):
            new = list(acc)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n2):
                    idx = rm * wave_reg_n2 + rn
                    new[idx] = rocdl.wmma_f32_16x16x16_f16(
                        new[idx].type, a_v[rm], b_v[rn], new[idx]).result
            return new

        n_ac2 = wave_reg_m * wave_reg_n2

        for nt_local in range_constexpr(nt_per_block):
            nt        = nt0 + nt_local           # absolute N-tile index (index-typed)
            n0_base   = nt * (tile_n2 // 16)
            tile_n2_0 = nt * tile_n2

            acc2 = [zero_acc for _ in range_constexpr(n_ac2)]
            for kt2 in range_constexpr(num_k2_tiles):
                a2 = _load_a2(kt2)
                b2 = _load_b2(kt2, n0_base)
                acc2 = _compute2(acc2, a2, b2)

            wave_n2_0 = tile_n2_0 + wave_n * (wave_reg_n2 * WMMA_N)

            # bias cache per rn
            bias_c = []
            for rn in range_constexpr(wave_reg_n2):
                col = wave_n2_0 + 16 * rn + lane16
                bias_c.append(ArithValue(buffer_ops.buffer_load(
                    bias_rsrc, col, vec_width=1, dtype=fx.Float32)))

            for rm in range_constexpr(wave_reg_m):
                wmma_m_off = wave_m_off + 16 * rm
                for rn in range_constexpr(wave_reg_n2):
                    idx   = rm * wave_reg_n2 + rn
                    col   = wave_n2_0 + 16 * rn + lane16
                    for si in range_constexpr(8):
                        local_row = wmma_m_off + base8 + si   # 0..tile_m-1 (block-relative)
                        g_row     = tile_m0 + local_row
                        val   = ArithValue(acc2[idx][si]) + bias_c[rn]
                        # Mask the tail: skip rows >= M (handles M % tile_m != 0).
                        g_row_i32 = fx.arith.index_cast(fx.T.i32(), g_row)
                        if g_row_i32 < m_i32:
                            # ih_rsrc is base-offset to tile_m0, so use the block-relative
                            # row → offset stays small (no i32 overflow at large M).
                            buffer_ops.buffer_store(
                                arith.truncf(fx.T.f16(), val), ih_rsrc, local_row * N + col)

    # ── Host launcher ─────────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_factored_gemm(
        arg_ih_out:    fx.Tensor,
        arg_x:         fx.Tensor,
        arg_scale_x:   fx.Tensor,
        arg_dn_weight: fx.Tensor,
        arg_scale_dn:  fx.Tensor,
        arg_up_weight: fx.Tensor,
        arg_bias:      fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1           = 1
        # ceil(M / tile_m): cover the tail tile; the kernel masks rows >= M.
        dyn_grid_m   = (m + (tile_m - 1)) // tile_m
        total_blocks = dyn_grid_m * n_split
        launcher = kernel_factored_gemm(
            arg_ih_out, arg_x, arg_scale_x, arg_dn_weight, arg_scale_dn,
            arg_up_weight, arg_bias, m,
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

    return launch_fp8_factored_gemm


# =============================================================================
# Down-projection only: x_down = x @ dn  (the precompute for the fused LSTM).
# =============================================================================

@functools.lru_cache(maxsize=64)
def compile_fp8_down_proj(
    *,
    C: int,            # contraction (input feature dim, e.g. 1024)
    R: int,            # output rank (e.g. 128)
    tile_m: int = 32,
    tile_k1: int = 32,
    k_unroll: int = 2,
    n_split: int = 1,  # split R output across this many blocks (more blocks at small M)
):
    """x_down[M,R] f16 = (x[M,C]_fp8 @ dn[C,R]_fp8) * scale_x[M] * scale_dn.
    M runtime; output written directly to global (no up-projection). Feeds the
    fused LSTM's x_down input.

    n_split: at small M (few M-tiles) the kernel is latency-bound — only ceil(M/tile_m)
    blocks, each serially walking the full K=C reduction. Splitting the R output across
    n_split blocks (grid = ceil(M/tile_m)*n_split) raises occupancy. Each block re-reads
    the same x rows (L2-resident, cheap) but computes a distinct R-chunk's columns."""
    assert C % tile_k1 == 0
    assert R % WMMA_N == 0
    assert tile_m % WMMA_M == 0
    assert R % n_split == 0, "n_split must divide R"
    R_chunk = R // n_split
    assert R_chunk % WMMA_N == 0, "R/n_split must be a multiple of WMMA_N"

    reg_m   = tile_m // WMMA_M
    reg_n_y = R_chunk // WMMA_N          # N-tiles per BLOCK (one R-chunk)
    reg_k1  = tile_k1 // WMMA_K
    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m   = reg_m // waves_m
    wave_reg_n_y = reg_n_y // waves_n
    assert wave_reg_n_y >= 1, "R/n_split too small for waves_n=2 (need R/n_split >= 32)"
    R0_chunk     = R_chunk // 16         # N0-blocks per chunk (for dn offset + r_col)
    num_k1_tiles = C // tile_k1

    K0_C            = C // 16
    B1_STRIDE_NLANE = 8
    B1_STRIDE_KLANE = 16 * 8
    B1_STRIDE_K0    = 2 * 16 * 8
    B1_STRIDE_N0    = K0_C * B1_STRIDE_K0

    @flyc.kernel
    def kernel_down_proj(
        arg_x_down:    fx.Tensor,   # [M, R]  f16 — output
        arg_x:         fx.Tensor,   # [M, C]  fp8
        arg_scale_x:   fx.Tensor,   # [M]     f32  per-token
        arg_dn_weight: fx.Tensor,   # preshuffled fp8 dn [C, R]  ([R//16,C//16,2,16,8])
        arg_scale_dn:  fx.Tensor,   # [1]     f32  scalar
        arg_m:         fx.Int32,
    ):
        tid    = gpu.thread_id("x")
        pid    = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        # Grid = ceil(M/tile_m) * n_split. pid decodes to (bid_m, bid_n):
        #   bid_m = pid // n_split  → which M-tile (rows)
        #   bid_n = pid %  n_split  → which R-chunk (output columns)
        bid_m  = pid // n_split
        bid_n  = pid - bid_m * n_split
        m_i32  = arg_m
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        tile_m0 = bid_m * tile_m

        x_rsrc   = buffer_ops.create_buffer_resource(arg_x,         max_size=True)
        dn_rsrc  = buffer_ops.create_buffer_resource(arg_dn_weight, max_size=True)
        sx_rsrc  = buffer_ops.create_buffer_resource(arg_scale_x,   max_size=True)
        sdn_rsrc = buffer_ops.create_buffer_resource(arg_scale_dn,  max_size=True)
        xd_base  = tile_m0 * (R * 2)   # base-offset (f16) → per-thread offset stays in i32
        xd_rsrc  = buffer_ops.create_buffer_resource(
            arg_x_down, max_size=True, base_byte_offset=xd_base)

        def _load_a1(kt):
            vecs = []
            for rk in range_constexpr(reg_k1):
                rv = []
                col = kt * tile_k1 + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    rv.append(buffer_ops.buffer_load(x_rsrc, (row * C + col) // 4,
                                                     vec_width=2, dtype=fx.Int32))
                vecs.append(rv)
            return vecs

        def _load_b1(kt):
            vecs = []
            n0_base = bid_n * R0_chunk + wave_n * wave_reg_n_y
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

        zero_acc = fx.full(8, 0.0, fx.Float32)
        accs1    = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n_y)]
        a1_cur   = _load_a1(0); b1_cur = _load_b1(0)
        full_out = (num_k1_tiles - 1) // k_unroll
        rem      = (num_k1_tiles - 1) % k_unroll

        def _flat(t):
            f = []
            for r in t: f.extend(r)
            return f
        def _unflat_a1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_m): r.append(f[i]); i += 1
                out.append(r)
            return out
        def _unflat_b1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_n_y): r.append(f[i]); i += 1
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
                    nkt = iv + j + 1
                    a_nxt = _load_a1(nkt); b_nxt = _load_b1(nkt)
                    s_ac  = _compute1(s_ac, s_a, s_b)
                    s_a   = _unflat_a1(_flat(a_nxt)); s_b = _unflat_b1(_flat(b_nxt))
                res = yield _flat(s_a) + list(s_ac) + _flat(s_b)
            a1_cur = _unflat_a1(list(res[:n_a1]))
            accs1  = list(res[n_a1 : n_a1 + n_ac1])
            b1_cur = _unflat_b1(list(res[n_a1 + n_ac1:]))

        if const_expr(rem > 0):
            for j in range_constexpr(rem):
                nkt = full_out * k_unroll + j + 1
                a_nxt = _load_a1(nkt); b_nxt = _load_b1(nkt)
                accs1 = _compute1(accs1, a1_cur, b1_cur)
                a1_cur = _unflat_a1(_flat(a_nxt)); b1_cur = _unflat_b1(_flat(b_nxt))

        accs1 = _compute1(accs1, a1_cur, b1_cur)

        # descale + store x_down[M,R] f16 to global (tail-masked)
        wave_n_r0  = wave_n * (wave_reg_n_y * WMMA_N)
        wave_m_off = wave_m * (wave_reg_m  * WMMA_M)
        base8      = klane * 8
        scale_dn = ArithValue(buffer_ops.buffer_load(sdn_rsrc, 0, vec_width=1, dtype=fx.Float32))
        sx_all = []
        for rm in range_constexpr(wave_reg_m):
            sx_rm = []
            for si in range_constexpr(8):
                m_row = tile_m0 + wave_m_off + 16 * rm + base8 + si
                sx_rm.append(ArithValue(buffer_ops.buffer_load(sx_rsrc, m_row, vec_width=1, dtype=fx.Float32)))
            sx_all.append(sx_rm)

        for rm in range_constexpr(wave_reg_m):
            for rn in range_constexpr(wave_reg_n_y):
                idx   = rm * wave_reg_n_y + rn
                r_col = bid_n * R_chunk + wave_n_r0 + 16 * rn + lane16
                for si in range_constexpr(8):
                    local_row = wave_m_off + 16 * rm + base8 + si
                    g_row     = tile_m0 + local_row
                    y_f32 = ArithValue(accs1[idx][si]) * sx_all[rm][si] * scale_dn
                    g_row_i32 = fx.arith.index_cast(fx.T.i32(), g_row)
                    if g_row_i32 < m_i32:
                        buffer_ops.buffer_store(arith.truncf(fx.T.f16(), y_f32),
                                                xd_rsrc, local_row * R + r_col)

    @flyc.jit
    def launch_fp8_down_proj(
        arg_x_down:    fx.Tensor,
        arg_x:         fx.Tensor,
        arg_scale_x:   fx.Tensor,
        arg_dn_weight: fx.Tensor,
        arg_scale_dn:  fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1 = 1
        dyn_grid_m = (m + (tile_m - 1)) // tile_m
        total_blocks = dyn_grid_m * n_split
        launcher = kernel_down_proj(arg_x_down, arg_x, arg_scale_x, arg_dn_weight, arg_scale_dn, m)
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_fp8_down_proj


# =============================================================================
# f16 down-projection: x_down = x @ dn, all f16 — NO quantization.
# Replaces (fp8 quantize + fp8 down_proj) for the pre-loop precompute. The precompute
# is bandwidth-bound, so f16 WMMA costs nothing vs fp8, and we save the entire quantize
# pass + the fp8 activation write/re-read (~half the traffic). x_down is more accurate.
# =============================================================================
def compile_f16_down_proj(
    *,
    C: int,            # contraction (input feature dim, e.g. 1024)
    R: int,            # output rank (e.g. 128)
    tile_m: int = 32,
    tile_k1: int = 32,
    k_unroll: int = 2,
    n_split: int = 1,
):
    """x_down[M,R] f16 = x[M,C]_f16 @ dn[C,R]_f16. No quantize, no scales. M runtime."""
    assert C % tile_k1 == 0
    assert R % WMMA_N == 0
    assert tile_m % WMMA_M == 0
    assert R % n_split == 0, "n_split must divide R"
    R_chunk = R // n_split
    assert R_chunk % WMMA_N == 0, "R/n_split must be a multiple of WMMA_N"

    reg_m   = tile_m // WMMA_M
    reg_n_y = R_chunk // WMMA_N
    reg_k1  = tile_k1 // WMMA_K
    waves_m, waves_n  = 1, 2
    NUM_WAVES         = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m   = reg_m // waves_m
    wave_reg_n_y = reg_n_y // waves_n
    assert wave_reg_n_y >= 1, "R/n_split too small for waves_n=2 (need R/n_split >= 32)"
    R0_chunk     = R_chunk // 16
    num_k1_tiles = C // tile_k1

    # dn f16 preshuffled [R//16, C//16, 2, 16, 8] — strides in f16 ELEMENTS.
    K0_C            = C // 16
    B1_STRIDE_NLANE = 8
    B1_STRIDE_KLANE = 16 * 8
    B1_STRIDE_K0    = 2 * 16 * 8
    B1_STRIDE_N0    = K0_C * B1_STRIDE_K0

    @flyc.kernel
    def kernel_f16_down_proj(
        arg_x_down:    fx.Tensor,   # [M, R]  f16 — output
        arg_x:         fx.Tensor,   # [M, C]  f16
        arg_dn_weight: fx.Tensor,   # preshuffled f16 dn [R//16, C//16, 2, 16, 8]
        arg_m:         fx.Int32,
    ):
        tid    = gpu.thread_id("x")
        pid    = gpu.block_id("x")
        wave_id = tid // 32
        lane    = tid % 32
        lane16  = lane % 16
        klane   = lane // 16

        bid_m  = pid // n_split
        bid_n  = pid - bid_m * n_split
        m_i32  = arg_m
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        tile_m0 = bid_m * tile_m

        x_rsrc   = buffer_ops.create_buffer_resource(arg_x,         max_size=True)
        dn_rsrc  = buffer_ops.create_buffer_resource(arg_dn_weight, max_size=True)
        xd_base  = tile_m0 * (R * 2)
        xd_rsrc  = buffer_ops.create_buffer_resource(
            arg_x_down, max_size=True, base_byte_offset=xd_base)

        def _load_a1(kt):
            vecs = []
            for rk in range_constexpr(reg_k1):
                rv = []
                col = kt * tile_k1 + 16 * rk + klane * 8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * (wave_reg_m * WMMA_M) + 16 * rm + lane16
                    rv.append(buffer_ops.buffer_load(x_rsrc, row * C + col,
                                                     vec_width=8, dtype=fx.Float16))
                vecs.append(rv)
            return vecs

        def _load_b1(kt):
            vecs = []
            n0_base = bid_n * R0_chunk + wave_n * wave_reg_n_y
            for rk in range_constexpr(reg_k1):
                rv = []
                k0 = kt * reg_k1 + rk
                for rn in range_constexpr(wave_reg_n_y):
                    n0 = n0_base + rn
                    f16_off = (n0 * B1_STRIDE_N0 + k0 * B1_STRIDE_K0
                               + klane * B1_STRIDE_KLANE + lane16 * B1_STRIDE_NLANE)
                    rv.append(buffer_ops.buffer_load(dn_rsrc, f16_off,
                                                     vec_width=8, dtype=fx.Float16))
                vecs.append(rv)
            return vecs

        def _compute1(acc, a, b):
            new = list(acc)
            for rk in range_constexpr(reg_k1):
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n_y):
                        idx = rm * wave_reg_n_y + rn
                        new[idx] = rocdl.wmma_f32_16x16x16_f16(
                            new[idx].type, a[rk][rm], b[rk][rn], new[idx]).result
            return new

        zero_acc = fx.full(8, 0.0, fx.Float32)
        accs1    = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n_y)]
        a1_cur   = _load_a1(0); b1_cur = _load_b1(0)
        full_out = (num_k1_tiles - 1) // k_unroll
        rem      = (num_k1_tiles - 1) % k_unroll

        def _flat(t):
            f = []
            for r in t: f.extend(r)
            return f
        def _unflat_a1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_m): r.append(f[i]); i += 1
                out.append(r)
            return out
        def _unflat_b1(f):
            out, i = [], 0
            for _ in range_constexpr(reg_k1):
                r = []
                for _ in range_constexpr(wave_reg_n_y): r.append(f[i]); i += 1
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
                    nkt = iv + j + 1
                    a_nxt = _load_a1(nkt); b_nxt = _load_b1(nkt)
                    s_ac  = _compute1(s_ac, s_a, s_b)
                    s_a   = _unflat_a1(_flat(a_nxt)); s_b = _unflat_b1(_flat(b_nxt))
                res = yield _flat(s_a) + list(s_ac) + _flat(s_b)
            a1_cur = _unflat_a1(list(res[:n_a1]))
            accs1  = list(res[n_a1 : n_a1 + n_ac1])
            b1_cur = _unflat_b1(list(res[n_a1 + n_ac1:]))

        if const_expr(rem > 0):
            for j in range_constexpr(rem):
                nkt = full_out * k_unroll + j + 1
                a_nxt = _load_a1(nkt); b_nxt = _load_b1(nkt)
                accs1 = _compute1(accs1, a1_cur, b1_cur)
                a1_cur = _unflat_a1(_flat(a_nxt)); b1_cur = _unflat_b1(_flat(b_nxt))

        accs1 = _compute1(accs1, a1_cur, b1_cur)

        # store x_down[M,R] f16 to global (no descale; tail-masked)
        wave_n_r0  = wave_n * (wave_reg_n_y * WMMA_N)
        wave_m_off = wave_m * (wave_reg_m  * WMMA_M)
        base8      = klane * 8
        for rm in range_constexpr(wave_reg_m):
            for rn in range_constexpr(wave_reg_n_y):
                idx   = rm * wave_reg_n_y + rn
                r_col = bid_n * R_chunk + wave_n_r0 + 16 * rn + lane16
                for si in range_constexpr(8):
                    local_row = wave_m_off + 16 * rm + base8 + si
                    g_row     = tile_m0 + local_row
                    g_row_i32 = fx.arith.index_cast(fx.T.i32(), g_row)
                    if g_row_i32 < m_i32:
                        buffer_ops.buffer_store(arith.truncf(fx.T.f16(), ArithValue(accs1[idx][si])),
                                                xd_rsrc, local_row * R + r_col)

    @flyc.jit
    def launch_f16_down_proj(
        arg_x_down:    fx.Tensor,
        arg_x:         fx.Tensor,
        arg_dn_weight: fx.Tensor,
        stream:        fx.Stream,
        m:             fx.Int32,
    ):
        c1 = 1
        dyn_grid_m = (m + (tile_m - 1)) // tile_m
        total_blocks = dyn_grid_m * n_split
        launcher = kernel_f16_down_proj(arg_x_down, arg_x, arg_dn_weight, m)
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_f16_down_proj


__all__ = [
    "compile_fp8_factored_gemm",
    "compile_fp8_down_proj",
    "compile_f16_down_proj",
    "preshuffle_b_fp8",
    "preshuffle_b_f16",
    "fp8_quantize_per_token",
    "fp8_quantize_scalar",
]
