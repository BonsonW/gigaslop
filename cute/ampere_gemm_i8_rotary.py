"""Fused INT8 GEMM + rotary embedding for NVIDIA Ampere (A100, sm80).

CUDA INT8 port of fly/rdna_fp8_gemm_rotary.py.

  C[M, N] = rotary_embed(A[M,K] @ B[N,K]^T)

A (M,K) int8 K-major, B (N,K) int8 K-major (nn.Linear weight). Output fp16.
Scales: per-token scale_a[M], per-channel scale_b[N].

N = 3 * nhead * head_dim  (QKV concatenated):
  cols [0,                 nhead*head_dim): Q  — rotary applied
  cols [nhead*head_dim,  2*nhead*head_dim): K  — rotary applied
  cols [2*nhead*head_dim,                N): V  — passthrough

Rotary (matches openfish / the RDNA kernel):
  rotary_half = rotary_dim // 2
  for k in [0, rotary_half):
    x0 = head col k ; x1 = head col k+rotary_half
    out[k]            = x0*cos[k] - x1*sin[k]
    out[k+rotary_half] = x0*sin[k] + x1*cos[k]
sin/cos: [seqlen, rotary_half] fp32; row m uses seq = m % seqlen.

The rotary "companion" column (rotary_half away) is held by a different warp under
the (2,2,1) MMA layout, so the epilogue rounds the scaled tile through shared
memory. bN is pinned to head_dim so each N-tile is exactly one head and the
companion column is always within the tile.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from ampere_gemm_i8_quant_rmem import (
    TensorOpGemmI8,
    _install_local_mma_i8_op,
    create_and_permute_tensor,
)

_install_local_mma_i8_op()


class TensorOpGemmI8Rotary(TensorOpGemmI8):
    """INT8 GEMM with fused rotary-embedding epilogue. bN must equal head_dim."""

    def __init__(self, *args, nhead, head_dim, rotary_dim, seqlen, **kwargs):
        super().__init__(*args, **kwargs)
        self.nhead = nhead
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rotary_half = rotary_dim // 2
        self.seqlen = seqlen
        self.qk_cols = 2 * nhead * head_dim
        # bN must be a multiple of head_dim so each N-tile spans whole heads and the
        # rotary companion column (rotary_half away, within a head) stays in-tile.
        assert self.bN % head_dim == 0, f"bN({self.bN}) must be a multiple of head_dim({head_dim})"
        # tile must not straddle the Q/K|V boundary so Q/K-vs-V is uniform per tile
        assert self.qk_cols % self.bN == 0, "bN must divide 2*nhead*head_dim (Q/K region)"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mSin: cute.Tensor,
        mCos: cute.Tensor,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        smem_size = (
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout)
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type, num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits)

        op = cute.nvgpu.warp.MmaI8Op(
            self.a_dtype, self.b_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA, mB, mC, mScaleA, mScaleB, mSin, mCos,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma, raster_factor,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mSin: cute.Tensor,
        mCos: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            gA = cute.local_tile(mA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            gC = cute.local_tile(mC[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, 1, None))

            gScaleA = cute.local_tile(mScaleA[None, bidz], tiler=(self.bM,), coord=(offset_tile_x,))
            gScaleB = cute.local_tile(mScaleB[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))

            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(mcA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            cB = cute.local_tile(mcB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0)),
                cutlass.Boolean)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0)),
                cutlass.Boolean)
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()
            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_A, tAgA[None, None, k, k_tile_index],
                              tAsA[None, None, k, 0], pred=tApA[None, None, k])
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_B, tBgB[None, None, k, k_tile_index],
                              tBsB[None, None, k, 0], pred=tBpB[None, None, k])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                          tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                          tBsB[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0)

            num_vals = int(cute.size(tCrC, mode=[0]))
            num_mma_m = int(cute.size(tCrC, mode=[1]))
            num_mma_n = int(cute.size(tCrC, mode=[2]))

            gScaleA_2d = cute.make_tensor(gScaleA.iterator, cute.make_layout((self.bM, self.bN), stride=(1, 0)))
            gScaleB_2d = cute.make_tensor(gScaleB.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1)))
            tCgScaleA = thr_mma.partition_C(gScaleA_2d)
            tCgScaleB = thr_mma.partition_C(gScaleB_2d)
            rScaleA = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_m, 1, 0)), cutlass.Float32)
            rScaleB = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    rScaleA[i, m, 0] = tCgScaleA[i, m, 0].to(cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rScaleB[i, 0, n] = tCgScaleB[i, 0, n].to(cutlass.Float32)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mA.element_type)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mB.element_type)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                                      tAsA[None, None, None, smem_pipe_write], pred=tApA)

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                                      tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////////
            # Rotary epilogue, entirely in registers (no smem round-trip).
            #
            # With the (waves_m, waves_n) MMA layout each MMA_N block covers
            # cols_per_mma_n contiguous columns, so the rotary companion column
            # (rotary_half away, in the same head) is held by the SAME thread at
            # a compile-time-known MMA_N index. companion_step = rotary_half //
            # cols_per_mma_n. A block is entirely in the first or second rotary
            # half (compile-time), so the rotation formula is selected at compile
            # time and the only runtime work is the sin/cos table lookup.
            # ///////////////////////////////////////////////////////////////////
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((self.bM, self.bN)))
            cols_per_mma_n = self.bN // num_mma_n
            companion_step = self.rotary_half // cols_per_mma_n
            assert self.rotary_half % cols_per_mma_n == 0, \
                "rotary_half must be a multiple of cols_per_mma_n for register companion"

            tile_n0 = offset_tile_y * self.bN
            qk_f = cute.elem_less(
                cutlass.Int32(tile_n0), cutlass.Int32(self.qk_cols)
            ).to(cutlass.Float32)
            row0 = cutlass.Int32(offset_tile_x * self.bM)
            rh = self.rotary_half

            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            for i in cutlass.range_constexpr(num_vals):
                for m in cutlass.range_constexpr(num_mma_m):
                    for n in cutlass.range_constexpr(num_mma_n):
                        # compile-time: which rotary half this MMA_N block sits in
                        block_pos = (n * cols_per_mma_n) % self.head_dim
                        is_first = block_pos < rh

                        r = cutlass.Int32(tCcC[i, m, n][0])
                        c = cutlass.Int32(tCcC[i, m, n][1])
                        seq = (row0 + r) % cutlass.Int32(self.seqlen)
                        pos = c % cutlass.Int32(self.head_dim)
                        self_v = (tCrC[i, m, n].to(cutlass.Float32)
                                  * rScaleA[i, m, 0] * rScaleB[i, 0, n])

                        if cutlass.const_expr(is_first):   # self=x0, companion=x1
                            comp_n = n + companion_step
                            comp_v = (tCrC[i, m, comp_n].to(cutlass.Float32)
                                      * rScaleA[i, m, 0] * rScaleB[i, 0, comp_n])
                            rot_idx = pos
                            sin_v = mSin[seq, rot_idx].to(cutlass.Float32)
                            cos_v = mCos[seq, rot_idx].to(cutlass.Float32)
                            rotated = self_v * cos_v - comp_v * sin_v
                        else:                              # self=x1, companion=x0
                            comp_n = n - companion_step
                            comp_v = (tCrC[i, m, comp_n].to(cutlass.Float32)
                                      * rScaleA[i, m, 0] * rScaleB[i, 0, comp_n])
                            rot_idx = pos - cutlass.Int32(rh)
                            sin_v = mSin[seq, rot_idx].to(cutlass.Float32)
                            cos_v = mCos[seq, rot_idx].to(cutlass.Float32)
                            rotated = comp_v * sin_v + self_v * cos_v

                        # passthrough for V (and columns outside rotary span)
                        inrot_f = cute.elem_less(pos, cutlass.Int32(self.rotary_dim)).to(cutlass.Float32)
                        do_rot_f = qk_f * inrot_f
                        out_v = do_rot_f * rotated + (1.0 - do_rot_f) * self_v
                        tCrD[i, m, n] = out_v.to(self.c_dtype)
            cute.autovec_copy(tCrD, tCgC)

        return


# =============================================================================
# Host-side reference
# =============================================================================
def rotary_ref(A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf,
               nhead, head_dim, rotary_dim, seqlen):
    import torch
    M, K = A_int8.shape
    N = B_int8.shape[0]
    rh = rotary_dim // 2
    A_dq = A_int8.float() * scale_a[:, None]
    B_dq = B_int8.float() * scale_b[:, None]
    C = A_dq @ B_dq.T                      # (M, N)
    out = C.clone()
    qk_cols = 2 * nhead * head_dim
    seq = (torch.arange(M, device=C.device) % seqlen)
    sin = sin_buf[seq]                     # (M, rh)
    cos = cos_buf[seq]
    for head_start in range(0, qk_cols, head_dim):
        x0 = C[:, head_start:head_start + rh]
        x1 = C[:, head_start + rh:head_start + rotary_dim]
        out[:, head_start:head_start + rh]            = x0 * cos - x1 * sin
        out[:, head_start + rh:head_start + rotary_dim] = x0 * sin + x1 * cos
    return out


__all__ = ["TensorOpGemmI8Rotary", "rotary_ref", "create_and_permute_tensor"]
