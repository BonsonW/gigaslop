"""Verify wmma_f32_16x16x16_f16 gives correct results on gfx1201.

Fragment layout (wave32), confirmed from working fp8 LSTM kernel:
  Thread T (klane = T//16, lane16 = T%16):
    A fragment: A[lane16, klane*8 .. klane*8+7]   (8 f16 from M-row lane16)
    B fragment: B[klane*8 .. klane*8+7, lane16]   (8 f16 from N-col lane16)
    C output:   C[klane*8+si, lane16] for si=0..7

buffer_load with dtype=Float16 uses f16 element units for offset.
B preshuffle [K,N] → [N0, K0, KLane=2, NLane=16, KPack=8], same as fp8.
B element strides: NLANE=8, KLANE=16*8=128, K0=2*128=256.
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import rocdl, buffer_ops, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

M = 16; K = 16; N = 16

B_STRIDE_NLANE  = 8          # 8 f16 per NLane slot
B_STRIDE_KLANE  = N * B_STRIDE_NLANE    # 128 f16
B_STRIDE_K0     = 2 * B_STRIDE_KLANE   # 256 f16


def preshuffle_b_f16(B_kn: torch.Tensor) -> torch.Tensor:
    """B[K,N] → [N0, K0, KLane=2, NLane=16, KPack=8], identical reshape to fp8."""
    K_, N_ = B_kn.shape
    assert K_ % 16 == 0 and N_ % 16 == 0
    B_r = B_kn.reshape(K_ // 16, 2, 8, N_ // 16, 16)
    return B_r.permute(3, 0, 1, 4, 2).contiguous()


@flyc.kernel
def kern_f16_gemm(
    arg_c_out:  fx.Tensor,   # [M=16, N=16] f32
    arg_a:      fx.Tensor,   # [M=16, K=16] f16 row-major
    arg_b_shuf: fx.Tensor,   # preshuffled f16, shape [1,1,2,16,8]
):
    lane   = gpu.thread_id("x") % 32
    klane  = lane // 16
    lane16 = lane % 16

    klane_i32  = ArithValue(fx.arith.index_cast(T.i32, klane))
    lane16_i32 = ArithValue(fx.arith.index_cast(T.i32, lane16))

    rsrc_a = buffer_ops.create_buffer_resource(arg_a,      max_size=True)
    rsrc_b = buffer_ops.create_buffer_resource(arg_b_shuf, max_size=True)
    rsrc_c = buffer_ops.create_buffer_resource(arg_c_out,  max_size=True)

    c8f32 = fx.full(8, 0.0, fx.Float32)

    # Load A: row=lane16, cols klane*8..klane*8+7  (f16 element offset)
    a_off = ArithValue(lane16_i32 * K + klane_i32 * 8)
    a_vec = buffer_ops.buffer_load(rsrc_a, a_off, vec_width=8, dtype=fx.Float16)

    # Load B: rows klane*8..klane*8+7, col=lane16  (f16 element offset)
    b_off = ArithValue(klane_i32 * B_STRIDE_KLANE + lane16_i32 * B_STRIDE_NLANE)
    b_vec = buffer_ops.buffer_load(rsrc_b, b_off, vec_width=8, dtype=fx.Float16)

    # WMMA: A/B both vector<8xf16>, C/output vector<8xf32>
    result = rocdl.wmma_f32_16x16x16_f16(c8f32.type, a_vec, b_vec, c8f32).result

    # Write C[klane*8+si, lane16] = result[si]  (row = klane*8+si, col = lane16)
    c_col      = lane16_i32
    c_row_base = ArithValue(klane_i32 * 8)
    for si in range_constexpr(8):
        val   = ArithValue(mlir_vector.extract(result, static_position=[si], dynamic_position=[]))
        c_off = ArithValue(c_row_base + si) * N + c_col
        buffer_ops.buffer_store(val, rsrc_c, c_off)


@flyc.jit
def launch_f16_gemm(c_out, a, b_shuf, stream: fx.Stream):
    kern_f16_gemm(c_out, a, b_shuf).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


if __name__ == "__main__":
    device = "cuda"
    stream = torch.cuda.current_stream()
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    C_ref = A.float() @ B.float()

    B_shuf = preshuffle_b_f16(B)
    C_out  = torch.zeros(M, N, dtype=torch.float32, device=device)

    launch_f16_gemm(C_out, A, B_shuf, stream)
    torch.cuda.synchronize()

    err = (C_out - C_ref).abs().max().item()
    rel = err / C_ref.abs().max().item()
    print(f"max abs err: {err:.4f}  rel: {rel:.4f}")
    print("f16 WMMA:", "PASS" if rel < 0.01 else "FAIL")
    if rel >= 0.01:
        print("C_out row 0:", C_out[0].tolist())
        print("C_ref row 0:", C_ref[0].tolist())
        print("C_out row 8:", C_out[8].tolist())
        print("C_ref row 8:", C_ref[8].tolist())
