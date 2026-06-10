import sys, os
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm_i8_quant_rmem import TensorOpGemmI8

def quantize_tensor(t, dim=-1):
    quant_max = 127
    fp_range = t.abs().amax(dim=dim).clamp_min(1e-8)
    quant_scale = quant_max / fp_range
    t_int8 = (t * quant_scale.unsqueeze(dim)).round().clamp(-quant_max, quant_max).to(torch.int8)
    dequant_scale = quant_scale.to(torch.float32).reciprocal()
    return t_int8, dequant_scale

M = 524288; K = 512; N = 4096; L = 1
A_int8, A_s = quantize_tensor(torch.randn(M, K))
B_int8, B_s = quantize_tensor(torch.randn(N, K))

a_dtype = cutlass.Int8; b_dtype = cutlass.Int8; c_dtype = cutlass.Float16; acc_dtype = cutlass.Int32
import ast
atom_layout_mnk = ast.literal_eval(os.environ.get("AL", "(2,2,1)"))
num_stages = int(os.environ.get("ST", "3"))
bm = int(os.environ.get("BM", "128")); bN = int(os.environ.get("BN", "128")); bK = 64
M_pad = ((M + bm - 1)//bm)*bm; N_pad = ((N + bN - 1)//bN)*bN; K_pad = ((K + bK - 1)//bK)*bK

def cpt(l, m0, m1, maj0, dt):
    shape = (l, m1, m0) if maj0 else (l, m0, m1)
    po = (2,1,0) if maj0 else (1,2,0)
    tt = torch.randint(-2,3, shape, dtype=cutlass_torch.dtype(dt)).permute(po).cuda()
    ct = (from_dlpack(tt, assumed_align=16).mark_layout_dynamic(leading_dim=(1 if not maj0 else 0))
          .mark_compact_shape_dynamic(mode=(1 if not maj0 else 0),
            stride_order=(2,0,1) if not maj0 else (2,1,0), divisibility=(128//dt.width)))
    return ct, tt

mA, a_torch = cpt(L, M_pad, K_pad, False, a_dtype)
mC, c_torch = cpt(L, M_pad, N_pad, False, c_dtype)
a_torch[:M,:K,0] = A_int8.cuda()

def preshuffle_B(b_int8, bN, bK, N_pad, K_pad):
    b = torch.zeros(N_pad, K_pad, dtype=torch.int8)
    b[:b_int8.shape[0],:b_int8.shape[1]] = b_int8
    b = b.reshape(N_pad//bN, bN, K_pad//bK, bK).permute(0,2,1,3).contiguous()
    return b.reshape(-1)

b_flat = preshuffle_B(B_int8, bN, bK, N_pad, K_pad)
_, b_torch = cpt(L, N_pad, K_pad, False, b_dtype)
b_torch.view(-1).copy_(b_flat); b_torch = b_torch.cuda()
mB = (from_dlpack(b_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
      .mark_compact_shape_dynamic(mode=1, stride_order=(2,0,1), divisibility=(128//b_dtype.width)))

sa = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); sa[:M,0]=A_s.cuda()
sb = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); sb[:N,0]=B_s.cuda()
mScaleA = from_dlpack(sa.contiguous(), assumed_align=16)
mScaleB = from_dlpack(sb.contiguous(), assumed_align=16)

g = TensorOpGemmI8(a_dtype, b_dtype, c_dtype, acc_dtype, atom_layout_mnk, True, bm, bn=bN, num_stages=num_stages)
compiled = cute.compile(g, mA, mB, mC, mScaleA, mScaleB, no_pred=True, b_preshuffled=True)
niter = int(os.environ.get("NITER", "3"))
for _ in range(niter):
    compiled(mA, mB, mC, mScaleA, mScaleB)
torch.cuda.synchronize()
print("done")
