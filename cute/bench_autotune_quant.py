import sys
import os
import multiprocessing as mp

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tutorial'))
from ampere_gemm_i8_quant import TensorOpGemmI8

# Simple test params - use larger size to match tile boundaries
batch_size = 512
timestep = 1024
out_features = 512
in_features = 512

def quantize_tensor(t, dim=-1):
    """Quantize float tensor to int8 with optional per-tensor scaling.

    Uses symmetric int8 range [-127, 127]. Handles zero ranges safely by
    clamping the FP range to a small epsilon.
    """
    levels = 256
    quant_max = (levels // 2) - 1  # 127

    fp_range = t.abs().amax(dim=dim)
    fp_range = fp_range.clamp_min(1e-8)
    quant_scale = quant_max / fp_range  # 127 / fp_range

    t_int8 = (t * quant_scale.unsqueeze(dim)).round().clamp(-quant_max, quant_max).to(torch.int8)
    dequant_scale = quant_scale.to(torch.float32).reciprocal()

    return t_int8, dequant_scale


def candidate_atom_layouts_for_bm(bm, mma_m=16, mma_n=8):
    """Return legal atom_layout_mnk candidates for the current CTA tile size."""
    atom_m_values = [1, 2, 4, 8, 16, 32]
    atom_n_values = [1, 2, 4, 8, 16, 32]

    legal_atom_m_values = [
        atom_m for atom_m in atom_m_values if atom_m * mma_m <= bm and bm % (atom_m * mma_m) == 0
    ]

    return [
        (atom_m, atom_n, 1)
        for atom_m in legal_atom_m_values
        for atom_n in atom_n_values
    ]


def autotune_int8_gemm(
    mA,
    mB,
    mC,
    mAScale,
    mBScale,
    bm_values=(32, 64, 128, 256),
    use_k32_values=(False, True),
    warmup_iterations=5,
    iterations=100,
):
    """Benchmark legal INT8 GEMM configurations and return the fastest one."""
    best_result = None
    all_results = []

    for bm in bm_values:
        for use_k32 in use_k32_values:
            for atom_layout_mnk in candidate_atom_layouts_for_bm(
                bm, mma_m=16, mma_n=8
            ):
                result = run_candidate_in_subprocess(
                    bm,
                    use_k32,
                    atom_layout_mnk,
                    warmup_iterations,
                    iterations,
                )
                if result is None:
                    continue

                all_results.append(result)

                print(
                    f"bm={bm}, use_k32={use_k32}, atom_layout_mnk={atom_layout_mnk} -> {result['avg_time_us']:.4f} us"
                )

                if best_result is None or result["avg_time_us"] < best_result["avg_time_us"]:
                    best_result = result

    if best_result is None:
        raise RuntimeError("No legal INT8 GEMM configuration could be compiled")

    print(
        "\nBest config: "
        f"bm={best_result['bm']}, use_k32={best_result['use_k32']}, "
        f"atom_layout_mnk={best_result['atom_layout_mnk']}, "
        f"time={best_result['avg_time_us']:.4f} us"
    )

    return best_result, all_results


def _candidate_worker(queue, bm, use_k32, atom_layout_mnk, warmup_iterations, iterations):
    try:
        A_float = torch.randn((batch_size, timestep, in_features), dtype=torch.float16, device='cuda').reshape(batch_size * timestep, in_features)
        B_float = torch.randn((out_features, in_features), dtype=torch.float16, device='cuda').reshape(out_features, in_features)
        A_int8, A_scale = quantize_tensor(A_float, dim=-1)
        B_int8, B_scale = quantize_tensor(B_float, dim=-1)

        A_int8 = A_int8.reshape(batch_size * timestep, in_features, 1)
        B_int8 = B_int8.reshape(out_features, in_features, 1)
        A_scale = A_scale.reshape(A_scale.size(0), 1)
        B_scale = B_scale.reshape(B_scale.size(0), 1)

        C = torch.zeros((batch_size * timestep, out_features, 1), dtype=torch.float16, device='cuda')

        mA = from_dlpack(A_int8, assumed_align=16)
        mB = from_dlpack(B_int8, assumed_align=16)
        mC = from_dlpack(C, assumed_align=16)
        mAScale = from_dlpack(A_scale, assumed_align=16)
        mBScale = from_dlpack(B_scale, assumed_align=16)

        tensor_op_gemm = TensorOpGemmI8(
            cutlass.Int8,
            cutlass.Int8,
            cutlass.Float16,
            cutlass.Int32,
            atom_layout_mnk=atom_layout_mnk,
            use_k32=use_k32,
            bm=bm,
        )
        compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC, mAScale, mBScale)
        avg_time_us = cute.testing.benchmark(
            compiled_gemm,
            kernel_arguments=cute.testing.JitArguments(mA, mB, mC, mAScale, mBScale),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        queue.put(("ok", avg_time_us))
    except Exception as exc:
        queue.put(("err", repr(exc)))


def run_candidate_in_subprocess(bm, use_k32, atom_layout_mnk, warmup_iterations, iterations):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_candidate_worker,
        args=(queue, bm, use_k32, atom_layout_mnk, warmup_iterations, iterations),
    )
    process.start()
    process.join()

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return {
                "bm": bm,
                "use_k32": use_k32,
                "atom_layout_mnk": atom_layout_mnk,
                "avg_time_us": payload,
            }

        message = str(payload).lower()
        if "illegal memory access" in message:
            print(
                f"Skipping bm={bm}, use_k32={use_k32}, atom_layout_mnk={atom_layout_mnk}: CUDA illegal memory access"
            )
            return None

        print(
            f"Skipping bm={bm}, use_k32={use_k32}, atom_layout_mnk={atom_layout_mnk}: {payload}"
        )
        return None

    if process.exitcode not in (0, None):
        print(
            f"Skipping bm={bm}, use_k32={use_k32}, atom_layout_mnk={atom_layout_mnk}: subprocess exit code {process.exitcode}"
        )
    return None

def main():
    # Create float input tensors
    A_float = torch.randn((batch_size, timestep, in_features), dtype=torch.float16, device='cuda').reshape(batch_size * timestep, in_features)
    B_float = torch.randn((out_features, in_features), dtype=torch.float16, device='cuda').reshape(out_features, in_features)
    print(f"A_float shape: {A_float.shape}, B_float shape: {B_float.shape}")

    # Quantize to int8
    A_int8, A_scale = quantize_tensor(A_float, dim=-1)
    B_int8, B_scale = quantize_tensor(B_float, dim=-1)

    A_int8 = A_int8.reshape(batch_size * timestep, in_features, 1)
    B_int8 = B_int8.reshape(out_features, in_features, 1)

    A_scale = A_scale.reshape(A_scale.size(0), 1)
    B_scale = B_scale.reshape(B_scale.size(0), 1)

    print(f"A_int8 shape: {A_int8.shape}, B_int8 shape: {B_int8.shape}")
    print(f"A_scale shape: {A_scale.shape}, B_scale shape: {B_scale.shape}")

    # Create output buffer
    C = torch.zeros((batch_size * timestep, out_features, 1), dtype=torch.float16, device='cuda')
    print(f"C shape: {C.shape}")

    # Convert to CUTE tensors
    mA = from_dlpack(A_int8, assumed_align=16)
    mB = from_dlpack(B_int8, assumed_align=16)
    mC = from_dlpack(C, assumed_align=16)
    mAScale = from_dlpack(A_scale, assumed_align=16)
    mBScale = from_dlpack(B_scale, assumed_align=16)

    print('=== Autotuning ampere_gemm kernel ===')
    best_result, all_results = autotune_int8_gemm(mA, mB, mC, mAScale, mBScale)
    
    # Recompile the best config in the main process (subprocess can't return the kernel object)
    print(f"Recompiling best config: bm={best_result['bm']}, use_k32={best_result['use_k32']}, atom_layout_mnk={best_result['atom_layout_mnk']}")
    tensor_op_gemm_best = TensorOpGemmI8(
        cutlass.Int8,
        cutlass.Int8,
        cutlass.Float16,
        cutlass.Int32,
        atom_layout_mnk=best_result["atom_layout_mnk"],
        use_k32=best_result["use_k32"],
        bm=best_result["bm"],
    )
    compiled_gemm = cute.compile(tensor_op_gemm_best, mA, mB, mC, mAScale, mBScale)
    
    print('=== Running best GEMM === ')
    compiled_gemm(mA, mB, mC, mAScale, mBScale)

    # Dequantize output: C_int32 * A_scale * B_scale
    C_dequantized = C

    # Float reference: A_float @ B_float^T
    A_ref = A_float.reshape(batch_size * timestep, in_features)  # (M, K)
    B_ref = B_float.reshape(out_features, in_features).t()  # (K, N)
    C_ref = torch.matmul(A_ref, B_ref)  # (M, N)

    print(f"\nC_dequantized shape: {C_dequantized.shape}")
    print(f"C_ref shape: {C_ref.shape}")
    print(f"\nC_dequantized[:3, :3]:\n{C_dequantized[:3, :3, 0]}")
    print(f"C_ref[:3, :3]:\n{C_ref[:3, :3]}")

    # Compare
    C_ref_match = C_ref[:, :out_features]
    if torch.allclose(C_dequantized[:, :, 0], C_ref_match, atol=2.0, rtol=1e-1):
        print("\n✓ Results match!")
    else:
        print("\n✗ Results differ")
        print(f"Max diff: {(C_dequantized[:, :, 0] - C_ref_match).abs().max()}")
        print(f"Mean diff: {(C_dequantized[:, :, 0] - C_ref_match).abs().mean()}")

    # Benchmark
    print("\n=== Benchmarking GEMM kernel ===")
    num_elements = sum([A_int8.numel(), B_int8.numel(), C.numel()])

    def benchmark_quant(callable, a_, b_, c_, a_scale_, b_scale_):
        avg_time_us = cute.testing.benchmark(
            callable,
            kernel_arguments=cute.testing.JitArguments(a_, b_, c_, a_scale_, b_scale_),
            warmup_iterations=5,
            iterations=100,
        )

        dtype = a_.element_type
        bytes_per_element = dtype.width // 8
        total_bytes = num_elements * bytes_per_element
        achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s
        gtops = num_elements / (avg_time_us * 1000)  # GTOPS

        print(f"Performance Metrics:")
        print(f"-------------------")
        print(f"Kernel execution time: {avg_time_us:.4f} us")
        print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")
        print(f"GTOPS: {gtops:.2f}")

    benchmark_quant(compiled_gemm, mA, mB, mC, mAScale, mBScale)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()