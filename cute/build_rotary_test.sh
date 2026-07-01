#!/usr/bin/env bash
# Build and run the AOT / C-ABI test for the fused INT8 GEMM + rotary kernel.
#
# This compiles the exported object (cute/artifacts/gemm_i8_rotary_*.o) against a
# C harness (main_rotary.cpp) and runs it — the export_to_c path the Python JIT
# test never exercises. Re-export first if the artifact is missing:
#     pyvenv/bin/python cute/export_gemm_i8_rotary.py
set -euo pipefail

CUTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUTE_DIR}/.." && pwd)"

KERNEL="gemm_i8_rotary_N1536_K512_H8D64R64S1024"
OBJ="${CUTE_DIR}/artifacts/${KERNEL}.o"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
RT_LIB="${REPO_ROOT}/pyvenv/lib/python3.10/site-packages/nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a"
BIN="${CUTE_DIR}/rotary_aot_test"

if [[ ! -f "${OBJ}" ]]; then
    echo "Missing ${OBJ}. Export it first:" >&2
    echo "    ${REPO_ROOT}/pyvenv/bin/python ${CUTE_DIR}/export_gemm_i8_rotary.py" >&2
    exit 1
fi

echo "Compiling ${BIN} ..."
g++ "${CUTE_DIR}/main_rotary.cpp" "${OBJ}" \
    -o "${BIN}" \
    -std=c++17 -O2 \
    -I"${CUTE_DIR}" \
    -I"${CUDA_HOME}/include" \
    -L"${CUDA_HOME}/lib64" -lcudart \
    -L/usr/lib/x86_64-linux-gnu -lcuda \
    "${RT_LIB}" \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu

echo "Running ${BIN} ..."
"${BIN}"
