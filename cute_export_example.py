import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream

@cute.kernel
def print_tensor_kernel(a: cute.Tensor):
    cute.printf("a: {}", a)

@cute.jit
def print_tensor(a: cute.Tensor, stream: CUstream):
    print_tensor_kernel(a).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

# Build fake args for AOT compilation — no real GPU data needed
n = cute.sym_int()
fake_a = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
fake_stream = CUstream(0)  # null stream is fine for type tracing

compiled_func = cute.compile(print_tensor, fake_a, fake_stream)

# Export to C
compiled_func.export_to_c(
    file_path="./artifacts",
    file_name="print_tensor_example",
    function_prefix="print_tensor"
)