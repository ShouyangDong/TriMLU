import triton
import triton.language as tl


#### START KERNEL
@triton.jit
def kernel_abs(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    tl.store(y_ptr + idx, tl.abs(x))


#### END KERNEL
