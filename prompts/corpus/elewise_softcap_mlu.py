import torch
import pytest
import triton
import triton.language as tl
from triton.language.extra.mlu import libdevice
import triton.backends.mlu.driver as driver

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CORE_NUM = _devprob.get("cluster_num") * _devprob.get("core_num_per_cluster")

#### START KERNEL


def get_autotune_config():
    configs = [
        triton.Config({"BLOCK_SIZE": B}, num_stages=s, num_warps=w)
        for B in [1024, 2048, 4096, 8192, 12288, 16384, 18432, 21760, 32768]
        for s in [1, 3, 5]
        for w in [1, 4]
    ]
    return configs


# @fast_libentry()
# @libentry.libtuner(
@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"],
)
@triton.jit
def softcap_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_jobs = tl.num_programs(axis=0)
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    # Add to tl.int64 for large tensor
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements

        # Load input from DRAM, masking out any extra elements in case the inputs is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        coef = 1 / softcap
        y = softcap * (
            triton.language.extra.mlu.libdevice.fast_tanh(x.to(tl.float32) * coef)
        )
        tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"],
)
@triton.jit
def softcap_bwd_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    # Add to tl.int64 for large tensor
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        # 1. load dy & x
        dy = tl.load(dy_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        # 2. softcap backward compute
        coef = 1 / softcap
        y = triton.language.extra.mlu.libdevice.fast_tanh(x.to(tl.float32) * coef).to(
            x.dtype
        )
        dx = dy * (1 - y * y)

        # 3. store dx
        tl.store(dx_ptr + offsets, dx, mask=mask)


class Softcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, softcap):
        assert x.is_contiguous(), "Tensors must be contiguous"
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]
        numel = x.numel()
        # Allocates output.
        y = torch.empty_like(x)
        grid = lambda meta: (
            min(triton.cdiv(numel, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
        )
        softcap_fwd_kernel[grid](
            x,
            y,
            numel,
            softcap,
        )

        ctx.save_for_backward(x)
        ctx.softcap = softcap
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        numel = x.numel()
        dx = torch.empty_like(x)
        grid = lambda meta: (
            min(triton.cdiv(numel, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
        )
        softcap_bwd_kernel[grid](
            dy,
            x,
            dx,
            numel,
            ctx.softcap,
        )

        return dx, None


#### END KERNEL
