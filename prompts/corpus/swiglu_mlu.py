import torch
import torch.nn.functional as F

import pytest

import triton
import triton.language as tl
from triton.language.extra.mlu import libdevice
import triton.backends.mlu.driver as driver


_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())

NUM_CLUSTER = _devprob.get("cluster_num")
TOTAL_CORE_NUM = NUM_CLUSTER * _devprob.get("core_num_per_cluster")


def diff(x, y):
    assert x.shape == y.shape
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    diff_max = torch.max(torch.abs(x - y)).item()
    diff_sum = torch.sum(torch.abs(x - y)).item()
    return f"diff.max: {diff_max:.3f}, diff.avg: {100.0 * diff_sum / (torch.sum(torch.abs(x)).item() + 1e-10):.3f}%"


#### START KERNEL


def naive_torch_swiglu(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(torch.matmul(x, w_g) + b_g)
    fc = torch.matmul(x, w_fc) + b_fc
    y = gate * fc
    return y


@triton.jit
def silu_fwd(x, perf_mode: tl.constexpr):
    if perf_mode:
        return triton.language.extra.mlu.libdevice.ultra_silu(x)
    else:
        return triton.language.extra.mlu.libdevice.fast_silu(x)


@triton.jit
def silu_bwd(x, perf_mode: tl.constexpr):
    if perf_mode:
        return triton.language.extra.mlu.libdevice.ultra_silubp(x)
    else:
        sig = triton.language.extra.mlu.libdevice.fast_sigmoid(x)
        return sig * (1 + x * (1 - sig))


def fwd_autotune_config_opt():
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK},
            num_stages=s,
            num_warps=w,
        )
        for BM in [256]
        for BN in [128]
        for BK in [512]
        for s in [5]
        for w in [4]
    ]
    return configs


# @fast_libentry()
# @libentry.libtuner(
@triton.autotune(
    configs=fwd_autotune_config_opt(),
    key=["real_seq_len", "N", "K", "perf_mode"],
)
@triton.jit
def fused_swiglu_fwd_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    g_ptr,
    fc_ptr,
    real_seq_len,
    M,
    N,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    perf_mode: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dtype = y_ptr.type.element_ty

    start = tl.program_id(axis=0)
    step = tl.num_programs(axis=0)

    num_block_m = tl.cdiv(real_seq_len, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_num_blocks = num_block_m * num_block_n

    offset_k = tl.arange(0, BLOCK_SIZE_K)
    for block_id in range(start, total_num_blocks, step):
        block_id_m = block_id // num_block_n
        block_id_n = block_id % num_block_n

        offset_m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offset_m < real_seq_len
        offset_n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offset_n < N

        x_ptrs = x_ptr + (offset_m[:, None] * stride_xm + offset_k[None, :] * stride_xk)
        w_g_ptrs = w_g_ptr + (
            offset_k[:, None] * stride_wk + offset_n[None, :] * stride_wn
        )
        w_fc_ptrs = w_fc_ptr + (
            offset_k[:, None] * stride_wk + offset_n[None, :] * stride_wn
        )
        b_g_ptrs = b_g_ptr + offset_n
        b_fc_ptrs = b_fc_ptr + offset_n

        b_g = tl.load(b_g_ptrs)
        b_fc = tl.load(b_fc_ptrs)

        accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        k_iters: tl.constexpr = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        for k in range(0, k_iters):
            mask_k = k * BLOCK_SIZE_K + offset_k < K
            x = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                cache_modifier=".ca",
            )
            w_g = tl.load(
                w_g_ptrs,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
                cache_modifier=".ca",
            )
            w_fc = tl.load(
                w_fc_ptrs,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
                cache_modifier=".ca",
            )
            accumulator_g = tl.dot(x, w_g, accumulator_g, allow_tf32=False)
            accumulator_fc = tl.dot(x, w_fc, accumulator_fc, allow_tf32=False)
            # Advance the ptrs to the next K block.
            x_ptrs += BLOCK_SIZE_K * stride_xk
            w_g_ptrs += BLOCK_SIZE_K * stride_wk
            w_fc_ptrs += BLOCK_SIZE_K * stride_wk
        accumulator_g += b_g[None, :]
        accumulator_fc += b_fc[None, :]
        if perf_mode:
            accumulator_g = accumulator_g.to(dtype)
            accumulator_fc = accumulator_fc.to(dtype)
            silu_g = silu_fwd(accumulator_g, perf_mode)
            hadamard_product = silu_g * accumulator_fc
        else:
            silu_g = silu_fwd(accumulator_g, perf_mode)
            hadamard_product = silu_g * accumulator_fc
            accumulator_g = accumulator_g.to(dtype)
            accumulator_fc = accumulator_fc.to(dtype)
        y = hadamard_product.to(dtype)

        y_ptrs = y_ptr + stride_ym * offset_m[:, None] + stride_yn * offset_n[None, :]
        g_ptrs = g_ptr + stride_ym * offset_m[:, None] + stride_yn * offset_n[None, :]
        fc_ptrs = fc_ptr + stride_ym * offset_m[:, None] + stride_yn * offset_n[None, :]

        y_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(y_ptrs, y, mask=y_mask)
        tl.store(g_ptrs, accumulator_g, mask=y_mask)
        tl.store(fc_ptrs, accumulator_fc, mask=y_mask)


def bwd_b_autotune_config():
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_stages=s, num_warps=w
        )
        for BM in [256]
        for BN in [64]
        for s in [3]
        for w in [1]
    ]
    return configs


@triton.autotune(
    configs=bwd_b_autotune_config(),
    key=["real_seq_len", "N", "perf_mode"],
)
@triton.jit
def fused_swiglu_bwd_b_kernel(
    dy_ptr,
    g_ptr,
    fc_ptr,
    dg_ptr,
    dfc_ptr,
    db_g_ptr,
    db_fc_ptr,
    real_seq_len,
    M: tl.constexpr,
    N: tl.constexpr,
    perf_mode: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    dtype = dy_ptr.type.element_ty

    col_idx = tl.program_id(axis=0)

    row_off = tl.arange(0, BLOCK_SIZE_M)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dy_ptrs = dy_ptr + (row_off[:, None] * N + col_off[None, :])
    g_ptrs = g_ptr + (row_off[:, None] * N + col_off[None, :])
    fc_ptrs = fc_ptr + (row_off[:, None] * N + col_off[None, :])
    dg_ptrs = dg_ptr + (row_off[:, None] * N + col_off[None, :])
    dfc_ptrs = dfc_ptr + (row_off[:, None] * N + col_off[None, :])

    sum_b_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sum_b_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for row_idx in range(0, tl.cdiv(real_seq_len, BLOCK_SIZE_M)):
        mask = (row_off[:, None] < real_seq_len - row_idx * BLOCK_SIZE_M) & (
            col_off[None, :] < N
        )
        dy = tl.load(dy_ptrs, mask=mask, other=0.0)
        g = tl.load(g_ptrs, mask=mask, other=0.0)
        fc = tl.load(fc_ptrs, mask=mask, other=0.0)

        if not perf_mode:
            dy = dy.to(tl.float32)
            g = g.to(tl.float32)
            fc = fc.to(tl.float32)

        dg = dy * fc * silu_bwd(g, perf_mode)
        dfc = dy * silu_fwd(g, perf_mode)

        tl.store(dg_ptrs, dg.to(dtype), mask=mask)
        tl.store(dfc_ptrs, dfc.to(dtype), mask=mask)

        sum_b_g += dg
        sum_b_fc += dfc
        dy_ptrs += BLOCK_SIZE_M * N
        g_ptrs += BLOCK_SIZE_M * N
        fc_ptrs += BLOCK_SIZE_M * N
        dg_ptrs += BLOCK_SIZE_M * N
        dfc_ptrs += BLOCK_SIZE_M * N

    # do reduction on M
    db_g = tl.sum(sum_b_g, 0).to(dtype)
    db_fc = tl.sum(sum_b_fc, 0).to(dtype)

    tl.store(db_g_ptr + col_off, db_g, mask=col_off < N)
    tl.store(db_fc_ptr + col_off, db_fc, mask=col_off < N)


def do_config_prune(configs, named_args, **kwargs):
    block_set = set()
    N = named_args["N"]

    for config in configs:
        block_size = config.kwargs["BLOCK_SIZE"]
        block_size_m = block_size // N
        block_size_n = N
        if block_size_m == 0:
            continue
        block_set.add((block_size_m, block_size_n))

    assert block_set, "no valid configs available"

    pruned_configs = []
    for block_size_m, block_size_n in block_set:
        for ns in [3]:
            pruned_configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_size_m,
                        "BLOCK_SIZE_N": block_size_n,
                    },
                    num_stages=ns,
                    num_warps=1,
                )
            )

    return pruned_configs


def bwd_b_autotune_config_opt():
    configs = [
        triton.Config({"BLOCK_SIZE": bs, "BLOCK_SIZE_M": 0, "BLOCK_SIZE_N": 0})
        for bs in [4096, 6144, 8192, 16384]
    ]
    return configs


@triton.jit
def fused_swiglu_bwd_b_kernel_opt(
    dy_ptr,
    g_ptr,
    fc_ptr,
    dg_ptr,
    dfc_ptr,
    partial_db_g_ptr,
    partial_db_fc_ptr,
    reduce_core_num_ptr,
    real_seq_len,
    M,
    N,
    perf_mode: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # NOTE: BLOCK_SIZE_N should be equal to N
    dtype = dy_ptr.type.element_ty

    pid = tl.program_id(axis=0)
    step = tl.num_programs(axis=0)

    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    total_num_blocks = num_blocks_m * 1  # N is intact

    if pid == 0:
        reduce_core_num = min(step, (real_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        tl.store(reduce_core_num_ptr, reduce_core_num)

    sum_db_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sum_db_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for block_id in range(pid, total_num_blocks, step):
        offs_m = block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < real_seq_len

        offs = offs_m[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        mask = mask_m[:, None]

        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)

        if not perf_mode:
            dy = dy.to(tl.float32)
            g = g.to(tl.float32)
            fc = fc.to(tl.float32)

        dg = dy * fc * silu_bwd(g, perf_mode)
        dfc = dy * silu_fwd(g, perf_mode)
        sum_db_g += dg
        sum_db_fc += dfc

        tl.store(dg_ptr + offs, dg.to(dtype), mask=mask)
        tl.store(dfc_ptr + offs, dfc.to(dtype), mask=mask)

    if pid < total_num_blocks:
        # do reduction on M
        partial_db_g = tl.sum(sum_db_g, 0)
        partial_db_fc = tl.sum(sum_db_fc, 0)

        st_offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        tl.store(partial_db_g_ptr + st_offs, partial_db_g)
        tl.store(partial_db_fc_ptr + st_offs, partial_db_fc)


def bwd_b_reduce_autotune_config_opt():
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": TOTAL_CORE_NUM, "BLOCK_SIZE_N": bs},
            num_stages=ns,
            num_warps=1,
        )
        for bs in [32, 64, 128, 256]
        for ns in [3]
    ]
    return configs


@triton.autotune(
    configs=bwd_b_reduce_autotune_config_opt(),
    key=["N"],
)
@triton.jit
def fused_swiglu_bwd_b_reduce_kernel(
    db_g_ptr,
    db_fc_ptr,
    partial_db_g_ptr,
    partial_db_fc_ptr,
    reduce_core_num_ptr,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    dtype = db_g_ptr.type.element_ty

    M = tl.load(reduce_core_num_ptr)

    pid = tl.program_id(0)
    step = tl.num_programs(0)

    total_num_blocks = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    for block_id in range(pid, total_num_blocks, step):
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M

        offs_n = block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N

        offs = offs_m[:, None] * N + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        part_db_g = tl.load(partial_db_g_ptr + offs, mask=mask, other=0)
        part_db_fc = tl.load(partial_db_fc_ptr + offs, mask=mask, other=0)

        db_g = tl.sum(part_db_g, 0).to(dtype)
        db_fc = tl.sum(part_db_fc, 0).to(dtype)

        tl.store(db_g_ptr + offs_n, db_g, mask=mask_n)
        tl.store(db_fc_ptr + offs_n, db_fc, mask=mask_n)


def bwd_x_autotune_config():
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK},
            num_stages=s,
            num_warps=w,
        )
        for BM in [128, 256]
        for BN in [128, 192, 256]
        for BK in [512, 672]
        for s in [6]
        for w in [4]
    ]
    return configs


@triton.autotune(
    configs=bwd_x_autotune_config(),
    key=["real_seq_len", "N", "K"],
)
@triton.jit
def fused_swiglu_bwd_x_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr_fp32_part_1,
    dx_ptr,
    real_seq_len,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_dym: tl.constexpr,
    stride_dyk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_dxm: tl.constexpr,
    stride_dxn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    step = tl.num_programs(0)

    block_num_m = tl.cdiv(real_seq_len, BLOCK_SIZE_M)
    block_num_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_block_num = block_num_m * block_num_n

    for block_id in range(pid, total_block_num, step):
        block_id_m = block_id // block_num_n
        block_id_n = block_id % block_num_n

        offset_m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offset_m < real_seq_len
        offset_n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offset_n < N
        offset_k = tl.arange(0, BLOCK_SIZE_K)

        offs_d = offset_m[:, None] * stride_dym + offset_k[None, :] * stride_dyk
        dy_ptrs = dy_ptr + offs_d
        offs_w = offset_k[:, None] * stride_wk + offset_n[None, :] * stride_wn
        w_ptrs = w_ptr + offs_w

        acc_dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            mask_k = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K

            dy = tl.load(
                dy_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0,
                cache_modifier=".ca",
            )
            w = tl.load(
                w_ptrs,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
                cache_modifier=".ca",
            )

            acc_dx = tl.dot(dy, w, acc_dx, allow_tf32=False)
            dy_ptrs += BLOCK_SIZE_K * stride_dyk
            w_ptrs += BLOCK_SIZE_K * stride_wk

        offsets_dx = stride_dxm * offset_m[:, None] + stride_dxn * offset_n[None, :]
        dx_ptrs_part_1 = dx_ptr_fp32_part_1 + offsets_dx
        dx_ptrs = dx_ptr + offsets_dx
        dx_mask = mask_m[:, None] & mask_n[None, :]
        acc_dx_part_1 = tl.load(dx_ptrs_part_1, mask=dx_mask, other=0)
        final_dx = (acc_dx_part_1 + acc_dx).to(dx_ptr.type.element_ty)
        tl.store(dx_ptrs, final_dx, mask=dx_mask)


"""
@torch.library.custom_op("torch_mlu_triton::fused_swiglu_kernel", mutates_args=(), device_types="mlu")
def fused_swiglu_kernel(x: torch.Tensor,
            w_g: torch.Tensor,
            w_fc: torch.Tensor,
            b_g: torch.Tensor,
            b_fc: torch.Tensor,
            perf_mode: bool) -> torch.Tensor:
    # Check constraints.
    assert w_g.shape == w_fc.shape
    assert x.shape[1] == w_g.shape[0], "Incompatible dimensions"
    assert b_g.shape[0] == w_g.shape[1]
    assert b_g.shape[0] == b_fc.shape[0]
    assert x.is_contiguous() and w_g.is_contiguous(
    ) and w_fc.is_contiguous(), "Tensors must be contiguous"
    assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [
        torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn
    ]
    seq_len, dim = x.shape
    dim, ex_dim = w_g.shape
    # Allocates output.
    y = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)
    g = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)
    fc = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)

    grid_warp_4 = lambda meta: (NUM_CLUSTER, )

    fused_swiglu_fwd_kernel[grid_warp_4](
        x,
        w_g,
        w_fc,
        b_g,
        b_fc,
        y,
        g,
        fc,
        seq_len,
        seq_len, #M
        ex_dim, #N
        dim,#K
        dim,
        1,
        ex_dim,
        1,
        ex_dim,
        1,
        perf_mode,
    )

    return y


@fused_swiglu_kernel.register_fake
def fused_swiglu_kernel_fake(x: torch.Tensor,
            w_g: torch.Tensor,
            w_fc: torch.Tensor,
            b_g: torch.Tensor,
            b_fc: torch.Tensor,
            perf_mode: bool) -> torch.Tensor:
    # Check constraints.
    assert w_g.shape == w_fc.shape
    assert x.shape[1] == w_g.shape[0], "Incompatible dimensions"
    assert b_g.shape[0] == w_g.shape[1]
    assert b_g.shape[0] == b_fc.shape[0]
    assert x.is_contiguous() and w_g.is_contiguous(
    ) and w_fc.is_contiguous(), "Tensors must be contiguous"
    assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [
        torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn
    ]
    seq_len, dim = x.shape
    dim, ex_dim = w_g.shape
    # Allocates output.
    y = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)

    return y
"""


class FusedSwiglu(torch.autograd.Function):
    # NOTE: perf_mode:
    #                 True:  use the origin data type to do silu and mult, and use ultra_silu
    #                 False: use float32 to do silu and mult, and use fast_silu
    @staticmethod
    # def forward(ctx, x, w_g, w_fc, b_g, b_fc, perf_mode):
    def forward(
        ctx,
        x,
        w_g,
        w_fc,
        b_g,
        b_fc,
        is_training=True,
        is_recompute=False,
        perf_mode=False,
    ):
        # Check constraints.
        assert w_g.shape == w_fc.shape
        assert x.shape[1] == w_g.shape[0], "Incompatible dimensions"
        assert b_g.shape[0] == w_g.shape[1]
        assert b_g.shape[0] == b_fc.shape[0]
        assert (
            x.is_contiguous() and w_g.is_contiguous() and w_fc.is_contiguous()
        ), "Tensors must be contiguous"
        assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]
        seq_len, dim = x.shape
        dim, ex_dim = w_g.shape
        # Allocates output.
        y = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)
        g = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)
        fc = torch.empty((seq_len, ex_dim), device=x.device, dtype=x.dtype)

        grid_warp_4 = lambda meta: (NUM_CLUSTER,)

        fused_swiglu_fwd_kernel[grid_warp_4](
            x,
            w_g,
            w_fc,
            b_g,
            b_fc,
            y,
            g,
            fc,
            seq_len,
            seq_len,
            ex_dim,
            dim,
            dim,
            1,
            ex_dim,
            1,
            ex_dim,
            1,
            perf_mode,
        )

        ctx.save_for_backward(x, w_g, w_fc, g, fc)
        ctx.perf_mode = perf_mode

        return y

    @staticmethod
    def backward(ctx, dy):
        device = dy.device
        x, w_g, w_fc, g, fc = ctx.saved_tensors
        perf_mode = ctx.perf_mode

        seq_len, ex_dim = dy.shape
        dim, ex_dim = w_g.shape
        dx = torch.empty_like(x)
        dg = torch.empty_like(dy)
        dfc = torch.empty_like(dy)
        db_g = torch.empty((ex_dim,), dtype=dy.dtype, device=device)
        db_fc = torch.empty((ex_dim,), dtype=dy.dtype, device=device)

        grid_warp_1 = lambda META: (TOTAL_CORE_NUM,)
        grid_warp_4 = lambda META: (NUM_CLUSTER,)
        if ex_dim * torch.finfo(dy.dtype).bits // 8 > 16 * 1024:
            # default kernel
            # dg & dfc & db_g & db_fc
            grid = lambda META: (triton.cdiv(ex_dim, META["BLOCK_SIZE_N"]),)
            fused_swiglu_bwd_b_kernel[grid](
                dy, g, fc, dg, dfc, db_g, db_fc, seq_len, seq_len, ex_dim, perf_mode
            )
        else:
            # NOTE: only support BLOCK_SIZE_N>=ex_dim
            # create some workspace with size of TOTAL_CORE_NUM*ex_dim*fp32
            partial_db_g = torch.empty(
                (TOTAL_CORE_NUM, ex_dim), dtype=torch.float32, device=device
            )
            partial_db_fc = torch.empty(
                (TOTAL_CORE_NUM, ex_dim), dtype=torch.float32, device=device
            )
            reduce_core_num = torch.empty((1,), dtype=torch.int32, device=device)

            # dg, dfc, partial db_g and db_fc
            fused_swiglu_bwd_b_kernel_opt[grid_warp_1](
                dy,
                g,
                fc,
                dg,
                dfc,
                partial_db_g,
                partial_db_fc,
                reduce_core_num,
                seq_len,
                seq_len,
                ex_dim,
                perf_mode,
                BLOCK_SIZE_M=16384 // ex_dim,
                BLOCK_SIZE_N=ex_dim,
            )

            # reduce on M
            fused_swiglu_bwd_b_reduce_kernel[grid_warp_1](
                db_g, db_fc, partial_db_g, partial_db_fc, reduce_core_num, ex_dim
            )

        # dx
        # M: seq_len, N: dim, K: ex_dim (reduce_axis)
        dx_partial = torch.mm(dg, w_g.t())
        dx = torch.addmm(dx_partial, dfc, w_fc.t())
        # dw_g & dw_fc
        # M: dim, N: ex_dim, K: bs * seq_len (reduce_axis)
        dw_g = torch.mm(x.t(), dg)
        dw_fc = torch.mm(x.t(), dfc)

        return dx, dw_g, dw_fc, db_g, db_fc, None, None, None

    #### END KERNEL
