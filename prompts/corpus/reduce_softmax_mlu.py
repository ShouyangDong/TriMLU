import torch
import torch_mlu

import triton
import triton.language as tl
import triton.backends.mlu.driver as driver

import copy
import math

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())

TOTAL_CORE_NUM = _devprob.get("cluster_num") * _devprob.get("core_num_per_cluster")

MAX_NRAM_SIZE = _devprob.get("max_nram_size")
MAX_SRAM_SIZE = _devprob.get("max_shared_mem")
MAX_N = 16385


def align(max_block, dtype):
    width = torch.tensor([], dtype=dtype).element_size()
    if max_block * width < 64:
        return max_block
    a = triton.next_power_of_2(max_block)
    return max_block if max_block == a else a // 2


#### START KERNEL


@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


def config_prune(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    input = named_args["input_ptr"]
    configs_map = {}
    # When N is less than MAX_C_MLU_SOFTMAX_FORWARD, no reduction loops
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            config.num_warps,
            config.num_stages,
        )
        if N < MAX_N:
            config = copy.deepcopy(config)
            BLOCK_N = config.kwargs["BLOCK_N"] = N
            m_per_core = math.ceil(M / TOTAL_CORE_NUM)

            BLOCK_M = config.kwargs["BLOCK_M"] = m_per_core
            num_stages = config.num_stages = 1
            key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
            configs_map.setdefault(key, config)

            config = copy.deepcopy(config)
            max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (2 * BLOCK_N + 1)
            BLOCK_M = config.kwargs["BLOCK_M"] = align(
                max_block_m_without_pipe, input.dtype
            )
            key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
            configs_map.setdefault(key, config)

            config = copy.deepcopy(config)
            max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (4 * BLOCK_N + 1)
            if input.dtype == torch.float32:
                max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (4 * BLOCK_N + 1)
            BLOCK_M = config.kwargs["BLOCK_M"] = align(
                max_block_m_without_pipe, input.dtype
            )
            num_stages = config.num_stages = 3
            key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
            configs_map.setdefault(key, config)

            config = copy.deepcopy(config)
            max_block_m_u1_with_pipe1 = MAX_NRAM_SIZE // 4 // (5 * BLOCK_N + 1)
            max_block_m_u1_with_pipe2 = MAX_SRAM_SIZE // 4 // (4 * BLOCK_N + 1) // 4
            max_block_m_u1_with_pipe = min(
                max_block_m_u1_with_pipe1, max_block_m_u1_with_pipe2
            )
            BLOCK_M = config.kwargs["BLOCK_M"] = align(
                max_block_m_u1_with_pipe, input.dtype
            )
            num_stages = config.num_stages = 5
            num_warps = config.num_warps = 4
            BLOCK_M = config.kwargs["BLOCK_M"] = 4 * BLOCK_M
            key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
            configs_map.setdefault(key, config)

        key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    # Add a heuristic config.
    extra_config = copy.deepcopy(pruned_configs[0])
    extra_config.kwargs["BLOCK_M"] = 1
    extra_config.kwargs["BLOCK_N"] = N
    extra_config.num_warps = 1
    extra_config.num_stages = 3
    pruned_configs.append(extra_config)
    extra_config2 = copy.deepcopy(extra_config)
    extra_config2.num_stages = 1
    pruned_configs.append(extra_config2)
    return pruned_configs


def softmax_tile_mode_for_inner(args):
    one_tile_m = args["BLOCK_M"] * TOTAL_CORE_NUM // args["num_warps"] >= args["M"]
    one_tile_n = args["BLOCK_N"] >= args["N"]
    if one_tile_n and one_tile_m:
        return 0
    elif one_tile_n and not one_tile_m:
        return 1
    else:
        return 2


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": 2**bn}, num_stages=s, num_warps=1)
        for bm in [1, 2, 6, 11, 22]
        for bn in range(8, 14, 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
    ],
    prune_configs_by={"early_config_prune": config_prune},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: softmax_tile_mode_for_inner(args),
    },
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_MODE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m

    if TILE_MODE == 0:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offset = tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        trans_inp = tl.trans(inp)
        row_minus_max = trans_inp - tl.max(trans_inp, axis=0)[None, :]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)[None, :]
        recip = 1.0 / denominator
        softmax_output = tl.trans(numerator * recip)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, softmax_output, mask=mask)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            n_offset = tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            # FIXME: merge the following two branches after
            # optimizing the movement of 'transpose'.
            if num_warps == 4:
                trans_inp = tl.trans(inp)
                row_minus_max = trans_inp - tl.max(trans_inp, axis=0)[None, :]
                numerator = tl.exp(row_minus_max)
                denominator = tl.sum(numerator, axis=0)[None, :]
                recip = 1.0 / denominator
                softmax_output = tl.trans(numerator * recip)
                output_ptrs = output_ptr + offset
                tl.store(output_ptrs, softmax_output, mask=mask)
            else:
                row_minus_max = inp - tl.max(inp, axis=1)[:, None]
                numerator = tl.exp(row_minus_max)
                denominator = tl.sum(numerator, axis=1)[:, None]
                recip = 1.0 / denominator
                softmax_output = numerator * recip
                output_ptrs = output_ptr + offset
                tl.store(output_ptrs, softmax_output, mask=mask)
    else:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            block_max = tl.full(
                [BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32
            )
            block_sum = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                cur_max = tl.maximum(block_max, inp)
                alpha = tl.exp(block_max - cur_max)
                block_sum = block_sum * alpha + tl.exp(inp - cur_max)
                block_max = cur_max

            trans_block_max = tl.trans(block_max)
            trans_block_sum = tl.trans(block_sum)
            max_reduced = tl.max(trans_block_max, 0)
            total_sum = tl.sum(
                trans_block_sum * tl.exp(trans_block_max - max_reduced[None, :]), 0
            )
            recip_total_sum = 1.0 / total_sum
            total_max = max_reduced

            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                o = tl.exp(inp - total_max[:, None]) * recip_total_sum[:, None]
                tl.store(output_ptr + offset, o, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = n_cols
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. We split all rows into TOTAL_CORE_NUM parts evenly.
    grid = lambda META: (TOTAL_CORE_NUM // META["num_warps"], 1, 1)
    softmax_kernel_inner[(grid)](y, x, n_rows, n_cols, bottleneck="simd")
    return y


#### END KERNEL
