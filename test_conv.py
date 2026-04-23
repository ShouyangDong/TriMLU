import math
import torch
import triton
import triton.language as tl
import time
from typing import Optional
import multiprocessing

torch.cuda.set_device = torch.mlu.set_device
torch.cuda.synchronize = torch.mlu.synchronize
torch.cuda.is_available = torch.mlu.is_available
torch.cuda.device_count = torch.mlu.device_count
torch.cuda.current_device = torch.mlu.current_device
torch.cuda.manual_seed = torch.mlu.manual_seed
torch.cuda.manual_seed_all = torch.mlu.manual_seed_all
torch.cuda.Event = torch.mlu.Event
torch.cuda.Stream = torch.mlu.Stream
torch.cuda.stream = torch.mlu.stream
torch.cuda.memory = torch.mlu.memory
torch.cuda.current_stream = torch.mlu.current_stream
torch.cuda.mem_get_info = torch.mlu.mem_get_info
torch.cuda.CUDAGraph = torch.mlu.MLUGraph
torch.cuda.graph = torch.mlu.graph
torch.cuda.nvtx = torch.mlu.cnpx
torch.cuda.get_device_name = torch.mlu.get_device_name

def get_device() -> str:
    return f"mlu:{torch.cuda.current_device()}"


@triton.jit
def triple_implicit_gemm_conv1d_fwd_kernel(
    output_ptr_0, input_ptr_0, weight_ptr_0, bias_ptr_0, res_ptr_0,
    N, C: tl.constexpr, H, K: tl.constexpr,
    P_0,
    R_0: tl.constexpr,
    str_h_0: tl.constexpr,
    pad_h_0: tl.constexpr,
    dil_h_0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    GEMM_M = N * P_0
    GEMM_N = K
    GEMM_K = C * R_0

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)

    if (pid >= num_pid_m * num_pid_n):
        return

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // P_0
    p = gemm_i % P_0
    k = gemm_j

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c = gemm_k % C
        r = gemm_k // C

        h = p[:, None] * str_h_0 + r[None, :] * dil_h_0 - pad_h_0

        mask_input = (h >= 0) & (h < H) & (r[None, :] < R_0)
        mask_weight = (r[:, None] < R_0)

        offs_input = n[:, None] * H * C + h * C + c[None, :]
        offs_weight = k[None, :] * C * R_0 + r[:, None] * C + c[:, None]

        input_ptrs = input_ptr_0 + offs_input
        weight_ptrs = weight_ptr_0 + offs_weight

        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc = tl.dot(input_data, weight_data, acc)

    if bias_ptr_0 is not None:
        offs_bias = k[None, :]
        bias_ptrs = bias_ptr_0 + offs_bias
        bias_data = tl.load(bias_ptrs).to(tl.float32)
        acc = acc + bias_data

    offs_npqk = n[:, None] * P_0 * K + p[:, None] * K + k[None, :]
    if res_ptr_0 is not None:
        res_ptrs = res_ptr_0 + offs_npqk
        res_data = tl.load(res_ptrs).to(tl.float32)
        acc = acc + res_data

    acc = acc.to(output_ptr_0.type.element_ty)
    output_ptrs = output_ptr_0 + offs_npqk
    tl.store(output_ptrs, acc)


def triton_implicit_gemm_conv1d_fwd(
        input0: torch.Tensor,
        weight0: torch.Tensor,
        bias0: torch.Tensor,
        res0: torch.Tensor,
        stride0: int,
        padding0: int,
        dilation0: int,
        run_config: Optional[dict] = None
    ):
    N, H, C = input0.shape
    K = weight0.shape[0]
    R0= weight0.shape[1]

    P0 = (H + 2 * padding0 - dilation0 * (R0 - 1) - 1) // stride0 + 1

    output0 = torch.empty((N, P0, K), dtype=input0.dtype, device=input0.device)

    if run_config is None:
        run_config = {
            'BLOCK_SIZE_M': 64, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 64, 
            'GROUP_SIZE_M': 1, 
            "num_stages": 3, 
            "num_warps": 4 
        }
    BLOCK_SIZE_M = run_config['BLOCK_SIZE_M']
    BLOCK_SIZE_N = run_config['BLOCK_SIZE_N']
    BLOCK_SIZE_K = run_config['BLOCK_SIZE_K']
    GROUP_SIZE_M = run_config['GROUP_SIZE_M']
    num_stages = run_config['num_stages']
    num_warps = run_config['num_warps']

    grid = lambda META: (triton.cdiv(N * P0, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']), )

    triple_implicit_gemm_conv1d_fwd_kernel[grid](
        output0, input0, weight0, bias0, res0,
        N, C, H, K,
        P0,
        R0,
        stride0,
        padding0,
        dilation0,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages, num_warps=num_warps
    )

    return output0


def kernel_test_1d(
    inputs
):
    torch.cuda.set_device(0)

    (N, C, H, F, K0, S0, D0, P0, has_bias, has_res, do_perf) = inputs

    a0 = torch.randn((N, H, C), device=get_device(), dtype=torch.float16)
    b0 = torch.randn((F, K0, C), device=get_device(), dtype=torch.float16)
    bias0= None
    if has_bias:
        bias0 = torch.randn((F,), device=get_device(), dtype=torch.float16)
    res0= None
    if has_res:
        res0 = torch.randn((N, H, F), device=get_device(), dtype=torch.float16)

    print(f"\n\n\n########### N:{N}, H:{H}, F:{F}, K0:{K0}, C:{C}, S0:{S0}##########")


    triton_output0 = triton_implicit_gemm_conv1d_fwd(
            a0,
            b0,
            bias0,
            res0,
            S0,
            P0,
            D0,
        )

    ref_out0 = torch.nn.functional.conv1d(
            a0.transpose(-1, -2), b0.transpose(-1, -2), bias0, S0, P0, D0
        ).transpose(-1, -2) + res0

    max_diff0 = torch.max(torch.abs(triton_output0 - ref_out0)).item()
    mean_diff0 = torch.mean(torch.abs(triton_output0 - ref_out0)).item()

    print(
            f"max_diff0: {max_diff0:.4f}  " + f"mean_diff0: {mean_diff0:.4f}"
        )
    assert max_diff0 < 0.4
    assert mean_diff0 < 0.03

    def conv_triton():
        triton_output0 = triton_implicit_gemm_conv1d_fwd(
            a0,
            b0,
            bias0,
            res0,
            S0,
            P0,
            D0,
        )

    def conv_torch():
        ref_out0 = torch.nn.functional.conv1d(
                a0.transpose(-1, -2), b0.transpose(-1, -2), bias0, S0, P0, D0
            ).transpose(-1, -2) + res0

    OH0 = (H + 2 * P0 - D0 * (K0 - 1) - 1) // S0 + 1

    ms_tl = triton.testing.do_bench(conv_triton)
    ms_torch = triton.testing.do_bench(conv_torch)
    flops = 2 * N * OH0 * F * K0 * C
    print(
        f"triton: {ms_tl:.3f} ms, {N=}, {H=}, {F=}, {K0=}, {C=}, {S0=}, FLOPS={flops*1e-9/ms_tl:.2f} T/s"
    )
    print(
        f"Torch: {ms_torch:.3f} ms, {N=}, {H=}, {F=}, {K0=}, {C=}, {S0=}, FLOPS={flops*1e-9/ms_torch:.2f} T/s"
    )

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    do_perf = True
    args_list = []
    for N in [10]:
        for C, H in [
            (768, 110),
            (384, 550),
            (192, 1650),
            (96, 3300),
            (48, 6600),
            (24, 13200),
        ]:
            for D in [1, 3, 5]:
                S0 = 1
                F = C  # F: out_channel, C: in_channel
                D0 = D
                K0= 7
                P0 = get_padding(K0, D0)
                args_list.append(
                    (
                        N,
                        C,
                        H,
                        F,
                        K0,
                        S0,
                        D0,
                        P0,
                        True,
                        True,
                        do_perf
                    )
                )

    for args in args_list:
        kernel_test_1d(args)
