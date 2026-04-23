#### START KERNEL
@triton.jit
def triple_implicit_gemm_conv1d_fwd_kernel(
    output_ptr_0,
    input_ptr_0,
    weight_ptr_0,
    bias_ptr_0,
    res_ptr_0,
    N,
    C: tl.constexpr,
    H,
    K: tl.constexpr,
    P_0,
    R_0: tl.constexpr,
    str_h_0: tl.constexpr,
    pad_h_0: tl.constexpr,
    dil_h_0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    GEMM_M = N * P_0
    GEMM_N = K
    GEMM_K = C * R_0

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)

    if pid >= num_pid_m * num_pid_n:
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
        gemm_k = idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        c = gemm_k % C
        r = gemm_k // C

        h = p[:, None] * str_h_0 + r[None, :] * dil_h_0 - pad_h_0

        mask_input = (h >= 0) & (h < H) & (r[None, :] < R_0)
        mask_weight = r[:, None] < R_0

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
    run_config: Optional[dict] = None,
):
    N, H, C = input0.shape
    K = weight0.shape[0]
    R0 = weight0.shape[1]

    P0 = (H + 2 * padding0 - dilation0 * (R0 - 1) - 1) // stride0 + 1

    output0 = torch.empty((N, P0, K), dtype=input0.dtype, device=input0.device)

    if run_config is None:
        run_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_stages": 3,
            "num_warps": 4,
        }
    BLOCK_SIZE_M = run_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = run_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = run_config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = run_config["GROUP_SIZE_M"]
    num_stages = run_config["num_stages"]
    num_warps = run_config["num_warps"]

    grid = lambda META: (
        triton.cdiv(N * P0, META["BLOCK_SIZE_M"])
        * triton.cdiv(K, META["BLOCK_SIZE_N"]),
    )

    triple_implicit_gemm_conv1d_fwd_kernel[grid](
        output0,
        input0,
        weight0,
        bias0,
        res0,
        N,
        C,
        H,
        K,
        P0,
        R0,
        stride0,
        padding0,
        dilation0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return output0


#### END KERNEL
