import json
import torch
import triton
import triton.language as tl


#### START KERNEL
def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_triton(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Stride variables
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_wrapper(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_triton[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


#### END KERNEL

# =============================================================================
# 2. 精度测试与性能基准 (适配 TriMLU 协议)
# =============================================================================


def test_matmul_correctness():
    print("🧪 正在进行 Matmul 精度校验...")
    configs = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (848, 432, 720),  # 测试非对齐维度
    ]
    for M, N, K in configs:
        a = torch.randn((M, K), device="mlu", dtype=torch.float16)
        b = torch.randn((K, N), device="mlu", dtype=torch.float16)

        # Triton 结果
        triton_out = matmul_wrapper(a, b)
        # Torch 结果
        torch_out = torch.matmul(a, b)

        # 比较精度 (FP16 允许较大的一点误差)
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Size {M}x{N}x{K}: {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_out - torch_out).abs().max()}")
            return False
    return True


def benchmark_matmul():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")

    # 用于收集数据并转换成约定格式
    results_for_trimlu = []

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[1024 * i for i in range(1, 5)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-performance",
            args={},
        )
    )
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="mlu", dtype=torch.float16)
        b = torch.randn((K, N), device="mlu", dtype=torch.float16)

        if provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        else:
            ms = triton.testing.do_bench(lambda: matmul_wrapper(a, b))
            # 将 Triton 的耗时记录下来用于 TriMLU 评估
            results_for_trimlu.append({"latency": ms, "M": M, "N": N, "K": K})

        # TFLOPS = (2 * M * N * K) / (ms * 1e-3) / 1e12
        tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return tflops

    # 我们需要运行多次采样，benchmark.run 会多次调用上面的函数
    # 这里通过 N, M, K 的组合来触发不同的 x_vals
    for val in [1024 * i for i in range(1, 5)]:
        benchmark(M=val, N=val, K=val, provider="triton")
        benchmark(M=val, N=val, K=val, provider="torch")

    return results_for_trimlu


if __name__ == "__main__":
    # 1. 精度校验
    passed = test_matmul_correctness()

    if not passed:
        # 如果精度没过，打印空 JSON 强制拒绝优化
        print("__TRIMLU_PERF_JSON__:[]")
    else:
        # 2. 性能测试
        try:
            perf_data = benchmark_rmsnorm_data = []
            # 运行性能测试逻辑并捕获 ms 延迟
            # 注意：这里直接手动调用逻辑以便更精准地控制 JSON 输出
            configs = [1024 * i for i in range(1, 5)]
            for val in configs:
                a = torch.randn((val, val), device="mlu", dtype=torch.float16)
                b = torch.randn((val, val), device="mlu", dtype=torch.float16)
                ms = triton.testing.do_bench(lambda: matmul_wrapper(a, b))
                benchmark_rmsnorm_data.append({"latency": ms, "size": val})

            # Orchestrator 通过识别此特定前缀来解析性能结果
            print(f"__TRIMLU_PERF_JSON__:{json.dumps(benchmark_rmsnorm_data)}")
        except Exception as e:
            print(f"⚠️ 性能测试跳过或出错: {e}")
            print("__TRIMLU_PERF_JSON__:[]")
