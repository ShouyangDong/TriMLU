import torch
import triton
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(1234)


#### START KERNEL
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # 程序启动的起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # 采用持久化循环 (Persistent Kernel) 模式处理多行
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        # 加载行数据，Mask 外补负无穷以确保 max 正确
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # 减去最大值以保证数值稳定性
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # 写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton_wrapper(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # MLU 硬件属性获取
    device = torch.mlu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["cluster_num"] * properties["core_num_per_cluster"]

    # 启发式参数设置
    num_warps = 8
    num_stages = 2  # MLU 上通常设为 2

    y = torch.empty_like(x)
    # 持久化线程组：取 SM 数量与行数的最小值
    num_programs = min(NUM_SM, n_rows)

    softmax_kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


#### END KERNEL

# =============================================================================
# 2. 精度测试与性能基准 (适配 TriMLU 协议)
# =============================================================================


def test_softmax_correctness():
    print("🧪 正在进行 Softmax 精度校验...")
    configs = [
        (128, 512),
        (1024, 4096),
    ]
    for M, N in configs:
        x = torch.randn((M, N), device="mlu", dtype=torch.float32)
        # Triton 结果
        triton_out = softmax_triton_wrapper(x)
        # Torch 结果
        torch_out = torch.softmax(x, dim=1)

        # 比较精度
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-5)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Shape {M}x{N}: {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_out - torch_out).abs().max()}")
            return False
    return True


def benchmark_softmax():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")

    # 用于收集数据并转换成约定格式
    results_for_trimlu = []

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(8, 13)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="softmax-performance",
            args={"M": 1024},
        )
    )
    def benchmark(N, M, provider):
        x = torch.randn((M, N), device="mlu", dtype=torch.float32)

        if provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
        else:
            ms = triton.testing.do_bench(lambda: softmax_triton_wrapper(x))
            # 将 Triton 的耗时记录下来
            results_for_trimlu.append({"latency": ms, "n_cols": N})

        return ms

    # 执行测试
    benchmark.run(show_plots=False, print_data=True)
    return results_for_trimlu


if __name__ == "__main__":
    # 1. 精度校验
    passed = test_softmax_correctness()

    if not passed:
        # 如果精度没过，可以主动抛错或者打印空的 JSON 强制拒绝优化
        print("__TRIMLU_PERF_JSON__:[]")
    else:
        # 2. 性能测试
        try:
            perf_data = benchmark_softmax()
            # 重要：Orchestrator 通过识别此特定前缀来解析性能结果
            # 将多组测试结果打印，Orchestrator 会取其平均值
            print(f"__TRIMLU_PERF_JSON__:{json.dumps(perf_data)}")
        except Exception as e:
            print(f"⚠️ 性能测试出错: {e}")
            print("__TRIMLU_PERF_JSON__:[]")
