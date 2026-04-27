import torch
import triton
import triton.language as tl


#### START KERNEL
@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add_wrapper(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()

    # 简单的网格配置
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动内核，预设 BLOCK_SIZE
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


#### END KERNEL

# =============================================================================
# 精度测试与性能基准
# =============================================================================


def test_add_correctness():
    print("🧪 正在进行精度校验...")
    configs = [2**10, 2**15, 2**20]
    for size in configs:
        x = torch.randn(size, device="mlu")
        y = torch.randn(size, device="mlu")

        # Triton 结果
        triton_output = add_wrapper(x, y)
        # Torch 结果 (参考标准)
        torch_output = x + y

        # 比较精度
        is_correct = torch.allclose(triton_output, torch_output, atol=1e-4)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Size {size:8d}: {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_output - torch_output).abs().max()}")


def benchmark_add():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],  # 这里的 N 对应测试规模
            x_vals=[2**i for i in range(12, 25, 2)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s",
            plot_name="add-performance",
            args={},
        )
    )
    def benchmark(N, provider):
        x = torch.randn(N, device="mlu", dtype=torch.float32)
        y = torch.randn(N, device="mlu", dtype=torch.float32)

        if provider == "torch":
            ms = triton.testing.do_bench(lambda: x + y)
        if provider == "triton":
            ms = triton.testing.do_bench(lambda: add_wrapper(x, y))

        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    # 1. 运行精度测试
    test_add_correctness()

    # 2. 运行性能测试
    try:
        benchmark_add()
    except Exception as e:
        print(f"⚠️ 性能测试跳过或出错: {e}")
