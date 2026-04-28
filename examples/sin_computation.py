import json
import torch
import triton
import triton.language as tl

torch.manual_seed(1234)


#### START KERNEL
@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)


def sin_triton_wrapper(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    # 优化 grid 计算与 BLOCK_SIZE
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sin_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


#### END KERNEL

# =============================================================================
# 2. 精度测试与性能基准 (适配 TriMLU 协议)
# =============================================================================


def test_sin_correctness():
    print("🧪 正在进行 Sin 精度校验...")
    configs = [1024, 4096, 16384]
    for size in configs:
        x = torch.randn(size, device="mlu", dtype=torch.float32)

        # Triton 结果
        triton_out = sin_triton_wrapper(x)
        # Torch 结果
        torch_out = torch.sin(x)

        # 比较精度
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-5)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Size {size:5d}: {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_out - torch_out).abs().max()}")
            return False
    return True


def benchmark_sin():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")

    # 用于收集数据并转换成约定格式
    results_for_trimlu = []

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(12, 24, 2)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s",
            plot_name="sin-performance",
            args={},
        )
    )
    def benchmark(N, provider):
        x = torch.randn(N, device="mlu", dtype=torch.float32)

        if provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.sin(x))
        else:
            ms = triton.testing.do_bench(lambda: sin_triton_wrapper(x))
            # 将 Triton 的耗时记录下来用于 TriMLU 评估
            results_for_trimlu.append({"latency": ms, "n_elements": N})

        # 带宽计算: 读取 x (N*4) + 写入 out (N*4)
        gbps = lambda ms: (2 * N * 4) * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)
    return results_for_trimlu


if __name__ == "__main__":
    # 1. 精度校验
    passed = test_sin_correctness()

    if not passed:
        # 如果精度没过，打印空 JSON 强制拒绝优化
        print("__TRIMLU_PERF_JSON__:[]")
    else:
        # 2. 性能测试
        try:
            perf_data = benchmark_sin()
            # Orchestrator 通过识别此特定前缀来解析性能结果
            print(f"__TRIMLU_PERF_JSON__:{json.dumps(perf_data)}")
        except Exception as e:
            print(f"⚠️ 性能测试跳过或出错: {e}")
            print("__TRIMLU_PERF_JSON__:[]")
