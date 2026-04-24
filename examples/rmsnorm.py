import torch
import triton
import triton.language as tl

torch.manual_seed(1234)

#### START KERNEL
@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_m,
    stride_x_k,
    stride_rms_w,
    stride_out_batch,
    stride_out_m,
    stride_out_k,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.extra.mlu.libdevice.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(
            x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0
        ).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = (
            pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        )
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)

def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[
        (
            batch,
            M,
        )
    ](
        x,
        rms_w,
        out,
        *x.stride(),
        *rms_w.stride(),
        *out.stride(),
        N_SIZE=K,
        eps=eps,
        BLOCK_N_SIZE=1024,
    )
    return out
#### END KERNEL

# =============================================================================
# 精度测试与性能基准
# =============================================================================

def test_rmsnorm_correctness():
    print("🧪 正在进行 RMSNorm 精度校验...")
    configs = [
        (2, 4, 1024),
        (4, 8, 2048),
        (1, 16, 4096),
    ]
    for B, M, K in configs:
        x = torch.randn((B, M, K), dtype=torch.float16, device="mlu")
        rms_w = torch.randn((K,), dtype=torch.float16, device="mlu")
        eps = 1e-6
        
        # Triton 结果
        triton_out = rmsnorm_triton_wrapper(x, rms_w, eps)
        
        # Torch 参考结果
        # RMSNorm 逻辑: x / sqrt(mean(x^2) + eps) * weight
        x_f32 = x.to(torch.float32)
        rms_w_f32 = rms_w.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + eps)
        torch_out = (x_f32 / rms * rms_w_f32).to(torch.float16)
        
        # 比较精度
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Shape ({B}, {M}, {K}): {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_out - torch_out).abs().max()}")

def benchmark_rmsnorm():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['K'],
            x_vals=[1024 * i for i in range(1, 9)], 
            line_arg='provider', 
            line_vals=['triton', 'torch'], 
            line_names=['Triton', 'Torch'], 
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s', 
            plot_name='rmsnorm-performance',
            args={'B': 4, 'M': 64}, 
        )
    )
    def benchmark(K, B, M, provider):
        x = torch.randn((B, M, K), device='mlu', dtype=torch.float16)
        rms_w = torch.randn((K,), device='mlu', dtype=torch.float16)
        
        # 手动定义 Torch 版 Benchmark 逻辑
        def torch_rmsnorm(x, w, eps=1e-6):
            rms = torch.sqrt(torch.mean(x.to(torch.float32)**2, dim=-1, keepdim=True) + eps)
            return (x.to(torch.float32) / rms * w.to(torch.float32)).to(torch.float16)

        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, rms_w))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: rmsnorm_triton_wrapper(x, rms_w))
        
        # 带宽计算: 读取 x(B*M*K*2) + 读取 w(K*2) + 写入 out(B*M*K*2)
        gbps = lambda ms: (B * M * K * 2 * 2 + K * 2) * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)

if __name__ == "__main__":
    # 1. 精度校验
    test_rmsnorm_correctness()
    
    # 2. 性能测试
    try:
        benchmark_rmsnorm()
    except Exception as e:
        print(f"⚠️ 性能测试跳过或出错: {e}")
