import torch
import triton
import triton.language as tl


#### START KERNEL
@triton.jit
def embedding_kernel(
    weight,
    input_ids,
    out,
    vob_start_id,
    vob_end_id,
    stride_weight_seq,
    stride_out_seq,
    n_ctx,
    hiden_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N

    offs_nn = start_n + tl.arange(0, BLOCK_NN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=vob_end_id)
        id_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = token_ids - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(
            weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :],
            mask=load_mask,
            other=0.0,
        )
        tl.store(
            out + offs_seq[:, None] * stride_out_seq + offs_d[None, :],
            vecs,
            mask=store_mask,
        )


@torch.no_grad()
def embedding_wrapper(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id):
    n_ctx = input_ids.shape[0]
    embedding_dim = weight.shape[1]
    out = torch.zeros((n_ctx, embedding_dim), dtype=weight.dtype, device=weight.device)

    BLOCK_N = 64
    BLOCK_NN = 1
    BLOCK_DMODEL = triton.next_power_of_2(embedding_dim)

    grid = (triton.cdiv(n_ctx, BLOCK_N), 1, 1)

    embedding_kernel[grid](
        weight,
        input_ids,
        out,
        vob_start_id,
        vob_end_id,
        weight.stride(0),
        out.stride(0),
        n_ctx=n_ctx,
        hiden_size=embedding_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        num_warps=1,
        num_stages=1,
    )
    return out


#### END KERNEL

# =============================================================================
# 精度测试与性能基准
# =============================================================================


def test_embedding_correctness():
    print("🧪 正在进行 Embedding 精度校验...")
    vocab_size = 1000
    embedding_dim = 512
    vob_start_id = 10
    vob_end_id = 900

    configs = [128, 512, 1024]
    for seq_len in configs:
        input_ids = torch.randint(
            0, vocab_size + 10, (seq_len,), dtype=torch.int32, device="mlu"
        )
        weight = torch.randn(
            vob_end_id - vob_start_id, embedding_dim, dtype=torch.float32, device="mlu"
        )

        # Triton 结果
        triton_out = embedding_wrapper(input_ids, weight, vob_start_id, vob_end_id)

        # Torch 参考结果 (手动实现逻辑以匹配 vob 范围限制)
        torch_out = torch.zeros(
            (seq_len, embedding_dim), dtype=torch.float32, device="mlu"
        )
        mask = (input_ids >= vob_start_id) & (input_ids < vob_end_id)
        valid_ids = input_ids[mask] - vob_start_id
        if valid_ids.numel() > 0:
            torch_out[mask] = torch.nn.functional.embedding(valid_ids, weight)

        # 比较
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-5)
        status = "✅ 通过" if is_correct else "❌ 失败"
        print(f"  - Sequence Length {seq_len:4d}: {status}")
        if not is_correct:
            print(f"    Max Diff: {(triton_out - torch_out).abs().max()}")


def benchmark_embedding():
    print("\n🚀 正在进行性能基准测试 (Performance Benchmark)...")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[2**i for i in range(7, 15)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s",
            plot_name="embedding-performance",
            args={"D": 1024, "vob_start": 0, "vob_end": 10000},
        )
    )
    def benchmark(SEQ_LEN, D, vob_start, vob_end, provider):
        input_ids = torch.randint(
            vob_start, vob_end, (SEQ_LEN,), dtype=torch.int32, device="mlu"
        )
        weight = torch.randn(vob_end - vob_start, D, dtype=torch.float32, device="mlu")

        if provider == "torch":
            # 简化 Torch 版测试，仅包含核心 embedding
            ms = triton.testing.do_bench(
                lambda: torch.nn.functional.embedding(input_ids, weight)
            )
        if provider == "triton":
            ms = triton.testing.do_bench(
                lambda: embedding_wrapper(input_ids, weight, vob_start, vob_end)
            )

        # 带宽计算：读取 input_ids(4 bytes) + 读取 weight(D * 4 bytes) + 写入 out(D * 4 bytes)
        gbps = lambda ms: (SEQ_LEN * 4 + SEQ_LEN * D * 4 * 2) * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    # 1. 运行精度测试
    test_embedding_correctness()

    # 2. 运行性能测试
    try:
        benchmark_embedding()
    except Exception as e:
        print(f"⚠️ 性能测试跳过或出错: {e}")
