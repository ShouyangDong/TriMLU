import triton
import torch
import triton.language as tl
import time
from typing import Tuple, List


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 1280,
            },
            num_stages=8,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 640,
            },
            num_stages=1,
            num_warps=1,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def channel_granul_fp8_quant_kernel_m(
    x_ptr,
    y_ptr,
    scale_ptr,
    B,
    stride_x_b,
    stride_x_m,
    stride_x_n,
    stride_y_b,
    stride_y_m,
    stride_y_n,
    stride_s_b,
    stride_s_m,
    stride_s_n,
    M,
    N,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X_ROW_MAJOR: tl.constexpr,
    Y_ROW_MAJOR: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)  # 现在是 batch 索引，范围 [0, min(16, B))

    # 计算每个 program 需要处理多少个 batch
    num_programs = tl.num_programs(0)  # min(16, B)
    batches_per_program = tl.cdiv(B, num_programs)

    # 计算当前 program 负责的 batch 范围
    batch_start = pid * batches_per_program
    batch_end = tl.minimum(batch_start + batches_per_program, B)

    # 遍历当前 program 负责的所有 batches
    for b in range(batch_start, batch_end):
        x_base = x_ptr + b * stride_x_b
        y_base = y_ptr + b * stride_y_b

        # 遍历所有 N blocks
        num_n_blocks = tl.cdiv(N, BLOCK_N)
        for bid in range(num_n_blocks):
            n0 = bid * BLOCK_N
            amax_n = tl.zeros((BLOCK_N,), dtype=tl.float32)
            x_bp = tl.make_block_ptr(
                base=x_base,
                shape=(M, N),
                strides=(stride_x_m, stride_x_n),
                offsets=(0, n0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0) if X_ROW_MAJOR else (0, 1),
            )
            x_bp1 = x_bp
            for _ in range(0, tl.cdiv(M, BLOCK_M)):
                x_tile = tl.load(x_bp1, boundary_check=(0, 1), padding_option="zero")
                x_abs = tl.abs(x_tile)
                amax_n = tl.maximum(amax_n, tl.max(x_abs, axis=0))
                x_bp1 = tl.advance(x_bp1, (BLOCK_M, 0))
            safe_amax = tl.maximum(amax_n, EPS)
            scale_n = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
            reciprocal_scale_n = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)
            scale_bp = tl.make_block_ptr(
                base=scale_ptr + b * stride_s_b,
                shape=(1, N),
                strides=(stride_s_m, stride_s_n),
                offsets=(0, n0),
                block_shape=(1, BLOCK_N),
                order=(1, 0),
            )
            tl.store(scale_bp, reciprocal_scale_n[None, :], boundary_check=(1,))
            x_bp2 = x_bp
            y_bp2 = tl.make_block_ptr(
                base=y_base,
                shape=(M, N),
                strides=(stride_y_m, stride_y_n),
                offsets=(0, n0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0) if Y_ROW_MAJOR else (0, 1),
            )
            for _ in range(0, tl.cdiv(M, BLOCK_M)):
                x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option="zero")
                y_tile = tl.cast(x_tile, tl.float32) * scale_n[None, :]
                y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
                y_fp8 = y_tile.to(y_ptr.dtype.element_ty)
                tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
                x_bp2 = tl.advance(x_bp2, (BLOCK_M, 0))
                y_bp2 = tl.advance(y_bp2, (BLOCK_M, 0))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 16,
            },
            num_stages=8,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
            },
            num_stages=4,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
            },
            num_stages=1,
            num_warps=1,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def channel_granul_fp8_quant_kernel_n(
    x_ptr,
    y_ptr,
    scale_ptr,
    B,
    stride_x_b,
    stride_x_m,
    stride_x_n,
    stride_y_b,
    stride_y_m,
    stride_y_n,
    stride_s_b,
    stride_s_m,
    stride_s_n,
    M,
    N,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X_ROW_MAJOR: tl.constexpr,
    Y_ROW_MAJOR: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)  # 现在是 batch 索引，范围 [0, min(16, B))

    # 计算每个 program 需要处理多少个 batch
    num_programs = tl.num_programs(0)  # min(16, B)
    batches_per_program = tl.cdiv(B, num_programs)

    # 计算当前 program 负责的 batch 范围
    batch_start = pid * batches_per_program
    batch_end = tl.minimum(batch_start + batches_per_program, B)

    # 遍历当前 program 负责的所有 batches
    for b in range(batch_start, batch_end):
        x_base = x_ptr + b * stride_x_b
        y_base = y_ptr + b * stride_y_b

        # 快速路径：一次性加载整个 N 维度
        num_m_blocks = tl.cdiv(M, BLOCK_M)
        for bid in range(num_m_blocks):
            m0 = bid * BLOCK_M

            # 使用 BLOCK_N 作为 block_shape，但实际只处理 N 个元素
            x_bp = tl.make_block_ptr(
                base=x_base,
                shape=(M, N),
                strides=(stride_x_m, stride_x_n),
                offsets=(m0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0) if X_ROW_MAJOR else (0, 1),
            )
            x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")

            # 直接在 N 维度求 max (实际只处理前 N 列)
            x_abs = tl.abs(x_tile)
            amax_m = tl.max(x_abs, axis=1)  # [BLOCK_M,]

            # 计算 scale
            safe_amax = tl.maximum(amax_m, EPS)
            scale_m = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
            reciprocal_scale_m = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)

            # 存储 scale
            scale_bp = tl.make_block_ptr(
                base=scale_ptr + b * stride_s_b,
                shape=(M, 1),
                strides=(stride_s_m, stride_s_n),
                offsets=(m0, 0),
                block_shape=(BLOCK_M, 1),
                order=(1, 0),
            )
            tl.store(scale_bp, reciprocal_scale_m[:, None], boundary_check=(0,))

            # 量化并存储
            y_tile = tl.cast(x_tile, tl.float32) * scale_m[:, None]
            # TODO(jyj): use libdevice when triton update
            # y_fp8 = tl.extra.mlu.libdevice.float2fp8(y_tile)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = y_tile.to(y_ptr.dtype.element_ty)

            y_bp = tl.make_block_ptr(
                base=y_base,
                shape=(M, N),
                strides=(stride_y_m, stride_y_n),
                offsets=(m0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0) if Y_ROW_MAJOR else (0, 1),
            )
            tl.store(y_bp, y_fp8, boundary_check=(0, 1))


def channel_granul_fp8_quant(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
    output_row_major: bool = True,
    scale_tol: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton 实现的 FP8 通道粒度量化"""
    assert x.dim() == 3, "only support 3D tensor now"
    if axiswise_dim not in (-1, -2):
        raise ValueError("axiswise_dim must be -1 or -2")
    reduce_along_n = True if axiswise_dim == -1 else False

    x = x.contiguous()
    x_row_major = x.is_contiguous()

    B, M, N = x.shape
    # print("JYJ quant", B, M, N, axiswise_dim, x_row_major)
    if output_row_major:
        y = torch.empty((B, M, N), device=x.device, dtype=float8_dtype)
    else:
        y = torch.empty((B, N, M), device=x.device, dtype=float8_dtype)
        y = y.transpose(-1, -2)
    reciprocal_scale = torch.empty(
        (B, M, 1) if reduce_along_n else (B, 1, N), dtype=torch.float32, device=x.device
    )
    stride_scale_b, stride_scale_m, stride_scale_n = reciprocal_scale.stride()

    fp8_max = float(torch.finfo(float8_dtype).max)

    # 修改 grid 为 (min(16, B), 1, 1)
    grid = (min(16, B),)

    if reduce_along_n:
        channel_granul_fp8_quant_kernel_n[grid](
            x,
            y,
            reciprocal_scale,
            B,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            stride_scale_b,
            stride_scale_m,
            stride_scale_n,
            M,
            N,
            fp8_max=fp8_max,
            X_ROW_MAJOR=x_row_major,
            Y_ROW_MAJOR=output_row_major,
            EPS=scale_tol,
            BLOCK_N=N,
        )
    else:
        channel_granul_fp8_quant_kernel_m[grid](
            x,
            y,
            reciprocal_scale,
            B,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            stride_scale_b,
            stride_scale_m,
            stride_scale_n,
            M,
            N,
            fp8_max=fp8_max,
            X_ROW_MAJOR=x_row_major,
            Y_ROW_MAJOR=output_row_major,
            EPS=scale_tol,
        )

    return y, reciprocal_scale
