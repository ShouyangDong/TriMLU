import os
import torch
import triton
import triton.language as tl
import triton.backends.mlu.driver as driver
import pytest
from rope_gpu import rope_impl as rope_impl_gpu

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CLUSTER_NUM = _devprob.get("cluster_num")
TOTAL_CORE_NUM = TOTAL_CLUSTER_NUM * _devprob.get("core_num_per_cluster")


# ---------------------------------------------------------------------------
# Optimized fold-style RoPE kernel (from rotary_embedding.py)
# Processes input shape (bs, seqlen, head_num, head_size).
# cos/sin table: (total_tokens, head_size), indexed via token_offsets.
# ---------------------------------------------------------------------------
#### START KERNEL
configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [512]
    for s in [3]
    for w in [1]
]

if os.environ.get("TRITON_DEBUG") == "1":
    configs = [triton.Config({"BLOCK_M": 4}, num_stages=1, num_warps=1)]


@triton.autotune(configs=configs, key=["DIM", "REVERSE"])
@triton.jit
def _apply_fold_rotary_kernel(
    # data pointers
    output,
    input,
    cos_emb,
    sin_emb,
    token_offsets,
    # dims
    bs,
    seqlen,
    head_num,
    head_size,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_headnum,
    stride_out_headsize,
    stride_input_batch,
    stride_input_seqlen,
    stride_input_headnum,
    stride_input_headsize,
    # meta parameters
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1)
    m_begin = pid_m * BLOCK_M

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    in_out_k_offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = m_offsets < seqlen

    ro_offsets = tl.load(
        token_offsets + pid_bs * seqlen + m_offsets, mask=mask, other=0
    )

    cos_begin = cos_emb + ro_offsets * head_size
    sin_begin = sin_emb + ro_offsets * head_size
    cos = tl.load(cos_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)

    input_offsets = (
        m_offsets * stride_input_seqlen + in_out_k_offsets * stride_input_headsize
    )
    for pid_head in range(head_num):
        input_ptr = (
            input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        )
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

        # Fold rotation: pair first half [0:K/2] with second half [K/2:K]
        # o = [-x2, x1]  =>  out = x*cos + o*sin
        x = tl.view(x, (BLOCK_M, 2, BLOCK_K // 2))
        o = tl.empty([BLOCK_M, 2, BLOCK_K // 2], dtype=tl.float32)
        o[:, 0, :] = -x[:, 1, :]
        o[:, 1, :] = x[:, 0, :]
        x = tl.view(x, (BLOCK_M, BLOCK_K))
        o = tl.view(o, (BLOCK_M, BLOCK_K))
        x = x * cos + o * sin

        out_offsets = (
            m_offsets * stride_out_seqlen + in_out_k_offsets * stride_out_headsize
        )
        output_ptr = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        tl.store(output_ptr + out_offsets, x, mask=mask)


# ---------------------------------------------------------------------------
# Helper: build per-token cos/sin table from position ids and base frequency.
#
# For the fold kernel the table must have full head_size columns.
# Standard fold RoPE pairs x[0:D/2] with x[D/2:D], so the frequency for
# dimension i (0-indexed, 0 <= i < D) is:
#   freq_i = base^(-2*(i % (D/2)) / D)
# which is equivalent to tiling the D/2 base frequencies twice.
# ---------------------------------------------------------------------------


def _build_cos_sin_table(
    position: torch.Tensor,  # (total_tokens,) int / long
    base: float,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos, sin tables of shape (total_tokens, dim)."""
    half = dim // 2
    # inv_freq: (half,)
    inv_freq = base ** (
        -2.0 * torch.arange(0, half, dtype=torch.float32, device=position.device) / dim
    )
    # freqs: (total_tokens, half)
    freqs = position.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos_half = torch.cos(freqs)  # (total_tokens, half)
    sin_half = torch.sin(freqs)

    # Tile to full dim so both halves of the head share the same frequencies
    cos = torch.cat([cos_half, cos_half], dim=-1)  # (total_tokens, dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def rope_impl(
    input: torch.Tensor,  # (total_len, head, dim)
    position: torch.Tensor,  # (total_len,)  position ids
    offset: torch.Tensor,  # (bs+1,)        cu_seqlens
    max_len: int,
    base: float = 10000.0,
    reverse: bool = False,
) -> torch.Tensor:
    total_len, head, dim = input.shape

    # 1. Pre-compute cos/sin table indexed by token position
    cos, sin = _build_cos_sin_table(position, base, dim)
    if reverse:
        sin = -sin

    # Keep cos/sin in fp32 to avoid quantization error during training.
    # The kernel casts them back to fp32 internally via .to(tl.float32),
    # so there is no dtype mismatch — but if we cast here first the
    # precision loss is permanent and cannot be recovered inside the kernel.

    # 2. Treat the whole packed sequence as a single "batch" of length total_len.
    #    token_offsets[0, i] = i  (cos/sin table is already in token order)
    token_offsets = torch.arange(
        total_len, device=input.device, dtype=torch.int32
    ).unsqueeze(
        0
    )  # (1, total_len)

    # 3. Reshape to (bs=1, seqlen=total_len, head, dim) for the kernel
    input_4d = input.unsqueeze(0).contiguous()  # (1, total_len, head, dim)
    out_4d = torch.empty_like(input_4d)

    BLOCK_K = dim
    bs_val = 1
    seqlen_val = total_len
    grid = lambda META: (triton.cdiv(seqlen_val, META["BLOCK_M"]), bs_val)

    # print("shape:", input_4d.shape)

    _apply_fold_rotary_kernel[grid](
        # data pointers
        out_4d,
        input_4d,
        cos,
        sin,
        token_offsets,
        # dims
        bs_val,
        seqlen_val,
        head,
        dim,
        # strides for output  (bs, seqlen, head, dim)
        out_4d.stride(0),
        out_4d.stride(1),
        out_4d.stride(2),
        out_4d.stride(3),
        # strides for input
        input_4d.stride(0),
        input_4d.stride(1),
        input_4d.stride(2),
        input_4d.stride(3),
        # meta
        BLOCK_K,
    )

    # Restore original shape (total_len, head, dim)
    return out_4d.squeeze(0)
#### END KERNEL

# ---------------------------------------------------------------------------
# Public autograd Function — interface unchanged from original rope.py
# ---------------------------------------------------------------------------


class RopeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, position, offset, max_len, base=10000.0, reverse=False):
        ctx.save_for_backward(position, offset)
        ctx.max_len = max_len
        ctx.base = base
        ctx.reverse = reverse
        return rope_impl(input, position, offset, max_len, base, reverse)

    @staticmethod
    def backward(ctx, do):
        position, offset = ctx.saved_tensors
        # Backward is the inverse rotation (negate sin / flip sign of reverse)
        grad_input = rope_impl(
            do, position, offset, ctx.max_len, ctx.base, not ctx.reverse
        )
        return grad_input, None, None, None, None, None
