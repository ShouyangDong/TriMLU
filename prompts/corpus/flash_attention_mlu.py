import os
import torch
import triton
import triton.language as tl
import torch.nn as nn
import numpy as np
import os, sys, random
import pytest

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [64]
    for s in [4]
    for w in [1]
]

bwd_preprocess_configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [128]
    for s in [4]
    for w in [1]
]


@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
    elif EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0,))
    elif EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1,))
    else:
        tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def apply_mask(s, q_mask, k_mask, offset_m, offset_n, TYPE: tl.constexpr):
    if TYPE == 1:
        causal_mask = offset_m[None, :] <= offset_n[:, None]
    elif TYPE == 2:
        causal_mask = offset_m[None, :] >= offset_n[:, None]
    k_nonzero_mask = k_mask != 0
    indices = tl.arange(0, k_mask.shape[0])
    k_nonzero_idx, k_nonzero_count = tl.masked_select(indices, k_nonzero_mask)
    if k_nonzero_count <= 6:
        for i in tl.range(k_nonzero_count, num_stages=1):
            k_pos = k_nonzero_idx[i]
            k_val = k_mask[k_pos]
            q_compare = q_mask == k_val
            diagonal_mask = offset_m == offset_n[k_pos]
            mask = q_compare | diagonal_mask
            new_row = causal_mask[k_pos, :] * mask
            row_selector = indices[:, None] == k_pos
            causal_mask = tl.where(row_selector, new_row[None, :], causal_mask)
        mask = causal_mask
    else:
        mask = (
            causal_mask
            & ((q_mask[None, :] == k_mask[:, None]) | (k_mask[:, None] == 0))
        ) | (offset_m[None, :] == offset_n[:, None])

    mask = 1.0 - mask.to(s.dtype)
    mask = mask * (-(2**30))
    s = s + mask
    return s


@triton.autotune(configs, key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
@triton.jit
def fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    q_mask_ptr,
    k_mask_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start

    if SPARSE_OPT:
        begin = 0
        end = k_len
    else:
        if MASK_FN & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(QK_DIM, q_len),
        strides=(1, q_head * QK_DIM),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(QK_DIM, BLOCK_M),
        order=(0, 1),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(k_len, QK_DIM),
        strides=(kv_head * QK_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_N, QK_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(V_DIM, k_len),
        strides=(1, kv_head * V_DIM),
        offsets=(0, begin),
        block_shape=(V_DIM, BLOCK_N),
        order=(0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_mask_block_ptr = tl.make_block_ptr(
        base=q_mask_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_mask_block_ptr = tl.make_block_ptr(
        base=k_mask_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    acc = tl.zeros((V_DIM, BLOCK_M), dtype=tl.float32)
    m = tl.full((BLOCK_M,), value=-(2**30), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    q = load_if(q_block_ptr, True, False)
    q_mask = load_if(q_mask_block_ptr, False, True)

    for start_n in range(begin, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = load_if(k_mask_block_ptr, False, True)
        offset_n = start_n + tl.arange(0, BLOCK_N)

        k = load_if(k_block_ptr, False, True)
        s = tl.dot(k, q)
        boundary_mask = (offset_n < k_len)[:, None]
        s = apply_mask(s, q_mask, k_mask, offset_m, offset_n, MASK_FN)
        s = tl.where(boundary_mask, s, -(2**30))

        m_new = tl.maximum(m, tl.max(s, 0))
        p = tl.math.exp2((s - m_new[None, :]) * qk_scale)
        p_sum = tl.sum(p, 0)
        alpha = tl.math.exp2((m - m_new) * qk_scale)
        v = load_if(v_block_ptr, True, False)
        acc = acc * alpha[None, :]
        acc += tl.dot(v, p.to(dtype))
        l = l * alpha + p_sum
        m = m_new
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (0, BLOCK_N))
        k_mask_block_ptr = tl.advance(k_mask_block_ptr, (BLOCK_N,))

    acc = acc / l[None, :]
    acc = tl.trans(acc)
    l = m * scale + tl.log(l)

    store_if(o_block_ptr, acc.to(dtype), False, True)
    store_if(l_block_ptr, l, False, True)


@triton.autotune(bwd_preprocess_configs, key=["V_DIM"])
@triton.jit
def bwd_preprocess(
    o_ptr,
    do_ptr,
    d_ptr,
    cu_seqlens_q,
    q_head: tl.constexpr,  # 总 head 数，同时作为 constexpr 用于循环和指针计算
    V_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)  # block 索引
    start_b = tl.program_id(1)  # batch 索引

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    q_start = q_start.to(tl.int64)
    # 遍历所有 head
    for h in range(q_head):
        # 为当前 head 创建 o 的 block_ptr
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + q_start * q_head * V_DIM + h * V_DIM,
            shape=(q_len, V_DIM),
            strides=(q_head * V_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, V_DIM),
            order=(1, 0),
        )
        # 为当前 head 创建 do 的 block_ptr
        do_block_ptr = tl.make_block_ptr(
            base=do_ptr + q_start * q_head * V_DIM + h * V_DIM,
            shape=(q_len, V_DIM),
            strides=(q_head * V_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, V_DIM),
            order=(1, 0),
        )
        # 为当前 head 创建 d 的 block_ptr
        d_block_ptr = tl.make_block_ptr(
            base=d_ptr + q_start * q_head + h,
            shape=(q_len,),
            strides=(q_head,),
            offsets=(start_m * BLOCK_M,),
            block_shape=(BLOCK_M,),
            order=(0,),
        )

        o = load_if(o_block_ptr, False, True).to(tl.float32)
        do = load_if(do_block_ptr, False, True).to(tl.float32)
        d = tl.sum(o * do, axis=1)  # 沿 V_DIM 求和，得到每个 token 的标量
        store_if(d_block_ptr, d, False, True)


@triton.autotune(configs, key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
@triton.jit
def bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    q_mask_ptr,
    k_mask_ptr,
    total_q_len,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = k_ptr.type.element_ty
    start_n = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start
    if start_n * BLOCK_N >= k_len:
        return

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start

    if SPARSE_OPT:
        begin = 0
        end = q_len
    else:
        if MASK_FN & 1:
            begin = 0
            end = tl.minimum(start_n * BLOCK_N // BLOCK_M * BLOCK_M + BLOCK_M, q_len)
        else:
            begin = start_n * BLOCK_N // BLOCK_M * BLOCK_M
            end = q_len

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)

    # Block pointers for KV computation
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(k_len, QK_DIM),
        strides=(kv_head * QK_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, QK_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(k_len, V_DIM),
        strides=(kv_head * V_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, V_DIM),
        order=(1, 0),
    )
    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr + k_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(k_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, QK_DIM),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dv_ptr + k_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(k_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, V_DIM),
        order=(1, 0),
    )

    # Block pointers for Q computation
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    base = dq_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM
    offset_m1 = begin + tl.arange(0, BLOCK_M)
    offset_qk = tl.arange(0, QK_DIM)
    dq_block_ptr = base + offset_m1[:, None] * q_head * QK_DIM + offset_qk[None, :]

    # Common block pointers
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start + start_qh * total_q_len,
        shape=(q_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + q_start + start_qh * total_q_len,
        shape=(q_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_mask_block_ptr = tl.make_block_ptr(
        base=q_mask_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_mask_block_ptr = tl.make_block_ptr(
        base=k_mask_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(start_n * BLOCK_N,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    # Initialize accumulators
    dk = tl.zeros((BLOCK_N, QK_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, V_DIM), dtype=tl.float32)
    dq = tl.zeros((BLOCK_M, QK_DIM), dtype=tl.float32)

    # Load K and V for this block
    k = load_if(k_block_ptr, False, True)
    v = load_if(v_block_ptr, False, True)
    k_mask = load_if(k_mask_block_ptr, False, True)

    for start_m in range(begin, end, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        q_mask = load_if(q_mask_block_ptr, False, True)
        offset_m = start_m + tl.arange(0, BLOCK_M)

        # Load Q, DO, L, D for this block
        q = load_if(q_block_ptr, False, True)
        do = load_if(do_block_ptr, False, True)
        l = load_if(l_block_ptr, False, True)
        d = load_if(d_block_ptr, False, True)

        # Compute attention scores
        s = tl.dot(k, tl.trans(q))
        boundary_mask = (offset_n < k_len)[:, None]
        s = apply_mask(s, q_mask, k_mask, offset_m, offset_n, MASK_FN)
        s = tl.where(boundary_mask, s, -(2**30))
        p = tl.math.exp2(s * qk_scale - l[None, :] * log2e)

        # Compute dv (p @ do)
        dv += tl.dot(p.to(dtype), do)  # [BLOCK_N, V_DIM]

        dp = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - d[None, :]
        # Compute dp (do @ v^T)
        dp += tl.dot(v, tl.trans(do))  # [BLOCK_M, BLOCK_N]

        # Compute ds
        ds = p * dp  # [BLOCK_M, BLOCK_N]

        # Compute dk (ds @ q)
        dk += tl.dot(ds.to(dtype), q)  # [BLOCK_N, QK_DIM]

        # Compute dq (ds^T @ k)
        dq = tl.dot(tl.trans(ds).to(dtype), k) * scale  # [BLOCK_M, QK_DIM]
        tl.atomic_add(dq_block_ptr, dq.to(dtype), mask=(offset_m < q_len)[:, None])
        dq_block_ptr += BLOCK_M * q_head * QK_DIM
        # Advance pointers
        q_block_ptr = tl.advance(q_block_ptr, (BLOCK_M, 0))
        do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M, 0))
        l_block_ptr = tl.advance(l_block_ptr, (BLOCK_M,))
        d_block_ptr = tl.advance(d_block_ptr, (BLOCK_M,))
        q_mask_block_ptr = tl.advance(q_mask_block_ptr, (BLOCK_M,))

    # Scale and store results
    dk *= scale

    store_if(dk_block_ptr, dk.to(dtype), False, True)
    store_if(dv_block_ptr, dv.to(dtype), False, True)


class FlashAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_mask,
        k_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        mask_fn,
        sparse_opt,
    ):
        q_len, q_head, qk_dim = q.shape
        k_len, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)
        # with torch.mlu.device(q.device.index):
        if 1:
            grid = lambda META: (
                triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
                q_head,
                batch_size,
            )
            fwd_kernel[grid](
                q,
                k,
                v,
                o,
                l,
                q_mask,
                k_mask,
                cu_seqlens_q,
                cu_seqlens_k,
                q_head,
                kv_head,
                scale,
                QK_DIM=qk_dim,
                V_DIM=v_dim,
                MASK_FN=mask_fn,
                SPARSE_OPT=sparse_opt,
                DTYPE=(19 if q.dtype == torch.float16 else 14),
            )
        ctx.save_for_backward(q, k, v, o, l, q_mask, k_mask, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scale = scale
        ctx.mask_fn = mask_fn
        ctx.sparse_opt = sparse_opt
        ctx.k_len = k_len
        ctx.q_len = q_len
        ctx.q_head = q_head
        ctx.kv_head = kv_head
        ctx.qk_dim = qk_dim
        ctx.v_dim = v_dim
        ctx.batch_size = batch_size
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dtype = 19 if q.dtype == torch.float16 else 14
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, q_mask, k_mask, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = k.new_empty(ctx.k_len, ctx.q_head, ctx.qk_dim)
        dv = v.new_empty(ctx.k_len, ctx.q_head, ctx.v_dim)
        d = torch.empty_like(l)
        q_len, q_head, qk_dim = q.shape
        # with torch.mlu.device(do.device.index):
        if 1:
            grid = lambda META: (
                triton.cdiv(ctx.max_seqlen_q, META["BLOCK_M"]),
                ctx.batch_size,
            )
            bwd_preprocess[grid](
                o,
                do,
                d,
                cu_seqlens_q,
                ctx.q_head,
                V_DIM=ctx.v_dim,
                DTYPE=ctx.dtype,
            )
            l = torch.transpose(l, 0, 1).contiguous()
            d = torch.transpose(d, 0, 1).contiguous()

            grid = lambda META: (
                triton.cdiv(ctx.max_seqlen_k, META["BLOCK_N"]),
                ctx.q_head,
                ctx.batch_size,
            )
            bwd_kernel[grid](
                q,
                k,
                v,
                dq,
                dk,
                dv,
                do,
                l,
                d,
                q_mask,
                k_mask,
                ctx.q_len,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.q_head,
                ctx.kv_head,
                ctx.scale,
                QK_DIM=ctx.qk_dim,
                V_DIM=ctx.v_dim,
                MASK_FN=ctx.mask_fn,
                SPARSE_OPT=ctx.sparse_opt,
                DTYPE=ctx.dtype,
            )
            head_group = ctx.q_head // ctx.kv_head
            if head_group > 1:
                dk = dk.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.qk_dim).sum(2)
                dv = dv.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.v_dim).sum(2)
        return (dq, dk, dv) + (None,) * 9
