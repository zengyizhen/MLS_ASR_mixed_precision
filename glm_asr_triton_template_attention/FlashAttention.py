"""
hw1-asr/glm_asr_triton_template/FlashAttention.py
Flash Attention implementation test - verifying correctness before integration
"""
import torch
import triton
import triton.language as tl
import numpy as np


ATTN_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _select_attention_compute_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in ATTN_LOW_PRECISION_DTYPES:
        return x.dtype
    if x.is_cuda and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _to_compute_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype and x.is_contiguous():
        return x
    return x.to(dtype=dtype).contiguous()


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale,
    seq_q, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    is_causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention 2: fused Q@K^T, softmax, @V in one kernel.
    Grid: (batch_heads, ceil(seq_q / BLOCK_Q))
    
    核心思想：online softmax，分块处理K/V，不把scores写回HBM
    """
    pid_bh = tl.program_id(0)   # batch * heads
    pid_q  = tl.program_id(1)   # Q block index

    # 当前 Q block 的行范围
    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,)
    offs_d = tl.arange(0, BLOCK_D)              # (BLOCK_D,)

    # 加载 Q block: (BLOCK_Q, BLOCK_D)
    q_ptrs = q_ptr + pid_bh * stride_q0 + offs_q[:, None] * stride_q1 + offs_d[None, :] * stride_q2
    q = tl.load(q_ptrs, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim), other=0.0)

    # 初始化 online softmax 状态
    # m: 每行当前最大值 (BLOCK_Q,)
    # l: 每行当前 exp 之和 (BLOCK_Q,)
    # acc: 累积输出 (BLOCK_Q, BLOCK_D)
    m = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # 遍历所有 K/V blocks
    num_k_blocks = tl.cdiv(seq_k, BLOCK_K)
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # 加载 K block: (BLOCK_K, BLOCK_D)
        k_ptrs = k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2
        k = tl.load(k_ptrs, mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)

        # 计算局部 scores: (BLOCK_Q, BLOCK_K)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # Causal mask: 只允许看到不超过自身位置的 token
        if is_causal:
            causal_mask = offs_q[:, None] >= offs_k[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Padding mask
        scores = tl.where(offs_k[None, :] < seq_k, scores, float("-inf"))

        # Online softmax 更新
        # 新的最大值
        m_new = tl.maximum(m, tl.max(scores, axis=1))  # (BLOCK_Q,)

        # 修正旧的累积值
        alpha = tl.exp(m - m_new)           # (BLOCK_Q,)  rescale factor
        l = l * alpha                        # 修正旧的 sum
        acc = acc * alpha[:, None]           # 修正旧的 output 累积

        # 当前 block 的 exp scores
        exp_scores = tl.exp(scores - m_new[:, None])  # (BLOCK_Q, BLOCK_K)
        l = l + tl.sum(exp_scores, axis=1)            # 累积新的 sum

        # 加载 V block: (BLOCK_K, BLOCK_D)
        v_ptrs = v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2
        v = tl.load(v_ptrs, mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)

        # 累积 weighted sum
        acc = acc + tl.dot(exp_scores.to(tl.float32), v.to(tl.float32), out_dtype=tl.float32)

        # 更新最大值
        m = m_new

    # 最终归一化
    acc = acc / l[:, None]

    # 写回输出
    out_ptrs = output_ptr + pid_bh * stride_o0 + offs_q[:, None] * stride_o1 + offs_d[None, :] * stride_o2
    tl.store(out_ptrs, acc, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim))


def flash_attention(q, k, v, is_causal=False, scale=None):
    """Flash attention wrapper."""
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    BLOCK_Q = 16
    BLOCK_K = 16
    BLOCK_D = triton.next_power_of_2(head_dim)

    compute_dtype = _select_attention_compute_dtype(q)
    output_dtype = q.dtype

    q_flat = _to_compute_dtype(q.reshape(batch * num_heads, seq_q, head_dim), compute_dtype)
    k_flat = _to_compute_dtype(k.reshape(batch * num_heads, seq_k, head_dim), compute_dtype)
    v_flat = _to_compute_dtype(v.reshape(batch * num_heads, seq_k, head_dim), compute_dtype)

    if head_dim != BLOCK_D:
        def pad_dim(x):
            padded = torch.zeros((x.shape[0], x.shape[1], BLOCK_D), dtype=compute_dtype, device=x.device)
            padded[:, :, :head_dim] = x
            return padded
        q_flat = pad_dim(q_flat)
        k_flat = pad_dim(k_flat)
        v_flat = pad_dim(v_flat)

    output = torch.zeros((batch * num_heads, seq_q, BLOCK_D), dtype=torch.float32, device=q.device)

    grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_Q))
    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, output,
        float(scale),
        seq_q, seq_k, head_dim,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        is_causal=is_causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
    output = output[:, :, :head_dim]
    return output.reshape(batch, num_heads, seq_q, head_dim).to(output_dtype)


def reference_attention(q, k, v, is_causal=False, scale=None):
    """Reference attention using PyTorch."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.einsum("bnqd,bnkd->bnqk", q.float(), k.float()) * scale
    if is_causal:
        seq_q, seq_k = q.shape[2], k.shape[2]
        mask = torch.triu(torch.ones(seq_q, seq_k, device=q.device), diagonal=1) * -1e9
        scores = scores + mask[None, None]
    scores = scores - scores.max(dim=-1, keepdim=True).values
    attn = torch.exp(scores) / torch.exp(scores).sum(dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", attn, v.float()).to(q.dtype)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    for seq_len, head_dim, desc in [
        (16, 64, "small (seq=16, dim=64)"),
        (64, 64, "medium (seq=64, dim=64)"),
        (128, 64, "large (seq=128, dim=64)"),
        (16, 128, "wide (seq=16, dim=128)"),
    ]:
        batch, heads = 2, 4
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        # Basic
        out_flash = flash_attention(q, k, v)
        out_ref   = reference_attention(q, k, v)
        diff = (out_flash - out_ref).abs().max().item()
        status = "✅" if diff < 1e-2 else "❌"
        print(f"{status} Basic    {desc}: max_diff={diff:.6f}")

        # Causal
        out_flash_c = flash_attention(q, k, v, is_causal=True)
        out_ref_c   = reference_attention(q, k, v, is_causal=True)
        diff_c = (out_flash_c - out_ref_c).abs().max().item()
        status_c = "✅" if diff_c < 1e-2 else "❌"
        print(f"{status_c} Causal   {desc}: max_diff={diff_c:.6f}")

    # Speed benchmark
    print("\n--- Speed Benchmark ---")
    q = torch.randn(2, 4, 128, 64, device=device)
    k = torch.randn(2, 4, 128, 64, device=device)
    v = torch.randn(2, 4, 128, 64, device=device)

    # Warmup
    for _ in range(10):
        flash_attention(q, k, v)
        reference_attention(q, k, v)
    torch.cuda.synchronize()

    import time
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        flash_attention(q, k, v)
    torch.cuda.synchronize()
    t_flash = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        reference_attention(q, k, v)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / N * 1000

    print(f"Flash Attention: {t_flash:.3f}ms")
    print(f"Reference:       {t_ref:.3f}ms")
    print(f"Speedup:         {t_ref/t_flash:.2f}x")
