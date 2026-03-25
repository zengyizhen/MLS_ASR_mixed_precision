import torch
import numpy as np
from layers import Linear, linear_kernel_tf32

def benchmark_op(func, num_warmup=50, num_iters=500):
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


def forward_triton_unfused(linear, x):
    """手动模拟融合前：triton GEMM + 单独 bias 加法"""
    from layers import pad_to_multiple
    original_shape = x.shape
    batch_dims = original_shape[:-1]
    M = int(np.prod(batch_dims))
    K = linear.in_features
    N = linear.out_features

    x_2d = x.reshape(M, K).to(torch.float32).contiguous()
    linear._ensure_weight_prepared()

    M_padded = pad_to_multiple(M, linear.TILE_M)
    if M_padded > M or linear._K_padded > K:
        x_padded = torch.zeros((M_padded, linear._K_padded), dtype=torch.float32, device=x.device)
        x_padded[:M, :K] = x_2d
    else:
        x_padded = x_2d

    output = torch.zeros((M_padded, linear._N_padded), dtype=torch.float32, device=x.device)
    dummy_bias = output.new_empty(0)

    grid = (
        __import__('triton').cdiv(M_padded, linear.TILE_M),
        __import__('triton').cdiv(linear._N_padded, linear.TILE_N),
    )
    linear_kernel_tf32[grid](
        x_padded,
        linear._weight_t_padded,
        output,
        dummy_bias,                  # 占位
        M_padded, linear._N_padded, linear._K_padded,
        x_padded.stride(0),          x_padded.stride(1),
        linear._weight_t_padded.stride(0), linear._weight_t_padded.stride(1),
        output.stride(0),            output.stride(1),
        HAS_BIAS=False,              # 不融合 bias
        BLOCK_M=linear.TILE_M,
        BLOCK_N=linear.TILE_N,
        BLOCK_K=linear.TILE_K,
    )

    output = output[:M, :N]
    output = output + linear.bias_param   # 单独的 elementwise kernel
    return output.reshape(*batch_dims, N)


if __name__ == "__main__":
    device = torch.device("cuda")

    shapes = [
        ("Audio qkv proj", 1280, 1280),
        ("Audio fc1",      1280, 5120),
        ("Audio fc2",      5120, 1280),
    ]

    for name, K, N in shapes:
        print(f"\n=== {name}  ({K} -> {N}) ===")

        linear = Linear(K, N, bias=True)
        linear.weight     = linear.weight.to(device)
        linear.bias_param = linear.bias_param.to(device)
        linear._ensure_weight_prepared()

        # ---- 正确性验证 ----
        torch.manual_seed(42)
        x_check = torch.randn(2, 16, K, device=device)

        Linear.BACKEND = "triton"
        out_fused   = linear(x_check)
        out_unfused = forward_triton_unfused(linear, x_check)

        assert torch.allclose(out_fused, out_unfused, atol=1e-5), \
            f"❌ 结果不一致！max diff = {(out_fused - out_unfused).abs().max():.6f}"
        print("✅ 正确性验证通过")

        # ---- Prefill: B=1, S=512 ----
        x_prefill = torch.randn(1, 512, K, device=device)

        Linear.BACKEND = "triton"
        t_fused   = benchmark_op(lambda: linear(x_prefill))
        t_unfused = benchmark_op(lambda: forward_triton_unfused(linear, x_prefill))

        print(f"Prefill (S=512) | unfused: {t_unfused:.4f} ms  fused: {t_fused:.4f} ms  speedup: {t_unfused/t_fused:.2f}x")

        # ---- Decode: B=1, S=1 ----
        x_decode = torch.randn(1, 1, K, device=device)

        t_fused_d   = benchmark_op(lambda: linear(x_decode))
        t_unfused_d = benchmark_op(lambda: forward_triton_unfused(linear, x_decode))

        print(f"Decode  (S=1)   | unfused: {t_unfused_d:.4f} ms  fused: {t_fused_d:.4f} ms  speedup: {t_unfused_d/t_fused_d:.2f}x")