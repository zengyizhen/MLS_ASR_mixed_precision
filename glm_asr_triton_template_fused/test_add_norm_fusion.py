import torch
from layers import RMSNorm

def benchmark_op(func, num_warmup=50, num_iters=200):
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters

if __name__ == "__main__":
    device = torch.device("cuda")

    # ==========================================
    # Part 1: 模拟 decode 阶段 (B=1, S=1)
    # ==========================================
    B, S, H = 1, 1, 4096
    NUM_LAYERS = 28
    print(f"=== Part 1: 模拟 decode 阶段: B={B}, S={S}, H={H}, {NUM_LAYERS} 层 ===")

    norms = [RMSNorm(hidden_size=H, eps=1e-5) for _ in range(NUM_LAYERS)]
    for n in norms:
        n.weight = n.weight.to(device)
    deltas = [torch.randn(B, S, H, device=device, dtype=torch.float32) * 0.1
              for _ in range(NUM_LAYERS)]

    # 正确性验证
    torch.manual_seed(42)
    h = torch.randn(B, S, H, device=device)

    h_unfused = h.clone()
    outputs_unfused = []
    for i in range(NUM_LAYERS):
        h_unfused = h_unfused + deltas[i]
        outputs_unfused.append(norms[i](h_unfused))

    res_fused = h.clone()
    outputs_fused = []
    for i in range(NUM_LAYERS):
        out = norms[i](deltas[i], residual=res_fused)
        outputs_fused.append(out)

    for i in range(NUM_LAYERS):
        assert torch.allclose(outputs_unfused[i], outputs_fused[i], atol=1e-4), f"第{i}层不一致"
    print("✅ 正确性验证通过\n")

    def run_unfused_decode():
        hidden = torch.randn(B, S, H, device=device, dtype=torch.float32)
        for i in range(NUM_LAYERS):
            hidden = hidden + deltas[i]
            normed = norms[i](hidden)
        return normed

    def run_fused_decode():
        residual = torch.randn(B, S, H, device=device, dtype=torch.float32)
        for i in range(NUM_LAYERS):
            normed = norms[i](deltas[i], residual=residual)
        return normed

    t1 = benchmark_op(run_unfused_decode)
    t2 = benchmark_op(run_fused_decode)
    print(f"unfused (28层): {t1:.4f} ms")
    print(f"fused   (28层): {t2:.4f} ms")
    print(f"speedup:        {t1/t2:.2f}x\n")

    # ==========================================
    # Part 2: 单层 kernel launch overhead 分析
    # ==========================================
    print(f"=== Part 2: 单层 kernel launch overhead ===")

    single_norm = RMSNorm(H, eps=1e-5)
    single_norm.weight = single_norm.weight.to(device)
    x_single = torch.randn(B, S, H, device=device)
    res_single = torch.randn(B, S, H, device=device)

    def run_single_unfused():
        tmp = x_single + res_single
        return single_norm(tmp)

    res_single2 = res_single.clone()
    def run_single_fused():
        return single_norm(x_single, residual=res_single2)

    t3 = benchmark_op(run_single_unfused)
    t4 = benchmark_op(run_single_fused)
    print(f"单层 unfused: {t3:.4f} ms")
    print(f"单层 fused:   {t4:.4f} ms")
    print(f"speedup:      {t3/t4:.2f}x\n")

    # ==========================================
    # Part 3: 大 tensor，有缓存（naive）
    # ==========================================
    B2, S2, H2 = 16, 1024, 4096
    print(f"=== Part 3: 大 tensor (有缓存): B={B2}, S={S2}, H={H2} ===")
    print(f"    tensor 大小约 {B2*S2*H2*4/1024/1024:.0f} MB")

    big_norm = RMSNorm(H2, eps=1e-5)
    big_norm.weight = big_norm.weight.to(device)
    big_x   = torch.randn(B2, S2, H2, device=device)
    big_res = torch.randn(B2, S2, H2, device=device)

    def run_big_unfused_cached():
        tmp = big_x + big_res
        return big_norm(tmp)

    big_res2 = big_res.clone()
    def run_big_fused_cached():
        return big_norm(big_x, residual=big_res2)

    t5 = benchmark_op(run_big_unfused_cached)
    t6 = benchmark_op(run_big_fused_cached)
    print(f"大tensor(有缓存) unfused: {t5:.4f} ms")
    print(f"大tensor(有缓存) fused:   {t6:.4f} ms")
    print(f"speedup:                  {t5/t6:.2f}x\n")

    # ==========================================
    # Part 4: 大 tensor，无缓存（cache-busting）
    # ==========================================
    print(f"=== Part 4: 大 tensor (无缓存 cache-busting) ===")

    # 显存不足，缩小 batch，但保证总大小超过 L2 cache
    # B=4, S=1024, H=4096 → 单个 tensor 约 64MB，10个 buf = 640MB > L2
    B3, S3, H3 = 4, 1024, 4096
    N_BUFS = 10
    print(f"    每个tensor约 {B3*S3*H3*4/1024/1024:.0f} MB, {N_BUFS} 个 buf")

    bust_norm = RMSNorm(H3, eps=1e-5)
    bust_norm.weight = bust_norm.weight.to(device)

    bufs_x    = [torch.randn(B3, S3, H3, device=device) for _ in range(N_BUFS)]
    bufs_res  = [torch.randn(B3, S3, H3, device=device) for _ in range(N_BUFS)]
    bufs_res2 = [t.clone() for t in bufs_res]

    idx_u = [0]
    def run_big_unfused_nocache():
        i = idx_u[0] % N_BUFS; idx_u[0] += 1
        tmp = bufs_x[i] + bufs_res[i]
        return bust_norm(tmp)

    idx_f = [0]
    def run_big_fused_nocache():
        i = idx_f[0] % N_BUFS; idx_f[0] += 1
        return bust_norm(bufs_x[i], residual=bufs_res2[i])

    t7 = benchmark_op(run_big_unfused_nocache)
    t8 = benchmark_op(run_big_fused_nocache)
    print(f"大tensor(无缓存) unfused: {t7:.4f} ms")
    print(f"大tensor(无缓存) fused:   {t8:.4f} ms")
    print(f"speedup:                  {t7/t8:.2f}x\n")

    # ==========================================
    # 汇总
    # ==========================================
    print("=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"{'场景':<30} {'unfused':>10} {'fused':>10} {'speedup':>10}")
    print("-" * 50)
    print(f"{'decode (B=1,S=1,28层)':<30} {t1:>10.4f} {t2:>10.4f} {t1/t2:>10.2f}x")
    print(f"{'单层 launch overhead':<30} {t3:>10.4f} {t4:>10.4f} {t3/t4:>10.2f}x")
    print(f"{'大tensor 有缓存':<30} {t5:>10.4f} {t6:>10.4f} {t5/t6:>10.2f}x")
    print(f"{'大tensor 无缓存':<30} {t7:>10.4f} {t8:>10.4f} {t7/t8:>10.2f}x")
    print("=" * 50)