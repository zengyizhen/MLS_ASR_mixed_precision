import torch
from layers import EncoderMLP

def benchmark_mlp(mlp, x, num_warmup=50, num_iters=1000):
    """
    科学的 GPU 测速函数：热身 -> 循环记录 -> 同步 -> 计算平均延迟
    """
    # 1. 热身 (Warm-up)：让 GPU 满载，把缓存填满
    for _ in range(num_warmup):
        _ = mlp(x)
    torch.cuda.synchronize()

    # 2. 准备 CUDA 事件计时器
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 3. 正式计时
    start_event.record()
    for _ in range(num_iters):
        _ = mlp(x)
    end_event.record()
    
    # 4. 强制同步！等待 GPU 把这 1000 次循环里的活儿全干完
    torch.cuda.synchronize()

    # 计算平均单次前向传播的毫秒数 (ms)
    total_time_ms = start_event.elapsed_time(end_event)
    return total_time_ms / num_iters

if __name__ == "__main__":
    print("=== 🚀 EncoderMLP 算子融合微观基准测试 (Micro-benchmark) ===")
    
    # 设置一个符合 ASR 模型特征的张量维度 (Batch=2, Seq_len=1024, Hidden=256)
    device = torch.device("cuda")
    x = torch.randn(2, 1024, 256, device=device, dtype=torch.float32)
    
    # 实例化我们的 EncoderMLP (隐藏层 256，中间层扩展到 1024)
    mlp = EncoderMLP(hidden_size=256, intermediate_size=1024, activation="gelu")
    
    # ==========================================
    # 对照组：未融合 (朴素的 Linear + GELU)
    # ==========================================
    EncoderMLP.FUSED = False
    time_unfused = benchmark_mlp(mlp, x)
    print(f"[-] 朴素模式 (Unfused) 平均延迟: {time_unfused:.4f} ms")

    # ==========================================
    # 实验组：算子融合 (Linear & GELU in one kernel)
    # ==========================================
    EncoderMLP.FUSED = True
    time_fused = benchmark_mlp(mlp, x)
    print(f"[+] 融合模式 (Fused)   平均延迟: {time_fused:.4f} ms")

    # ==========================================
    # 计算报告所需的终极数据
    # ==========================================
    speedup = time_unfused / time_fused
    print("-" * 50)
    print(f"🏆 加速比 (Speedup): {speedup:.2f}x")
    print(f"📉 绝对延迟降低: {time_unfused - time_fused:.4f} ms")
    print("-" * 50)