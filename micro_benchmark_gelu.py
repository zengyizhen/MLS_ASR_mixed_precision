#!/usr/bin/env python3
"""
Micro-Benchmark: GELU Kernel BLOCK_SIZE Experiment
Tests gelu_kernel with different BLOCK_SIZE values.
Input: M=750 (audio seq len), N=5120 (intermediate_size)
Total elements = 750 * 5120 = 3,840,000
"""

import torch
import triton
import numpy as np
import sys

FOLDER = "glm_asr_triton_template"
sys.path.insert(0, FOLDER)

from layers import gelu_kernel

# Encoder MLP 中间层尺寸
M = 750
N = 5120
TOTAL = M * N  # 3,840,000

NUM_RUNS = 100  # element-wise kernel 很快，多跑几次取平均

# A100 HBM 理论峰值带宽
A100_BW_GB = 2000.0

# BLOCK_SIZE 候选值
BLOCK_SIZES = [64, 128, 256, 512, 1024]


def run_single_config(block_size, device):
    """运行单个 BLOCK_SIZE 配置，返回平均时间（ms）和带宽（GB/s）"""

    x = torch.randn(TOTAL, dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    grid = (triton.cdiv(TOTAL, block_size),)

    # warmup
    for _ in range(10):
        gelu_kernel[grid](x, y, TOTAL, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(NUM_RUNS):
        start.record()
        gelu_kernel[grid](x, y, TOTAL, BLOCK_SIZE=block_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)

    # Bandwidth = 读 + 写 = 2 × total_elements × 4 bytes
    bytes_accessed = 2 * TOTAL * 4
    bandwidth_gb = bytes_accessed / (avg_time_ms / 1000) / 1e9

    return avg_time_ms, std_time_ms, bandwidth_gb


def verify_correctness(block_size, device):
    """验证不同 BLOCK_SIZE 结果一致"""
    x = torch.randn(TOTAL, dtype=torch.float32, device=device)
    y_ref = torch.empty_like(x)
    y_test = torch.empty_like(x)

    # 用 baseline（256）作为参考
    grid_ref = (triton.cdiv(TOTAL, 256),)
    gelu_kernel[grid_ref](x, y_ref, TOTAL, BLOCK_SIZE=256)

    grid_test = (triton.cdiv(TOTAL, block_size),)
    gelu_kernel[grid_test](x, y_test, TOTAL, BLOCK_SIZE=block_size)

    max_diff = (y_ref - y_test).abs().max().item()
    return max_diff < 1e-5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print("=" * 70)
    print(f"Micro-Benchmark: GELU Kernel BLOCK_SIZE")
    print(f"Input: M={M}, N={N}, Total elements={TOTAL:,}")
    print(f"A100 peak HBM bandwidth: {A100_BW_GB:.0f} GB/s")
    print("=" * 70)

    # 先验证正确性
    print("\nVerifying correctness...")
    for bs in BLOCK_SIZES:
        if bs == 256:
            continue
        ok = verify_correctness(bs, device)
        status = "✓" if ok else "✗"
        print(f"  BLOCK_SIZE={bs}: {status}")

    print()
    print(f"{'BLOCK_SIZE':<12} | {'Time(ms)':>10} | {'Std(ms)':>8} | {'BW(GB/s)':>10} | {'BW利用率':>8} | {'说明'}")
    print("-" * 70)

    results = []
    for i, block_size in enumerate(BLOCK_SIZES):
        desc = "baseline" if block_size == 256 else ""
        try:
            avg_time, std_time, bandwidth = run_single_config(block_size, device)
            bw_util = bandwidth / A100_BW_GB * 100
            results.append((block_size, avg_time, std_time, bandwidth, bw_util))
            print(f"{block_size:<12} | {avg_time:>10.4f} | {std_time:>8.4f} | {bandwidth:>10.2f} | {bw_util:>7.1f}% | {desc}")
        except Exception as e:
            print(f"{block_size:<12} | {'ERROR':>10} | {'N/A':>8} | {'N/A':>10} | {'N/A':>8} | {e}")

    # 总结
    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 70)

    valid = [(bs, t, s, bw, u) for bs, t, s, bw, u in results]
    if valid:
        best  = max(valid, key=lambda x: x[3])
        worst = min(valid, key=lambda x: x[3])
        baseline = next((r for r in valid if r[0] == 256), None)

        print(f"Best     : BLOCK_SIZE={best[0]} → {best[1]:.4f}ms, {best[3]:.2f} GB/s ({best[4]:.1f}% of peak)")
        print(f"Worst    : BLOCK_SIZE={worst[0]} → {worst[1]:.4f}ms, {worst[3]:.2f} GB/s ({worst[4]:.1f}% of peak)")
        if baseline:
            print(f"Baseline : BLOCK_SIZE=256 → {baseline[1]:.4f}ms, {baseline[3]:.2f} GB/s ({baseline[4]:.1f}% of peak)")
        print(f"Speedup best vs worst: {worst[1]/best[1]:.2f}x")


if __name__ == "__main__":
    main()
