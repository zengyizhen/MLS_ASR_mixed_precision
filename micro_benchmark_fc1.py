#!/usr/bin/env python3
"""
Micro-Benchmark: Encoder FC1 Layer
Tests linear_kernel_tf32 in isolation with realistic matrix dimensions.
M=750 (audio sequence after conv downsampling), K=1280, N=5120
Covers all configs from round 1 (tile size) and round 2 (warps/stages)
"""

import torch
import triton
import triton.language as tl
import numpy as np
import sys
import os

FOLDER = "glm_asr_triton_template"
sys.path.insert(0, FOLDER)

from layers import linear_kernel_tf32, pad_to_multiple

# Encoder FC1 实际矩阵尺寸
M = 750
K = 1280
N = 5120

NUM_RUNS = 20


def run_single_config(tile_m, tile_n, tile_k, num_warps, num_stages, device):
    """运行单个配置并返回平均时间（ms）和 TFLOPS"""

    M_pad = pad_to_multiple(M, tile_m)
    K_pad = pad_to_multiple(K, tile_k)
    N_pad = pad_to_multiple(N, tile_n)

    a = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=device)
    b = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
    c = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=device)

    a[:M, :K] = torch.randn(M, K, device=device)
    b[:K, :N] = torch.randn(K, N, device=device)

    # bias_ptr 占位（HAS_BIAS=False 时 kernel 不会访问）
    bias_ptr = c.new_empty(0)

    grid = (triton.cdiv(M_pad, tile_m), triton.cdiv(N_pad, tile_n))

    # warmup
    for _ in range(5):
        linear_kernel_tf32[grid](
            a, b, c,
            bias_ptr,
            M_pad, N_pad, K_pad,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            HAS_BIAS=False,
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            BLOCK_K=tile_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(NUM_RUNS):
        start.record()
        linear_kernel_tf32[grid](
            a, b, c,
            bias_ptr,
            M_pad, N_pad, K_pad,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            HAS_BIAS=False,
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            BLOCK_K=tile_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time_ms = np.mean(times)
    tflops = 2 * M * N * K / (avg_time_ms / 1000) / 1e12

    return avg_time_ms, tflops


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print("=" * 75)
    print(f"Micro-Benchmark: Encoder FC1 Layer (M={M}, K={K}, N={N})")
    print("=" * 75)
    print(f"{'Config [MxNxK]':<20} | {'Warps':>5} | {'Stages':>6} | {'Time(ms)':>10} | {'TFLOPS':>8}")
    print("-" * 75)

    # ============================================================
    # 第一轮：固定 warps=4, stages=1，变 tile size
    # ============================================================
    round1_configs = [
        (64,  64,  32, 4, 1, "baseline"),
        (128, 64,  32, 4, 1, "增大TILE_M"),
        (64,  256, 64, 4, 1, "Decoder最优"),
        (128, 256, 64, 4, 1, "同时增大M和N/K"),
        (128, 128, 64, 4, 1, "平衡配置"),
        (128, 128, 32, 4, 1, "增大M和N"),
    ]

    print("--- Round 1: Fixed warps=4, stages=1, Variable tile size ---")
    all_results = []
    for tile_m, tile_n, tile_k, warps, stages, desc in round1_configs:
        label = f"[{tile_m}x{tile_n}x{tile_k}]"
        try:
            avg_time, tflops = run_single_config(
                tile_m, tile_n, tile_k, warps, stages, device
            )
            all_results.append((label, warps, stages, avg_time, tflops, desc))
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {avg_time:>10.3f} | {tflops:>8.2f}  # {desc}")
        except Exception as e:
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {'ERROR':>10} | {'N/A':>8}  # {e}")

    # ============================================================
    # 第二轮：固定 TILE_M=128, TILE_N=64, TILE_K=32，变 warps/stages
    # ============================================================
    round2_configs = [
        (128, 64, 32, 4, 1, "baseline"),
        (128, 64, 32, 4, 2, "增加stages"),
        (128, 64, 32, 4, 3, "进一步增加stages"),
        (128, 64, 32, 8, 1, "增加warps"),
        (128, 64, 32, 8, 2, "warps和stages同时增加"),
        (128, 64, 32, 8, 3, "A100推荐配置"),
        (128, 64, 32, 2, 2, "少warps+流水线"),
    ]

    print("\n--- Round 2: Fixed TILE_M=128, TILE_N=64, TILE_K=32, Variable warps/stages ---")
    for tile_m, tile_n, tile_k, warps, stages, desc in round2_configs:
        label = f"[{tile_m}x{tile_n}x{tile_k}]"
        try:
            avg_time, tflops = run_single_config(
                tile_m, tile_n, tile_k, warps, stages, device
            )
            all_results.append((label, warps, stages, avg_time, tflops, desc))
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {avg_time:>10.3f} | {tflops:>8.2f}  # {desc}")
        except Exception as e:
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {'ERROR':>10} | {'N/A':>8}  # {e}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 75)
    print("Summary:")
    print("-" * 75)

    valid = [
        (l, w, s, t, f, d) for l, w, s, t, f, d in all_results
        if isinstance(t, float)
    ]

    if valid:
        best  = max(valid, key=lambda x: x[4])
        worst = min(valid, key=lambda x: x[4])
        print(f"Best  : {best[0]} warps={best[1]},stages={best[2]}"
              f" → {best[3]:.3f}ms, {best[4]:.2f} TFLOPS  ({best[5]})")
        print(f"Worst : {worst[0]} warps={worst[1]},stages={worst[2]}"
              f" → {worst[3]:.3f}ms, {worst[4]:.2f} TFLOPS  ({worst[5]})")
        print(f"Speedup best vs worst: {worst[3]/best[3]:.2f}x")
        print(f"\nReference: A100 peak FP32 = 19.5 TFLOPS")
        print(f"GPU utilization of best config: {best[4]/19.5*100:.1f}%")
    else:
        print("No valid results.")


if __name__ == "__main__":
    main()