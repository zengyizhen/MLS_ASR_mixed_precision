#!/usr/bin/env python3
"""
Micro-Benchmark: Encoder FC1 Layer
Tests linear_kernel_tf32 in isolation with realistic matrix dimensions.
M=750 (audio sequence after conv downsampling), K=1280, N=5120

Execution Flow:
1. Round 1: Tests various Tile Sizes (M, N, K) with fixed warps=4, stages=1.
2. Heuristic Dispatch: Automatically selects the best Tile Size from Round 1.
3. Round 2: Tests various warps and stages configurations based on the optimal Tile Size.
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


def run_single_config(tile_m, tile_n, tile_k, num_warps, num_stages, device):
    """运行单个配置并使用 triton.testing.do_bench 返回平均时间（ms）和 TFLOPS"""

    M_pad = pad_to_multiple(M, tile_m)
    K_pad = pad_to_multiple(K, tile_k)
    N_pad = pad_to_multiple(N, tile_n)

    a = torch.randn((M_pad, K_pad), dtype=torch.float32, device=device)
    b = torch.randn((K_pad, N_pad), dtype=torch.float32, device=device)
    c = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=device)

    # bias_ptr 占位（HAS_BIAS=False 时 kernel 不会访问）
    bias_ptr = c.new_empty(0)

    grid = (triton.cdiv(M_pad, tile_m), triton.cdiv(N_pad, tile_n))

    # 定义 Kernel 调用闭包
    kernel_call = lambda: linear_kernel_tf32[grid](
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

    # do_bench 自动处理 warmup 并在稳定后返回中位数耗时 (ms)
    avg_time_ms = triton.testing.do_bench(kernel_call, warmup=25, rep=100)

    # 计算理论 TFLOPS
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
    # Round 1: 固定 warps=4, stages=1，变 tile size
    # ============================================================
    round1_configs = [
        # 极小分块组 (探底 Decode 性能边界)
        (16,  32,  32, 4, 1, "极小分块(高频调度)"),
        (16,  64,  32, 4, 1, "极小M(TensorCore下界)"),
        (32,  32,  32, 4, 1, "同时缩小M和N"),
        (32,  64,  32, 4, 1, "缩小M(适应Decode)"),
        
        # 现有基准与大分块组
        (64,  64,  32, 4, 1, "baseline"),
        (64,  256, 64, 4, 1, "增大N和K"),
        (128, 64,  32, 4, 1, "增大TILE_M"),
        (128, 128, 32, 4, 1, "增大M和N"),
        (128, 128, 64, 4, 1, "平衡配置(Encoder最优)"),
        (128, 256, 64, 4, 1, "同时增大M和N/K"),
    ]

    print("--- Round 1: Fixed warps=4, stages=1, Variable tile size ---")
    r1_results = []
    for tile_m, tile_n, tile_k, warps, stages, desc in round1_configs:
        label = f"[{tile_m}x{tile_n}x{tile_k}]"
        try:
            avg_time, tflops = run_single_config(tile_m, tile_n, tile_k, warps, stages, device)
            r1_results.append((tile_m, tile_n, tile_k, warps, stages, avg_time, tflops, desc, label))
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {avg_time:>10.3f} | {tflops:>8.2f}  # {desc}")
        except Exception as e:
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {'ERROR':>10} | {'N/A':>8}  # {e}")

    valid_r1 = [res for res in r1_results if isinstance(res[6], float)]
    if not valid_r1:
        print("\nERROR: Round 1 failed to produce valid results. Exiting.")
        return

    # 动态寻优：获取 Round 1 中 TFLOPS 最高的配置
    best_r1 = max(valid_r1, key=lambda x: x[6])
    best_m, best_n, best_k = best_r1[0], best_r1[1], best_r1[2]

    print(f"\n=> Round 1 Best Config: [{best_m}x{best_n}x{best_k}] achieved {best_r1[6]:.2f} TFLOPS.")

    # ============================================================
    # Round 2: 固定 Round 1 的最优 Tile Size，变 warps/stages
    # ============================================================
    round2_configs = [
        (best_m, best_n, best_k, 4, 2, "增加 stages"),
        (best_m, best_n, best_k, 4, 3, "进一步增加 stages"),
        (best_m, best_n, best_k, 8, 1, "增加 warps"),
        (best_m, best_n, best_k, 8, 2, "warps 和 stages 同时增加"),
        (best_m, best_n, best_k, 8, 3, "A100 高并发+深流水线"),
        (best_m, best_n, best_k, 2, 2, "少 warps + 流水线"),
    ]

    print(f"\n--- Round 2: Fixed Tile Size [{best_m}x{best_n}x{best_k}], Variable warps/stages ---")
    r2_results = []
    for tile_m, tile_n, tile_k, warps, stages, desc in round2_configs:
        label = f"[{tile_m}x{tile_n}x{tile_k}]"
        try:
            avg_time, tflops = run_single_config(tile_m, tile_n, tile_k, warps, stages, device)
            r2_results.append((tile_m, tile_n, tile_k, warps, stages, avg_time, tflops, desc, label))
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {avg_time:>10.3f} | {tflops:>8.2f}  # {desc}")
        except Exception as e:
            print(f"{label:<20} | {warps:>5} | {stages:>6} | {'ERROR':>10} | {'N/A':>8}  # {e}")

    # ============================================================
    # 全局总结
    # ============================================================
    print("\n" + "=" * 75)
    print("Summary:")
    print("-" * 75)

    all_results = valid_r1 + [res for res in r2_results if isinstance(res[6], float)]

    if all_results:
        global_best  = max(all_results, key=lambda x: x[6])
        global_worst = min(all_results, key=lambda x: x[6])
        
        print(f"Global Best  : [{global_best[0]}x{global_best[1]}x{global_best[2]}] "
              f"warps={global_best[3]}, stages={global_best[4]}")
        print(f"               → {global_best[5]:.3f}ms, {global_best[6]:.2f} TFLOPS ({global_best[7]})")
        
        print(f"Global Worst : [{global_worst[0]}x{global_worst[1]}x{global_worst[2]}] "
              f"warps={global_worst[3]}, stages={global_worst[4]}")
        print(f"               → {global_worst[5]:.3f}ms, {global_worst[6]:.2f} TFLOPS ({global_worst[7]})")
        
        print(f"Speedup (Best vs Worst): {global_worst[5]/global_best[5]:.2f}x")
        
        print(f"\nReference: A100 peak TF32 = 156.0 TFLOPS")
        print(f"GPU utilization of best config: {global_best[6]/156.0*100:.1f}%")
    else:
        print("No valid results for summary.")

if __name__ == "__main__":
    main()