#!/usr/bin/env python3
"""
Tile Size Experiment for linear_kernel_tf32
Fixed: num_warps=4, num_stages=1
Variable: TILE_M, TILE_N, TILE_K
Conditions: Linear.BACKEND="triton", MLP.FUSED=False, EncoderMLP.FUSED=False
"""

import subprocess
import re
import os

FOLDER = "glm_asr_triton_template"
LAYERS_PATH = os.path.join(FOLDER, "layers.py")

# 实验配置：(TILE_M, TILE_N, TILE_K, 描述)
CONFIGS = [
    # 极小分块组 (探底 Decode 性能边界)
    (16,  32,  32, "极小分块(高频调度)"),
    (16,  64,  32, "极小M(TensorCore下界)"),
    (32,  32,  32, "同时缩小M和N"),
    (32,  64,  32, "缩小M(适应Decode)"),
    
    # 现有基准与大分块组
    (64,  64,  32, "baseline"),
    (64,  256, 64, "增大N和K"),
    (128, 64,  32, "增大TILE_M"),
    (128, 128, 32, "增大M和N"),
    (128, 128, 64, "平衡配置(Encoder最优)"),
    (128, 256, 64, "同时增大M和N/K"),
]


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def set_tile_size(tile_m, tile_n, tile_k):
    """修改 layers.py 里 Linear 类的 TILE_M, TILE_N, TILE_K"""
    code = read_file(LAYERS_PATH)

    # 找到 Linear 类的位置，只替换 Linear 类里的 TILE 值
    # 用 class Linear 作为定位锚点
    linear_class_pattern = r'(class Linear:.*?TILE_M = )\d+'
    code = re.sub(linear_class_pattern, f'\\g<1>{tile_m}', code, flags=re.DOTALL)

    linear_n_pattern = r'(class Linear:.*?TILE_M = \d+\s*\n\s*TILE_N = )\d+'
    code = re.sub(linear_n_pattern, f'\\g<1>{tile_n}', code, flags=re.DOTALL)

    linear_k_pattern = r'(class Linear:.*?TILE_N = \d+\s*\n\s*TILE_K = )\d+'
    code = re.sub(linear_k_pattern, f'\\g<1>{tile_k}', code, flags=re.DOTALL)

    write_file(LAYERS_PATH, code)


def run_benchmark():
    """运行 benchmark_detailed.sh 并提取各阶段时间"""
    result = subprocess.run(
        ["./benchmark_detailed.sh", FOLDER],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    # 提取 Decode Step (avg) 时间
    match = re.search(r"Decode Step \(avg\):\s*([\d.]+)ms", output)
    decode_time = float(match.group(1)) if match else None

    # 提取 Audio Encoder 时间
    match_enc = re.search(r"Audio Encoder:\s*([\d.]+)ms", output)
    encoder_time = float(match_enc.group(1)) if match_enc else None

    # 提取 Decoder Prefill 时间
    match_prefill = re.search(r"Decoder Prefill:\s*([\d.]+)ms", output)
    prefill_time = float(match_prefill.group(1)) if match_prefill else None

    return decode_time, encoder_time, prefill_time


def main():
    print("=" * 80)
    print("Tile Size Experiment v2: linear_kernel_tf32")
    print("Fixed: num_warps=4(default), num_stages=1(default)")
    print("Conditions: BACKEND=triton, MLP.FUSED=False, EncoderMLP.FUSED=False")
    print("=" * 80)
    print(f"{'Config':<30} {'Decode(ms)':<14} {'Encoder(ms)':<14} {'Prefill(ms)':<14} {'描述'}")
    print("-" * 80)

    results = []

    for tile_m, tile_n, tile_k, desc in CONFIGS:
        label = f"M={tile_m},N={tile_n},K={tile_k}"

        print(f"Running {label}...", end="", flush=True)

        set_tile_size(tile_m, tile_n, tile_k)

        decode_time, encoder_time, prefill_time = run_benchmark()

        results.append((tile_m, tile_n, tile_k, desc, decode_time, encoder_time, prefill_time))

        decode_str  = f"{decode_time:.2f}"  if decode_time  else "N/A"
        encoder_str = f"{encoder_time:.2f}" if encoder_time else "N/A"
        prefill_str = f"{prefill_time:.2f}" if prefill_time else "N/A"

        print(f"\r{label:<30} {decode_str:<14} {encoder_str:<14} {prefill_str:<14} {desc}")

    # 恢复 baseline
    set_tile_size(64, 64, 32)
    print("\n[恢复 baseline 配置: TILE_M=64, TILE_N=64, TILE_K=32]")

    # 按各阶段分别输出最优
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)

    valid = [(m, n, k, d, dec, enc, pre) for m, n, k, d, dec, enc, pre in results
             if dec is not None and enc is not None and pre is not None]

    best_decode  = min(valid, key=lambda x: x[4])
    best_encoder = min(valid, key=lambda x: x[5])
    best_prefill = min(valid, key=lambda x: x[6])

    print(f"Best Decode Step : M={best_decode[0]},N={best_decode[1]},K={best_decode[2]}"
          f" → {best_decode[4]:.2f}ms ({best_decode[3]})")
    print(f"Best Encoder     : M={best_encoder[0]},N={best_encoder[1]},K={best_encoder[2]}"
          f" → {best_encoder[5]:.2f}ms ({best_encoder[3]})")
    print(f"Best Prefill     : M={best_prefill[0]},N={best_prefill[1]},K={best_prefill[2]}"
          f" → {best_prefill[6]:.2f}ms ({best_prefill[3]})")

    print("\nFull results (sorted by Encoder time):")
    print("-" * 80)
    valid_sorted = sorted(valid, key=lambda x: x[5])
    for i, (m, n, k, d, dec, enc, pre) in enumerate(valid_sorted):
        tag = " ← best encoder" if i == 0 else ""
        print(f"  M={m},N={n},K={k}: encoder={enc:.2f}ms, prefill={pre:.2f}ms, decode={dec:.2f}ms{tag}")


if __name__ == "__main__":
    main()