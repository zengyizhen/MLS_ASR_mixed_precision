#!/usr/bin/env python3
"""
Tile Size Experiment for linear_kernel_tf32
Fixed: TILE_M=64, num_warps=4, num_stages=1
Variable: TILE_N and TILE_K
"""

import subprocess
import re
import os

FOLDER = "glm_asr_triton_template"
LAYERS_PATH = os.path.join(FOLDER, "layers.py")

# 实验配置：(TILE_N, TILE_K)
# TILE_M 固定为 64
CONFIGS = [
    (64,  32),   # baseline
    (128, 32),   # 增大 TILE_N
    (64,  64),   # 增大 TILE_K
    (128, 64),   # 同时增大 TILE_N 和 TILE_K
    (256, 64),   # 进一步增大 TILE_N
]


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def set_tile_size(tile_n, tile_k):
    """修改 layers.py 里 Linear 类的 TILE_N 和 TILE_K"""
    code = read_file(LAYERS_PATH)

    # 只替换 Linear 类里的 TILE_N 和 TILE_K
    # 用更精确的匹配避免误改 MLP 类的值
    code = re.sub(r"(class Linear.*?TILE_M = 64\s*\n\s*)TILE_N = \d+", 
                  f"\\1TILE_N = {tile_n}", code, flags=re.DOTALL)
    code = re.sub(r"(class Linear.*?TILE_N = \d+\s*\n\s*)TILE_K = \d+",
                  f"\\1TILE_K = {tile_k}", code, flags=re.DOTALL)

    write_file(LAYERS_PATH, code)


def run_benchmark():
    """运行 benchmark_detailed.sh 并提取 Decode Step 时间"""
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
    print("=" * 65)
    print("Tile Size Experiment: linear_kernel_tf32")
    print("Fixed: TILE_M=64, num_warps=4(default), num_stages=1(default)")
    print("=" * 65)
    print(f"{'Config':<25} {'Decode(ms)':<15} {'Encoder(ms)':<15} {'Prefill(ms)':<15}")
    print("-" * 65)

    results = []

    for i, (tile_n, tile_k) in enumerate(CONFIGS):
        label = f"TILE_N={tile_n}, TILE_K={tile_k}"
        if i == 0:
            label += " (baseline)"

        print(f"Running {label}...", end="", flush=True)

        # 修改 tile size
        set_tile_size(tile_n, tile_k)

        # 跑 benchmark
        decode_time, encoder_time, prefill_time = run_benchmark()

        results.append((tile_n, tile_k, decode_time, encoder_time, prefill_time))

        decode_str   = f"{decode_time:.2f}"   if decode_time   else "N/A"
        encoder_str  = f"{encoder_time:.2f}"  if encoder_time  else "N/A"
        prefill_str  = f"{prefill_time:.2f}"  if prefill_time  else "N/A"

        print(f"\r{label:<25} {decode_str:<15} {encoder_str:<15} {prefill_str:<15}")

    # 恢复 baseline
    set_tile_size(64, 32)

    # 输出总结
    print("=" * 65)
    print("Summary (sorted by Decode Step time):")
    print("-" * 65)
    valid = [(n, k, d, e, p) for n, k, d, e, p in results if d is not None]
    valid.sort(key=lambda x: x[2])
    for i, (n, k, d, e, p) in enumerate(valid):
        tag = " ← best" if i == 0 else ""
        print(f"  TILE_N={n}, TILE_K={k}: decode={d:.2f}ms{tag}")


if __name__ == "__main__":
    main()