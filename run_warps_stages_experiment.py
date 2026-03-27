#!/usr/bin/env python3
"""
num_warps and num_stages Experiment for linear_kernel_tf32
Fixed: TILE_M=128, TILE_N=64, TILE_K=32 (best encoder config from round 1)
Variable: NUM_WARPS, NUM_STAGES
Conditions: Linear.BACKEND="triton", MLP.FUSED=False, EncoderMLP.FUSED=False
"""

import subprocess
import re
import os

FOLDER = "glm_asr_triton_template"
LAYERS_PATH = os.path.join(FOLDER, "layers.py")

# 实验配置：(NUM_WARPS, NUM_STAGES, 描述)
# 固定 TILE_M=128, TILE_N=64, TILE_K=32
CONFIGS = [
    (4, 1, "baseline(Triton默认)"),
    (4, 2, "增加stages"),
    (4, 3, "进一步增加stages"),
    (8, 1, "增加warps"),
    (8, 2, "warps和stages同时增加"),
    (8, 3, "A100推荐配置"),
    (2, 2, "少warps+流水线"),
]


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def set_warps_stages(num_warps, num_stages):
    """修改 layers.py 里 Linear 类的 NUM_WARPS 和 NUM_STAGES"""
    code = read_file(LAYERS_PATH)

    code = re.sub(
        r'(class Linear:.*?NUM_WARPS = )\d+',
        f'\\g<1>{num_warps}',
        code, flags=re.DOTALL
    )
    code = re.sub(
        r'(class Linear:.*?NUM_STAGES = )\d+',
        f'\\g<1>{num_stages}',
        code, flags=re.DOTALL
    )

    write_file(LAYERS_PATH, code)


def run_benchmark():
    """运行 benchmark_detailed.sh 并提取各阶段时间"""
    result = subprocess.run(
        ["./benchmark_detailed.sh", FOLDER],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    match = re.search(r"Decode Step \(avg\):\s*([\d.]+)ms", output)
    decode_time = float(match.group(1)) if match else None

    match_enc = re.search(r"Audio Encoder:\s*([\d.]+)ms", output)
    encoder_time = float(match_enc.group(1)) if match_enc else None

    match_prefill = re.search(r"Decoder Prefill:\s*([\d.]+)ms", output)
    prefill_time = float(match_prefill.group(1)) if match_prefill else None

    return decode_time, encoder_time, prefill_time


def main():
    print("=" * 80)
    print("num_warps & num_stages Experiment: linear_kernel_tf32")
    print("Fixed: TILE_M=128, TILE_N=64, TILE_K=32")
    print("Conditions: BACKEND=triton, MLP.FUSED=False, EncoderMLP.FUSED=False")
    print("=" * 80)
    print(f"{'Config':<30} {'Decode(ms)':<14} {'Encoder(ms)':<14} {'Prefill(ms)':<14} {'描述'}")
    print("-" * 80)

    results = []

    for num_warps, num_stages, desc in CONFIGS:
        label = f"warps={num_warps},stages={num_stages}"

        print(f"Running {label}...", end="", flush=True)

        set_warps_stages(num_warps, num_stages)

        decode_time, encoder_time, prefill_time = run_benchmark()

        results.append((num_warps, num_stages, desc, decode_time, encoder_time, prefill_time))

        decode_str  = f"{decode_time:.2f}"  if decode_time  else "N/A"
        encoder_str = f"{encoder_time:.2f}" if encoder_time else "N/A"
        prefill_str = f"{prefill_time:.2f}" if prefill_time else "N/A"

        print(f"\r{label:<30} {decode_str:<14} {encoder_str:<14} {prefill_str:<14} {desc}")

    # 恢复 baseline
    set_warps_stages(4, 1)
    print("\n[恢复 baseline 配置: NUM_WARPS=4, NUM_STAGES=1]")

    # 输出总结
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)

    valid = [(w, s, d, dec, enc, pre) for w, s, d, dec, enc, pre in results
             if dec is not None and enc is not None and pre is not None]

    best_decode  = min(valid, key=lambda x: x[3])
    best_encoder = min(valid, key=lambda x: x[4])
    best_prefill = min(valid, key=lambda x: x[5])

    print(f"Best Decode Step : warps={best_decode[0]},stages={best_decode[1]}"
          f" → {best_decode[3]:.2f}ms ({best_decode[2]})")
    print(f"Best Encoder     : warps={best_encoder[0]},stages={best_encoder[1]}"
          f" → {best_encoder[4]:.2f}ms ({best_encoder[2]})")
    print(f"Best Prefill     : warps={best_prefill[0]},stages={best_prefill[1]}"
          f" → {best_prefill[5]:.2f}ms ({best_prefill[2]})")

    print("\nFull results (sorted by Encoder time):")
    print("-" * 80)
    valid_sorted = sorted(valid, key=lambda x: x[4])
    for i, (w, s, d, dec, enc, pre) in enumerate(valid_sorted):
        tag = " ← best encoder" if i == 0 else ""
        print(f"  warps={w},stages={s}: encoder={enc:.2f}ms, prefill={pre:.2f}ms, decode={dec:.2f}ms  {d}{tag}")


if __name__ == "__main__":
    main()
