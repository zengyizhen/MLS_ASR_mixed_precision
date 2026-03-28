"""
NVIDIA Nsight Compute Kernel-Level Profiling Script
====================================================
用途：精确找出每个 kernel 的耗时和内存访问模式

使用方法（两种）：
-----------------
方法 A — Nsight Compute CLI（推荐，最详细）：
    ncu --set full \
        --kernel-name-base function \
        --launch-count 5 \
        --csv \
        -o profile_report \
        python profile_nsight.py

    # 或者只看特定 kernel（如 linear_kernel_tf32）：
    ncu --kernel-name "linear_kernel_tf32" \
        --metrics l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        python profile_nsight.py

方法 B — torch.profiler（无需 root/管理员权限）：
    python profile_nsight.py --mode torch

方法 C — 只生成 NVTX 注解，然后用 Nsight Systems 查看时间线：
    nsys profile --stats=true python profile_nsight.py --mode nvtx
"""

import argparse
import sys
import time
import torch
import numpy as np

# ============================================================================
# 导入模型组件
# ============================================================================
try:
    from layers import RMSNorm, LayerNorm, Linear, Embedding, MLP, gelu, silu
    from model import GlmAsrConfig, DecoderLayer, TextDecoder
    from rope import RotaryEmbedding
except ImportError as e:
    print(f"[ERROR] 无法导入模型文件: {e}")
    print("请确保从项目根目录运行此脚本")
    sys.exit(1)


# ============================================================================
# 配置
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# 模拟 autoregressive decode 时的形状
BATCH_SIZE   = 1
SEQ_LEN_PREF = 375   # prefill: audio + prompt tokens
SEQ_LEN_DEC  = 1     # decode: 每次只生成 1 个 token
HIDDEN_SIZE  = 3584
INTER_SIZE   = 18944
NUM_HEADS    = 28
NUM_KV_HEADS = 4
HEAD_DIM     = HIDDEN_SIZE // NUM_HEADS

WARMUP_ITERS = 3
PROFILE_ITERS = 10


# ============================================================================
# NVTX 标注助手（供 Nsight Systems / ncu 使用）
# ============================================================================
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class _NullNVTX:
        def range_push(self, name): pass
        def range_pop(self): pass
    nvtx = _NullNVTX()


class NVTXRange:
    """上下文管理器：自动 push/pop NVTX range"""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        nvtx.range_push(self.name)
        return self
    def __exit__(self, *args):
        nvtx.range_pop()


# ============================================================================
# 计时工具
# ============================================================================
class CUDATimer:
    """使用 CUDA Events 精确测量 GPU kernel 时间"""

    def __init__(self, name: str, warmup: int = WARMUP_ITERS, iters: int = PROFILE_ITERS):
        self.name    = name
        self.warmup  = warmup
        self.iters   = iters
        self.results = []

    def __call__(self, fn, *args, **kwargs):
        # warmup
        for _ in range(self.warmup):
            out = fn(*args, **kwargs)
        torch.cuda.synchronize()

        # 正式计时
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(self.iters):
            start_ev.record()
            with NVTXRange(self.name):
                out = fn(*args, **kwargs)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))  # ms

        mean_ms = np.mean(times)
        std_ms  = np.std(times)
        self.results.append((self.name, mean_ms, std_ms))
        print(f"  {self.name:<45s}  {mean_ms:7.3f} ms  ±{std_ms:5.3f} ms")
        return out


# ============================================================================
# 被测组件（模拟 Decoder 中每个算子）
# ============================================================================

def make_inputs(seq_len=SEQ_LEN_DEC):
    """生成随机输入张量"""
    x = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    return x


def profile_rmsnorm(timer_fn):
    print("\n── RMSNorm (hidden=3584) ──")
    norm = RMSNorm(HIDDEN_SIZE).weight.to(DEVICE)
    # 单独构造一个 RMSNorm 对象
    rms = RMSNorm(HIDDEN_SIZE)
    rms.weight = rms.weight.to(DEVICE)
    x = make_inputs()

    # 普通 RMSNorm
    t = CUDATimer("RMSNorm (plain)")
    t(rms, x)

    # Fused Add+RMSNorm
    t2 = CUDATimer("Fused Add+RMSNorm")
    residual = torch.randn_like(x)
    def fused_call():
        res = residual.clone()  # 避免 in-place 破坏
        return rms(x, residual=res)
    t2(fused_call)


def profile_linear(timer_fn):
    print("\n── Linear GEMV (decode: M=1) ──")
    # Typical sizes in the decoder
    shapes = [
        ("q_proj  3584→3584", HIDDEN_SIZE, HIDDEN_SIZE),
        ("o_proj  3584→3584", HIDDEN_SIZE, HIDDEN_SIZE),
        ("gate_proj 3584→18944", HIDDEN_SIZE, INTER_SIZE),
        ("down_proj 18944→3584", INTER_SIZE, HIDDEN_SIZE),
    ]
    for label, in_f, out_f in shapes:
        lin = Linear(in_f, out_f, bias=False)
        lin.weight = lin.weight.to(DEVICE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN_DEC, in_f, device=DEVICE, dtype=DTYPE)
        t = CUDATimer(f"Linear {label}")
        t(lin, x)


def profile_linear_prefill(timer_fn):
    print("\n── Linear GEMM (prefill: M=375) ──")
    shapes = [
        ("gate_proj 3584→18944", HIDDEN_SIZE, INTER_SIZE),
        ("down_proj 18944→3584", INTER_SIZE, HIDDEN_SIZE),
    ]
    for label, in_f, out_f in shapes:
        lin = Linear(in_f, out_f, bias=False)
        lin.weight = lin.weight.to(DEVICE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN_PREF, in_f, device=DEVICE, dtype=DTYPE)
        t = CUDATimer(f"Linear PREFILL {label}")
        t(lin, x)


def profile_mlp(timer_fn):
    print("\n── MLP SwiGLU (decode) ──")
    mlp = MLP(HIDDEN_SIZE, INTER_SIZE, activation="silu", use_gating=True)
    mlp.gate_proj.weight = mlp.gate_proj.weight.to(DEVICE)
    mlp.up_proj.weight   = mlp.up_proj.weight.to(DEVICE)
    mlp.down_proj.weight = mlp.down_proj.weight.to(DEVICE)
    x = make_inputs(SEQ_LEN_DEC)

    t_fused = CUDATimer("MLP fused SwiGLU (decode M=1)")
    t_fused(mlp, x)

    x_pref = make_inputs(SEQ_LEN_PREF)
    t_pref = CUDATimer("MLP fused SwiGLU (prefill M=375)")
    t_pref(mlp, x_pref)


def profile_attention(timer_fn):
    print("\n── Scaled Dot-Product Attention ──")
    from attention import scaled_dot_product_attention
    # decode: q=(1,28,1,128), k=(1,4,T,128)
    for kv_len in [1, 50, 200, 375]:
        q = torch.randn(BATCH_SIZE, NUM_HEADS,    SEQ_LEN_DEC, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, kv_len,      HEAD_DIM, device=DEVICE, dtype=DTYPE)
        v = torch.randn(BATCH_SIZE, NUM_KV_HEADS, kv_len,      HEAD_DIM, device=DEVICE, dtype=DTYPE)
        t = CUDATimer(f"Attention decode (kv_len={kv_len:4d})")
        t(scaled_dot_product_attention, q, k, v, None)


def profile_gelu_silu():
    print("\n── Element-wise Activations ──")
    x_inter = torch.randn(BATCH_SIZE, SEQ_LEN_DEC, INTER_SIZE, device=DEVICE, dtype=DTYPE)

    t1 = CUDATimer("GELU (intermediate_size=18944)")
    t1(gelu, x_inter)

    t2 = CUDATimer("SiLU (intermediate_size=18944)")
    t2(silu, x_inter)


# ============================================================================
# torch.profiler 模式
# ============================================================================

def run_torch_profiler():
    """用 torch.profiler 生成 Chrome trace，可在 chrome://tracing 查看"""
    from torch.profiler import profile, record_function, ProfilerActivity

    print("\n[torch.profiler] 正在 profile 一次完整的 Decoder forward...")

    config = GlmAsrConfig()
    rope   = RotaryEmbedding(
        dim=HEAD_DIM,
        max_position_embeddings=config.text_max_position_embeddings,
        base=config.text_rope_base,
    )
    layer = DecoderLayer(
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, INTER_SIZE, rope
    )
    # move weights to device
    for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        proj = getattr(layer, attr)
        proj.weight = proj.weight.to(DEVICE)
    layer.mlp.gate_proj.weight = layer.mlp.gate_proj.weight.to(DEVICE)
    layer.mlp.up_proj.weight   = layer.mlp.up_proj.weight.to(DEVICE)
    layer.mlp.down_proj.weight = layer.mlp.down_proj.weight.to(DEVICE)
    layer.input_layernorm.weight        = layer.input_layernorm.weight.to(DEVICE)
    layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight.to(DEVICE)

    x    = make_inputs(SEQ_LEN_DEC)
    pos  = torch.zeros(BATCH_SIZE, SEQ_LEN_DEC, dtype=torch.int64, device=DEVICE)

    # warmup
    for _ in range(WARMUP_ITERS):
        _ = layer(x, position_ids=pos)
    torch.cuda.synchronize()

    trace_path = "/sessions/keen-nifty-davinci/mnt/glm_asr_triton_template_final_MP/torch_profile_trace"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
    ) as prof:
        for _ in range(7):
            with record_function("DecoderLayer.forward"):
                _ = layer(x, position_ids=pos)
            prof.step()

    # 打印 top kernels
    print("\n[torch.profiler] Top 20 CUDA kernels by self CUDA time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))

    # 导出 JSON trace
    json_path = "/sessions/keen-nifty-davinci/mnt/glm_asr_triton_template_final_MP/torch_profile.json"
    prof.export_chrome_trace(json_path)
    print(f"\nChrome trace 已保存到: {json_path}")
    print("用 chrome://tracing 打开，或用 perfetto.dev 查看")


# ============================================================================
# 内存带宽分析（理论 vs 实际 roofline）
# ============================================================================

def analyze_roofline():
    """
    简单 Roofline 分析：判断每个算子是 compute-bound 还是memory-bound。

    A100:  FP32 peak = 19.5 TFLOPS, HBM2e BW = 2.0 TB/s
    V100:  FP32 peak = 14.1 TFLOPS, HBM2  BW = 0.9 TB/s
    RTX3090: FP32 peak = 35.6 TFLOPS, GDDR6X BW = 0.936 TB/s
    """
    print("\n" + "="*70)
    print("ROOFLINE ANALYSIS — Decode step (M=1 token)")
    print("="*70)

    # 用你实际的 GPU 填这里
    GPU_FLOPS_TFLOPS = 19.5   # A100 fp32
    GPU_BW_TBs       = 2.0    # A100 HBM

    ops = [
        # (name, FLOPs, bytes_read_write)
        ("q_proj  (1×3584 @ 3584×3584)", 2*1*HIDDEN_SIZE*HIDDEN_SIZE, 4*(HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE)),
        ("gate_proj (1×3584 @ 3584×18944)", 2*1*HIDDEN_SIZE*INTER_SIZE, 4*(HIDDEN_SIZE*INTER_SIZE + HIDDEN_SIZE + INTER_SIZE)),
        ("down_proj (1×18944 @ 18944×3584)", 2*1*INTER_SIZE*HIDDEN_SIZE, 4*(INTER_SIZE*HIDDEN_SIZE + INTER_SIZE + HIDDEN_SIZE)),
        ("RMSNorm  (1×3584)", 5*1*HIDDEN_SIZE, 4*(2*HIDDEN_SIZE + HIDDEN_SIZE)),  # read x+weight, write y
        ("Fused Add+RMSNorm (1×3584)", 6*1*HIDDEN_SIZE, 4*(2*HIDDEN_SIZE + HIDDEN_SIZE)),  # 减少1次read
    ]

    print(f"\n  {'Operation':<40s} {'ArithIntensity':>15s}  {'Bound'}")
    print(f"  {'-'*40} {'-'*15}  {'-'*14}")
    ridge_point = (GPU_FLOPS_TFLOPS * 1e12) / (GPU_BW_TBs * 1e12)  # FLOP/byte
    for name, flops, bytes_io in ops:
        intensity = flops / bytes_io  # FLOP/byte
        bound = "COMPUTE" if intensity > ridge_point else "MEMORY BW"
        print(f"  {name:<40s} {intensity:>12.1f} F/B   {bound}")

    print(f"\n  GPU 峰值计算强度 (ridge point) = {ridge_point:.1f} FLOP/Byte")
    print(f"  → M=1 时所有 Linear 都是 MEMORY BW 瓶颈")
    print(f"  → 优化 RMSNorm 无法突破 Linear GEMV 的带宽墙")


# ============================================================================
# ncu 命令提示
# ============================================================================

def print_ncu_commands():
    script = "profile_nsight.py"
    print("\n" + "="*70)
    print("NVIDIA NSIGHT COMPUTE 使用命令")
    print("="*70)
    print("""
# 1. 完整分析所有 kernel（写入 .ncu-rep 文件，用 ncu-ui 打开）：
ncu --set full \\
    --kernel-name-base function \\
    -o nsight_report \\
    python profile_nsight.py --mode nvtx

# 2. 只分析 linear_kernel_tf32（减少采集时间）：
ncu --kernel-name "linear_kernel_tf32" \\
    --metrics \\
      l1tex__t_bytes.sum,\\
      dram__bytes.sum,\\
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
      gpu__time_duration.sum \\
    python profile_nsight.py --mode nvtx

# 3. 只分析 fused_add_rmsnorm_kernel：
ncu --kernel-name "fused_add_rmsnorm_kernel" \\
    --metrics \\
      dram__bytes_read.sum,dram__bytes_write.sum,\\
      l1tex__t_bytes_lookup_miss_pipe_l1tex_m_l1tex2xbar_read_sectors.sum,\\
      gpu__time_duration.sum \\
    python profile_nsight.py --mode nvtx

# 4. 生成 roofline 分析报告：
ncu --set roofline \\
    -o nsight_roofline \\
    python profile_nsight.py --mode nvtx

# 5. 查看 .ncu-rep（命令行）：
ncu --import nsight_report.ncu-rep --print-summary per-kernel

# 说明：
#   --set full       收集所有可用 metrics（速度慢但信息全）
#   --set basic      只收集基本 metrics（速度快）
#   --kernel-name    只 profile 指定名称的 kernel（加速分析）
#   -o               输出文件名（.ncu-rep 格式）
#   ncu-ui           GUI 工具，可查看 roofline、source-level 分析
""")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-ASR Nsight Compute Profiling")
    parser.add_argument(
        "--mode",
        choices=["all", "torch", "nvtx", "roofline"],
        default="all",
        help="profiling 模式：all=完整计时, torch=torch.profiler, nvtx=仅NVTX标注, roofline=理论分析",
    )
    args = parser.parse_args()

    if DEVICE.type != "cuda":
        print("[WARNING] 未检测到 CUDA，将在 CPU 上运行（无 GPU 计时）")

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"Model config: hidden={HIDDEN_SIZE}, intermediate={INTER_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}")

    if args.mode == "roofline":
        analyze_roofline()
        print_ncu_commands()
        return

    if args.mode == "torch":
        run_torch_profiler()
        return

    if args.mode in ("all", "nvtx"):
        print("\n" + "="*70)
        print("CUDA KERNEL TIMING  (mean ± std over 10 iters, CUDA Events)")
        print("="*70)
        # 依次 profile 每个算子
        profile_rmsnorm(None)
        profile_linear(None)
        profile_linear_prefill(None)
        profile_mlp(None)
        profile_attention(None)
        profile_gelu_silu()

        analyze_roofline()
        print_ncu_commands()

    if args.mode == "nvtx":
        print("\n[NVTX 模式] 所有算子已用 NVTX range 标注。")
        print("请用 nsys 或 ncu 的 --kernel-name 配合上面的命令使用。")


if __name__ == "__main__":
    main()
