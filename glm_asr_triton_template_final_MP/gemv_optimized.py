"""
GEMV Optimization for Autoregressive Decode (M=1)
==================================================

问题根因分析：
  当前 linear_kernel_tf32 使用 TILE_M=128。
  Decode 阶段 M=1，代码将 M pad 到 128 再送入 GEMM kernel，
  意味着实际计算了 128 行输出但只有 1 行有效 → 浪费 128× 算力。
  同时权重以 FP32 存储，GEMV 是内存带宽瓶颈，FP32 比 FP16 多耗 2× 带宽。

三级优化方案：
  Level 1 — 专用 GEMV Triton kernel     （修复 128× 浪费，约 2-4× 加速）
  Level 2 — FP16 权重存储               （HBM 带宽减半，约 1.5-2× 加速）
  Level 3 — INT8 权重量化               （带宽再减半，约 1.5× 额外加速）
"""

import numpy as np
import torch
import triton
import triton.language as tl

# ============================================================================
# Level 1: 专用 GEMV Triton Kernel（修复 M=1 时的 128× 浪费）
# ============================================================================
#
# 设计思路：
#   weight shape = [N_out, K_in]  (row-major, weight[i, :] = 第 i 个输出神经元的权重行)
#   x shape = [K_in]              (输入向量)
#   y shape = [N_out]             (输出向量)
#
#   y[i] = sum_k(weight[i, k] * x[k])
#
#   Grid = (N_out // BLOCK_N,)
#   每个 block 负责 BLOCK_N 个输出元素，沿 K 方向做 reduction。
#   关键：x[k] 可以在 BLOCK_N 行之间共享（L1 复用），效率远高于 GEMM。

@triton.jit
def gemv_kernel(
    weight_ptr,   # [N, K]  row-major
    x_ptr,        # [K]
    y_ptr,        # [N]
    N,            # out_features
    K,            # in_features
    stride_wn,    # weight.stride(0) = K（每行 K 个元素）
    stride_wk,    # weight.stride(1) = 1
    BLOCK_N: tl.constexpr,   # 每个 block 处理多少输出元素
    BLOCK_K: tl.constexpr,   # 每次加载 x 的多少元素（K 方向 tile）
):
    """
    专用 GEMV kernel：y = weight @ x，针对 x 是向量（M=1）优化。

    与 linear_kernel_tf32(TILE_M=128) 的关键区别：
      - 不再 pad M 到 128，不浪费算力
      - x 在 BLOCK_N 行之间被 L1/寄存器复用
      - 支持 triton.autotune 自动搜索 (BLOCK_N, BLOCK_K) 最优值
    """
    pid   = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # 当前 block 负责的输出索引
    mask_n = offs_n < N

    # 累加器（每个输出元素一个）
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # 沿 K 方向分块迭代
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # 加载权重 tile [BLOCK_N, BLOCK_K]
        w = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )

        # 加载输入向量 tile [BLOCK_K]（在 BLOCK_N 行间共享）
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)

        # 点积并累加：[BLOCK_N, BLOCK_K] × [BLOCK_K] → [BLOCK_N]
        acc += tl.sum(w * x[None, :], axis=1)

    # 写回结果
    tl.store(y_ptr + offs_n, acc, mask=mask_n)


# autotune 候选配置（Triton 自动搜索最优 BLOCK_N / BLOCK_K）
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32,  "BLOCK_K": 256}),
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 256}),
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 128}),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}),
    ],
    key=["N", "K"],
)
@triton.jit
def gemv_kernel_autotuned(
    weight_ptr, x_ptr, y_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """gemv_kernel 的 autotuned 版本，Triton 自动选最优 tile size。"""
    pid   = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc, mask=mask_n)


# ============================================================================
# Level 2: FP16 权重的 GEMV（带宽减半）
# ============================================================================

@triton.jit
def gemv_fp16_kernel(
    weight_ptr,   # [N, K]  float16
    x_ptr,        # [K]     float32
    y_ptr,        # [N]     float32
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    权重以 FP16 存储，输入/输出保持 FP32，累加在 FP32 进行。
    相比 FP32 权重版本，HBM 读取量减半 → 理论加速 ~2×（纯带宽瓶颈时）。
    """
    pid   = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # 加载 FP16 权重并升精度到 FP32
        w_fp16 = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = w_fp16.to(tl.float32)

        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(y_ptr + offs_n, acc, mask=mask_n)


# ============================================================================
# Level 3: INT8 权重量化 GEMV（带宽再减半，比 FP32 节省 75%）
# ============================================================================
#
# 量化方案：Per-channel（per-row）对称量化
#   scale[i] = max(abs(weight[i, :])) / 127
#   weight_int8[i, k] = round(weight[i, k] / scale[i])
#   反量化：weight[i, k] ≈ weight_int8[i, k] * scale[i]

@triton.jit
def gemv_int8_kernel(
    weight_int8_ptr,  # [N, K]  int8  量化权重
    scale_ptr,        # [N]     fp32  每行 scale
    x_ptr,            # [K]     fp32  输入向量
    y_ptr,            # [N]     fp32  输出向量
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    INT8 权重量化 GEMV：
      1. 加载 INT8 权重（4× 小于 FP32）
      2. 在寄存器中乘以 per-row scale 完成反量化
      3. FP32 点积累加

    INT8 vs FP32 内存量对比（gate_proj: 3584×18944）：
      FP32: 18944 × 3584 × 4B = 271 MB
      INT8:  18944 × 3584 × 1B =  68 MB  → 节省 203 MB HBM 读取/次
    """
    pid   = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # 加载当前行的 scale
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # 加载 INT8 权重并转为 FP32
        w_int8 = tl.load(
            weight_int8_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        )
        w = w_int8.to(tl.float32) * scale[:, None]   # 反量化

        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(y_ptr + offs_n, acc, mask=mask_n)


# ============================================================================
# Python 量化工具函数
# ============================================================================

def quantize_weight_int8(weight: torch.Tensor):
    """
    Per-channel（per-row）对称 INT8 量化。
    weight: [N, K] float32
    返回: (weight_int8 [N, K] int8, scale [N] float32)
    """
    # 每行的最大绝对值
    max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale   = max_val / 127.0           # [N, 1]
    w_int8  = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return w_int8.contiguous(), scale.squeeze(1).contiguous()


# ============================================================================
# OptimizedLinear：自动分派 GEMV / GEMM，支持 FP16 / INT8 权重
# ============================================================================

GEMV_THRESHOLD = 4   # M ≤ 此值时使用 GEMV kernel

class OptimizedLinear:
    """
    直接替换原有 Linear 类，增加三级优化：
      mode="fp32"  → 专用 Triton GEMV kernel（修复 128× padding 浪费）
      mode="fp16"  → FP16 权重存储（带宽减半）
      mode="int8"  → INT8 权重量化（带宽减至 1/4）

    用法：
      lin = OptimizedLinear.from_linear(original_linear, mode="fp16")
      y   = lin(x)
    """

    BLOCK_N = 128
    BLOCK_K = 128

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mode: str = "fp16",
    ):
        self.in_features  = in_features
        self.out_features = out_features
        self.has_bias     = bias
        self.mode         = mode   # "fp32" | "fp16" | "int8"

        # 权重将在 from_linear() 或手动设置后初始化
        self.weight_fp32 = None   # [N, K] float32（原始）
        self.weight_fp16 = None   # [N, K] float16（FP16 模式）
        self.weight_int8 = None   # [N, K] int8   （INT8 模式）
        self.weight_scale= None   # [N]    float32（INT8 per-row scale）
        self.bias_param  = None

    @classmethod
    def from_linear(cls, linear, mode: str = "fp16"):
        """从已有 Linear 对象创建 OptimizedLinear。"""
        obj = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.has_bias,
            mode=mode,
        )
        obj.weight_fp32 = linear.weight.float()   # 保留 FP32 原始权重
        obj.bias_param  = linear.bias_param

        # 预转换权重（只做一次）
        if mode == "fp16":
            obj.weight_fp16 = linear.weight.half().contiguous()
        elif mode == "int8":
            w_int8, scale = quantize_weight_int8(linear.weight.float())
            obj.weight_int8  = w_int8
            obj.weight_scale = scale

        print(f"  OptimizedLinear [{linear.in_features}→{linear.out_features}] mode={mode} "
              f"weight_size={obj._weight_bytes()/1e6:.1f} MB")
        return obj

    def _weight_bytes(self):
        N, K = self.out_features, self.in_features
        if self.mode == "int8":
            return N * K * 1  # int8
        if self.mode == "fp16":
            return N * K * 2  # fp16
        return N * K * 4      # fp32

    def _to_device(self, device):
        """确保权重在正确的设备上。"""
        if self.mode == "fp16" and self.weight_fp16 is not None:
            if self.weight_fp16.device != device:
                self.weight_fp16 = self.weight_fp16.to(device)
        elif self.mode == "int8" and self.weight_int8 is not None:
            if self.weight_int8.device != device:
                self.weight_int8  = self.weight_int8.to(device)
                self.weight_scale = self.weight_scale.to(device)
        else:
            if self.weight_fp32 is not None and self.weight_fp32.device != device:
                self.weight_fp32 = self.weight_fp32.to(device)
        if self.bias_param is not None and self.bias_param.device != device:
            self.bias_param = self.bias_param.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_dims     = original_shape[:-1]
        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features

        device = x.device
        self._to_device(device)

        x_2d = x.reshape(M, K).float().contiguous()

        # ── 路由逻辑 ──────────────────────────────────────────────────────
        if not x.is_cuda or M > GEMV_THRESHOLD:
            # 大 M：回退到 torch GEMM（cuBLAS 对 GEMM 已足够高效）
            return self._forward_torch_gemm(x_2d, M, K, N, batch_dims)

        # 小 M（decode 阶段）：用专用 GEMV kernel
        if self.mode == "int8":
            return self._forward_int8_gemv(x_2d, M, K, N, batch_dims)
        if self.mode == "fp16":
            return self._forward_fp16_gemv(x_2d, M, K, N, batch_dims)
        return self._forward_fp32_gemv(x_2d, M, K, N, batch_dims)

    # ── FP32 GEMV ─────────────────────────────────────────────────────────

    def _forward_fp32_gemv(self, x_2d, M, K, N, batch_dims):
        assert M == 1, "gemv_kernel 只针对 M=1 优化"
        x_vec = x_2d.squeeze(0)   # [K]
        y_vec = torch.empty(N, dtype=torch.float32, device=x_vec.device)

        grid = (triton.cdiv(N, self.BLOCK_N),)
        gemv_kernel[grid](
            self.weight_fp32, x_vec, y_vec,
            N, K,
            self.weight_fp32.stride(0), self.weight_fp32.stride(1),
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
        )
        if self.has_bias and self.bias_param is not None:
            y_vec = y_vec + self.bias_param
        return y_vec.reshape(*batch_dims, N)

    # ── FP16 GEMV ─────────────────────────────────────────────────────────

    def _forward_fp16_gemv(self, x_2d, M, K, N, batch_dims):
        assert M == 1
        x_vec = x_2d.squeeze(0)
        y_vec = torch.empty(N, dtype=torch.float32, device=x_vec.device)

        grid = (triton.cdiv(N, self.BLOCK_N),)
        gemv_fp16_kernel[grid](
            self.weight_fp16, x_vec, y_vec,
            N, K,
            self.weight_fp16.stride(0), self.weight_fp16.stride(1),
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
        )
        if self.has_bias and self.bias_param is not None:
            y_vec = y_vec + self.bias_param
        return y_vec.reshape(*batch_dims, N)

    # ── INT8 GEMV ─────────────────────────────────────────────────────────

    def _forward_int8_gemv(self, x_2d, M, K, N, batch_dims):
        assert M == 1
        x_vec = x_2d.squeeze(0)
        y_vec = torch.empty(N, dtype=torch.float32, device=x_vec.device)

        grid = (triton.cdiv(N, self.BLOCK_N),)
        gemv_int8_kernel[grid](
            self.weight_int8, self.weight_scale, x_vec, y_vec,
            N, K,
            self.weight_int8.stride(0), self.weight_int8.stride(1),
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
        )
        if self.has_bias and self.bias_param is not None:
            y_vec = y_vec + self.bias_param
        return y_vec.reshape(*batch_dims, N)

    # ── Torch GEMM fallback（大 M 时使用 cuBLAS）─────────────────────────

    def _forward_torch_gemm(self, x_2d, M, K, N, batch_dims):
        w = self.weight_fp32
        out = x_2d @ w.t()
        if self.has_bias and self.bias_param is not None:
            out = out + self.bias_param
        return out.reshape(*batch_dims, N)


# ============================================================================
# Benchmark：对比四种 Linear 实现在 M=1 时的速度
# ============================================================================

def run_benchmark():
    import time

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过 benchmark")
        return

    device = torch.device("cuda")
    WARMUP = 50
    ITERS  = 200

    # 典型 GLM-ASR decode 场景下的两种尺寸
    configs = [
        ("q_proj  / o_proj   (3584→3584)",  3584,  3584),
        ("gate_proj (3584→18944)",           3584, 18944),
        ("down_proj (18944→3584)",          18944,  3584),
    ]

    print("\n" + "=" * 72)
    print("GEMV BENCHMARK  (batch=1, seq=1, dtype=float32 input)")
    print("=" * 72)
    print(f"  {'Layer':<38s} {'Method':<14s} {'ms/call':>8s}  {'Speedup':>8s}")
    print(f"  {'-'*38} {'-'*14} {'-'*8}  {'-'*8}")

    # 用原始 Linear 类做基准
    from layers import Linear as OriginalLinear

    for label, K, N in configs:
        x = torch.randn(1, 1, K, device=device, dtype=torch.float32)

        # ── 基准：原始 Triton GEMM（TILE_M=128 padding）─────────────────
        orig_lin = OriginalLinear(K, N, bias=False)
        orig_lin.weight = orig_lin.weight.to(device)
        for _ in range(WARMUP):
            _ = orig_lin(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = orig_lin(x)
        torch.cuda.synchronize()
        t_orig = (time.perf_counter() - t0) / ITERS * 1000

        results = [("Original GEMM", t_orig)]

        # ── cuBLAS（torch @ ）─────────────────────────────────────────────
        w_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
        x_vec  = x.reshape(1, K)
        for _ in range(WARMUP):
            _ = x_vec @ w_fp32.t()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = x_vec @ w_fp32.t()
        torch.cuda.synchronize()
        t_cublas = (time.perf_counter() - t0) / ITERS * 1000
        results.append(("cuBLAS fp32", t_cublas))

        # ── Level 1: 专用 GEMV FP32 ──────────────────────────────────────
        opt_fp32 = OptimizedLinear.from_linear(orig_lin, mode="fp32")
        for _ in range(WARMUP):
            _ = opt_fp32(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = opt_fp32(x)
        torch.cuda.synchronize()
        t_fp32 = (time.perf_counter() - t0) / ITERS * 1000
        results.append(("GEMV fp32", t_fp32))

        # ── Level 2: 专用 GEMV FP16 ──────────────────────────────────────
        opt_fp16 = OptimizedLinear.from_linear(orig_lin, mode="fp16")
        for _ in range(WARMUP):
            _ = opt_fp16(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = opt_fp16(x)
        torch.cuda.synchronize()
        t_fp16 = (time.perf_counter() - t0) / ITERS * 1000
        results.append(("GEMV fp16", t_fp16))

        # ── Level 3: 专用 GEMV INT8 ──────────────────────────────────────
        opt_int8 = OptimizedLinear.from_linear(orig_lin, mode="int8")
        for _ in range(WARMUP):
            _ = opt_int8(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = opt_int8(x)
        torch.cuda.synchronize()
        t_int8 = (time.perf_counter() - t0) / ITERS * 1000
        results.append(("GEMV int8", t_int8))

        # 打印
        for i, (name, t) in enumerate(results):
            speedup = t_orig / t
            marker = " ◀ baseline" if i == 0 else f"  {speedup:.2f}×"
            print(f"  {label if i == 0 else '':<38s} {name:<14s} {t:8.3f}  {marker}")
        print()

    # ── 内存使用对比 ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WEIGHT MEMORY FOOTPRINT (gate_proj: 3584→18944)")
    print("=" * 72)
    N, K = 18944, 3584
    for dtype, label in [(torch.float32, "FP32"), (torch.float16, "FP16"),
                          (torch.int8,   "INT8")]:
        size_mb = N * K * torch.tensor([], dtype=dtype).element_size() / 1e6
        saved   = (1 - size_mb / (N * K * 4 / 1e6)) * 100
        print(f"  {label}: {size_mb:6.1f} MB  (HBM 节省 {saved:4.0f}%)")

    # ── 量化精度损失测试 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("INT8 QUANTIZATION ERROR  (gate_proj: 3584→18944)")
    print("=" * 72)
    w_fp32 = torch.randn(18944, 3584, dtype=torch.float32)
    w_int8, scale = quantize_weight_int8(w_fp32)
    w_dequant = w_int8.float() * scale.unsqueeze(1)
    err = (w_fp32 - w_dequant).abs()
    print(f"  Max error : {err.max().item():.6f}")
    print(f"  Mean error: {err.mean().item():.6f}")
    print(f"  Relative error (||Δ||/||W||): {(err.norm() / w_fp32.norm()).item():.4%}")


# ============================================================================
# 正确性验证
# ============================================================================

def verify_correctness():
    """验证三种 GEMV kernel 的数值正确性（与 torch.matmul 对比）。"""
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过验证")
        return

    device = torch.device("cuda")
    print("\n" + "=" * 72)
    print("CORRECTNESS VERIFICATION")
    print("=" * 72)

    from layers import Linear as OriginalLinear

    for K, N, label in [(3584, 3584, "q_proj"), (3584, 18944, "gate_proj")]:
        lin = OriginalLinear(K, N, bias=False)
        lin.weight = lin.weight.to(device)
        x = torch.randn(1, 1, K, device=device, dtype=torch.float32)

        ref = lin._forward_torch(x)   # cuBLAS 参考答案

        for mode in ("fp32", "fp16", "int8"):
            opt = OptimizedLinear.from_linear(lin, mode=mode)
            out = opt(x)
            diff = (out - ref).abs().max().item()
            status = "✓ OK" if diff < 0.5 else "✗ FAIL"
            print(f"  [{label}] mode={mode:<5s}  max_diff={diff:.6f}  {status}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",    action="store_true", help="运行正确性验证")
    parser.add_argument("--benchmark", action="store_true", help="运行性能 benchmark")
    parser.add_argument("--all",       action="store_true", help="运行验证 + benchmark")
    args = parser.parse_args()

    if args.all or args.verify:
        verify_correctness()

    if args.all or args.benchmark:
        run_benchmark()

    if not any([args.verify, args.benchmark, args.all]):
        print("用法:")
        print("  python gemv_optimized.py --verify     # 验证数值正确性")
        print("  python gemv_optimized.py --benchmark  # 性能对比")
        print("  python gemv_optimized.py --all        # 两者都运行")
