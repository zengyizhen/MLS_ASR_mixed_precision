# GLM-ASR Triton Implementation — Optimization Notes

## Key Optimizations

- **FlashAttention** (`attention.py`): Replaces the baseline three-kernel attention path with a fused Triton kernel that maintains online softmax state in SRAM, reducing HBM memory complexity from O(N²) to O(N).

- **Fused Add + RMSNorm** (`layers.py`): Combines residual addition and RMSNorm into a single kernel (`fused_add_rmsnorm_kernel`), performing the residual update in-place to eliminate one HBM read-write round-trip and one kernel launch per decoder layer.

- **FP16 GEMV for Decoder** (`layers.py`): Introduces a dedicated GEMV kernel for the M=1 autoregressive decode path, storing weights natively in FP16 to halve weight-load bandwidth and avoid the structural inefficiency of a tiled GEMM kernel at single-token batch size.

- **Tile Size and Warp/Stage Tuning** (`layers.py`): `linear_kernel_tf32` is configured with `BLOCK_M=128`, `BLOCK_N=64`, `BLOCK_K=32`, `num_warps=4`, `num_stages=3`, selected via ablation study on the Encoder FC1 layer. `num_stages=3` enables software pipelining to hide HBM latency; `num_warps=4` is preferred over 8 as both yield equivalent throughput at this stage count.

## Benchmarking Modification

A warmup pass was added to `benchmark_detailed.py` before timed measurements begin. This executes one full forward pass through all model components to trigger Triton JIT compilation and CUDA initialization, ensuring that reported latencies reflect steady-state inference performance rather than first-run overhead.
