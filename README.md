# Distributed Training Profiler

A lightweight research prototype to analyze multi-GPU communication performance and throughput bottlenecks during distributed model training.
This project was inspired by my work on telemetry-driven ML systems optimization and aims to explore scalable performance profiling techniques for modern deep learning frameworks.

# 🚀 Overview

Modern large-scale training often faces inefficiencies due to communication overhead between GPUs.
This profiler monitors NCCL, PyTorch DistributedDataParallel (DDP), and TensorBoard metrics to visualize:

GPU utilization and interconnect bandwidth (NVLink, PCIe)

Gradient synchronization time across ranks

Step latency and throughput under different batch sizes

Communication vs. computation ratio per iteration

# ⚙️ Implementation

Framework: PyTorch 2.x, CUDA 12.x

Distributed backend: NCCL with multi-GPU configuration

Visualization: TensorBoard + Matplotlib plots

Test model: ResNet-50 (ImageNet subset, synthetic data)

# Example usage
torchrun --nproc_per_node=4 train_profiler.py --batch-size 256 --epochs 3 --log-dir logs/

Metrics are automatically logged under /logs/ and summarized as:

communication_time_per_step

throughput_images_per_sec

gpu_memory_utilization

latency_histogram.png

# 🧠 Key Insights (Sample Results)

4×A100 setup achieved ~18% improvement in throughput after optimizing bucket_cap_mb and gradient accumulation settings.

Profiling revealed ~23% step time spent in gradient synchronization — mitigated via fused optimizer strategy.

# 🧩 Future Work

Integrate XLA and TensorRT backends for compiler-level optimization

Extend to JAX and DeepSpeed for broader benchmarking

Automate run comparison with MLflow for reproducibility

# 📚 Skills & Concepts Demonstrated

Distributed Deep Learning (Data/Model Parallelism, NCCL)

Performance Profiling & System Bottleneck Analysis

CUDA Optimization, GPU Memory Management

Scalable Experimentation (AWS EC2/SageMaker)

PyTorch DDP, TensorBoard, MLflow Integration

# 📄 License

MIT License © 2025 Partha Yashraj

✅ This repository serves as a proof-of-concept and ongoing exploration into optimizing large-scale deep learning systems.
