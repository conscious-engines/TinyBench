# TinyBench

A cross-platform benchmarking suite for sustained LLM inference on edge and consumer hardware. Measures throughput, latency, power, and thermal behaviour of small language models under repeated warm-condition load — the conditions that matter for always-on personal agents.

This repository accompanies the paper:

> **LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load**
> Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan — Conscious Engines

---

## Overview

Existing benchmarks report peak or single-shot performance. TinyBench characterises **sustained** inference: each platform runs 20 back-to-back iterations of a fixed 250-token prompt using **Qwen 2.5 1.5B Instruct (4-bit quantised)**, capturing per-iteration throughput, power draw, temperature, and thermal state across the full run.

### Platforms Evaluated

| Platform | Accelerator | Inference Stack | Model Format |
|---|---|---|---|
| Raspberry Pi 5 | Hailo-10H NPU (40 TOPS, <5 W) | hailo-ollama + HailoRT 4.17 | GGUF Q4_0 |
| Samsung Galaxy S24 Ultra | Adreno 750 GPU (45 TOPS) | MLC-LLM 0.1.0 / OpenCL + TVM | MLC binary Q4f16_2 |
| iPhone 16 Pro | Apple A18 Pro GPU (Metal) | MLX Swift v0.29.1 | MLX safetensors Q4_0 |
| RTX 4050 Laptop | NVIDIA RTX 4050 (80 Tensor Cores) | vLLM + PyTorch / CUDA 12.1 | GPTQ Int4 safetensors |

### Key Findings

- **RTX 4050** sustains **131.7 tok/s** at 34.1 W (CV = 2.2%) — thermally stable with no throttling observed over 20 runs.
- **Hailo-10H NPU** sustains **6.9 tok/s** at under 2 W with near-zero variance (CV = 0.04%) — the most thermally stable platform. Matches the RTX 4050 in energy proportionality at 19× lower throughput.
- **iPhone 16 Pro** peaks at ~40 tok/s but loses 44% throughput by iteration 8 as the device enters a sustained Hot thermal state (22.6 tok/s steady-state). The 1-second inter-iteration gap is insufficient for thermal recovery.
- **Samsung Galaxy S24 Ultra** runs only 5 valid iterations before Android's thermal governor hard-floors the Adreno 750 GPU frequency to 231 MHz (down from ~650 MHz), effectively terminating representative inference.
- **Thermal management, not peak compute**, is the binding constraint for sustained mobile inference.

---

## Repository Structure

```
edge-benchmarking/
├── rtx4050-laptop/  # Laptop/desktop RTX 4050 — vLLM + multiple backends
├── rpi5-hailo/      # Raspberry Pi 5 + Hailo-10H NPU (hailo-ollama)
├── mlc-llm/         # Samsung Galaxy S24 Ultra — MLC-LLM Android app (forked)
├── mlx-llm/         # iPhone 16 Pro — MLX Swift iOS benchmarking app
└── report/          # Full paper (PDF)
```

---

## Benchmark Suites

### 1. `rtx4050-laptop/` — NVIDIA RTX 4050 Laptop GPU

Benchmarks **Qwen 2.5 1.5B Instruct** on a Linux laptop with an RTX 4050 GPU via vLLM (CUDA 12.1 / PyTorch 2.1). Runs 1 warm-up + 20 timed iterations. GPU power and utilisation logged via `nvidia-smi` at 100 ms intervals. Benchmarks conducted on battery, sustaining ~34 W under inference.

**Target hardware:** Linux + NVIDIA RTX 4050 (Acer Nitro V), Intel Core i7-13700H, 32 GB DDR5

**Metrics:** TTFT, prefill time, decode throughput (tok/s), avg/peak system power (W), energy per token (mJ), GPU/CPU temperature (°C), battery drain.

See [`rtx4050-laptop/README.md`](rtx4050-laptop/README.md) for setup and usage.

---

### 2. `rpi5-hailo/` — Raspberry Pi 5 + Hailo-10H NPU

Benchmarks **Qwen 2.5 1.5B Instruct** on a Raspberry Pi 5 with a Hailo-10H M.2 NPU module via hailo-ollama. The NPU handles matrix-multiply layers; attention runs on the ARM Cortex-A76. Power is estimated from the RPi 5 PMIC's voltage rails via INA219 sensor at 1 kHz.

**Target hardware:** Raspberry Pi 5 (BCM2712, 4×A76 @ 2.4 GHz, 8 GB LPDDR4X) + Hailo-10H M.2

**Metrics:** TTFT, prefill time, decode throughput (tok/s), avg/peak system power (W), energy per token (mJ), NPU/CPU temperature (°C).

See [`rpi5-hailo/README.md`](rpi5-hailo/README.md) for setup and usage.

---

### 3. `mlc-llm/` — Samsung Galaxy S24 Ultra (Android / MLC-LLM)

A fork of the [MLC-LLM](https://github.com/mlc-ai/mlc-llm) Android app extended with an automated headless benchmarking service (`BenchmarkService`). Targets the **Snapdragon 8 Gen 3 / Adreno 750** via OpenCL + Apache TVM. Runs 1 warm-up + 20 timed iterations on app launch and exports per-iteration telemetry to CSV.

**Target hardware:** Samsung Galaxy S24 Ultra (SM-S928B/U), Android 16, arm64-v8a

**Model:** Qwen2.5-1.5B-Instruct compiled to MLC binary (Q4f16_2) for OpenCL

**Metrics:** prefill time, decode throughput (tok/s), GPU temperature/frequency (Adreno sysfs), CPU temperature, battery drain.

**Note:** Android Battery Manager API power figures were found unreliable under GPU load and are excluded from analysis.

**Build requirements:** Python ≥ 3.9, JDK 17, Android Studio 2023+, NDK r26b, CMake ≥ 3.18, Rust with `aarch64-linux-android` target.

See [`mlc-llm/README.md`](mlc-llm/README.md) for the full build and deployment guide.

---

### 4. `mlx-llm/` — iPhone 16 Pro (iOS / MLX)

A custom SwiftUI benchmarking application (`MLXBenchmark`) using MLX Swift (v0.29.1) with inference dispatched to the Apple A18 Pro GPU via Metal compute kernels. The Neural Engine is not utilised by MLX. Thermal state is monitored via `ProcessInfo.thermalState` and battery level via `UIDevice.current.batteryLevel` at each iteration boundary. Token generation is capped at 1,000 tokens on iOS.

**Target hardware:** iPhone 16 Pro, Apple A18 Pro (6-core GPU), 8 GB Unified RAM, iOS 26

**Model:** Qwen2.5-1.5B-Instruct — MLX safetensors Q4_0

**Metrics:** decode throughput (tok/s), thermal state (Normal / Warm / Hot), battery drain (% SoC).

**Note:** iOS does not expose per-component power draw to third-party applications; battery state-of-charge is the sole energy proxy.

See [`mlx-llm/README.md`](mlx-llm/README.md) for setup and usage.

---

## Experimental Protocol

Each device undergoes the following procedure independently:

1. Reboot and equilibrate for 10 minutes at ambient temperature (22°C ± 2°C).
2. Load model into memory and discard one warm-up inference.
3. Verify thermal stability (ΔT < 2°C over 60 s) before starting timed runs.
4. Execute 20 inference iterations with a fixed 250-token prompt and 1 s inter-iteration gap.
5. Export per-iteration metrics to CSV; validate for token count anomalies.

Cold-start model loading time is recorded where observable but excluded from throughput analysis.

---

## Metrics Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time To First Token — latency from prompt submission to first output token (prefill duration) |
| **Throughput (TPS)** | Decode tokens per second — N_decode / t_decode |
| **Avg power** | Mean system power draw over inference duration (W) |
| **Peak power** | Maximum instantaneous power observed (W) |
| **Energy/token** | P_avg × t_decode / N_decode (mJ) |
| **Thermal state** | iOS categorical signal: Normal, Warm, or Hot |
| **Battery drain** | Change in state-of-charge over the full benchmark run (% SoC) |
| **GPU frequency** | Observed GPU clock frequency per iteration — Android (Adreno sysfs) only |
