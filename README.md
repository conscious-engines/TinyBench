# Edge Benchmarking

A cumulative collection of LLM inference benchmarking suites targeting edge and consumer hardware. Each sub-project measures latency, throughput, and power metrics for running small language models on a specific platform or inference backend.

---

## Repository Structure

```
edge-benchmarking/
├── rtx4050-laptop/  # Laptop/desktop NVIDIA GPU — multiple inference backends
├── rpi5-hailo/      # Raspberry Pi 5 + Hailo AI accelerator
└── mlc-llm/         # On-device benchmarking via MLC-LLM (Android + iOS)
```

---

## Benchmark Suites

### 1. `rtx4050-laptop/` — Laptop / Desktop (NVIDIA GPU)

Benchmarks **Qwen 2.5 1.5B Instruct** across five inference backends on a Linux machine with an NVIDIA GPU. Each script runs a fixed ~500-token prompt, generates up to 4 000–15 000 tokens, and logs latency, throughput, and power metrics to CSV files under `logs/`.

**Target hardware:** Linux + NVIDIA GPU (≥ 8 GB VRAM recommended)

| Script | Backend | Quantization |
|---|---|---|
| `run_qwen_1_5b_single.py` | PyTorch / Transformers | FP16 |
| `run_spo02.py` | Transformers + AutoGPTQ | GPTQ-Int4 (3 runs) |
| `run_spo02_q4.py` | Transformers + BitsAndBytes | Q4 NF4 (20 runs) |
| `run_spo03_vllm.py` | vLLM | GPTQ-Int4 (20 runs) |
| `run_sp04_ollama.py` | Ollama (cold-start) | INT4 via Ollama |

**Metrics:** TTFT, TTML, prefill time, decode throughput (tok/s), ITL, TPOT, GPU/CPU power & temperature, energy per token.

See [`rtx4050-laptop/README.md`](rtx4050-laptop/README.md) for full setup and usage instructions.

---

### 2. `mlc-llm/android/` — Samsung Galaxy S24 Ultra (Android)

A fork of the [MLC-LLM](https://github.com/mlc-ai/mlc-llm) Android app extended with an automated benchmarking service. Targets the **Samsung Galaxy S24 Ultra** (Snapdragon 8 Gen 3, Adreno 750, OpenCL). Runs 1 warmup + 20 timed iterations automatically on app launch and exports per-iteration telemetry to `benchmark_log.csv` on device.

**Target hardware:** Samsung Galaxy S24 Ultra (SM-S928B/U), arm64-v8a, Android API 33–35

**Model:** `Qwen2.5-1.5B-Instruct` compiled for OpenCL (Q4 quantized)

**Metrics:** prefill latency, decode tokens/sec, battery level/temperature, CPU temperature, GPU temperature/frequency (Adreno sysfs), instantaneous/average/peak power (mW), energy per token (mJ/tok).

**Build requirements:** Python ≥ 3.9, JDK 17, Android Studio (2023+), NDK r26+, CMake ≥ 3.18, Rust with `aarch64-linux-android` target.

See [`mlc-llm/README.md`](mlc-llm/README.md) for the full build and deployment guide.

---

### 3. `mlc-llm/ios/` — iPhone (iOS)

MLC-LLM iOS app configurations targeting iPhone. Two Xcode projects are included:

| Project | Description |
|---|---|
| `MLCChat/` | Full chat UI app with model download and management |
| `MLCEngineExample/` | Minimal MLCSwift API example |

**Configured models (MLCChat):**

| Model | Quantization | Est. VRAM |
|---|---|---|
| Llama 3.2 3B Instruct | q4f16_1 | ~3 GB |
| Gemma 2 2B IT | q4f16_1 | ~3 GB |
| Phi 3.5 Mini Instruct | q4f16_1 | ~3 GB |
| Qwen3 0.6B | q0f16 | ~3 GB |
| Qwen3 1.7B | q4f16_1 | ~3 GB |

**Build:** Run `mlc_llm package` then open the `.xcodeproj` in Xcode. See the [MLC-LLM iOS documentation](https://llm.mlc.ai/docs/deploy/ios.html) for full instructions.

---

### 4. `rpi5-hailo/` — Raspberry Pi 5 + Hailo AI Accelerator

Benchmarks **Qwen 2.5 1.5B Instruct** on a **Raspberry Pi 5** with a Hailo AI accelerator. Covers the Hailo inference backend (streaming and non-streaming) and a CPU-baseline cold-start via Ollama. Platform metrics are read from `vcgencmd` (temperature, CPU frequency, input voltage, core voltage, throttle status).

**Target hardware:** Raspberry Pi 5 + Hailo M.2 HAT/HAT+

| Script | Backend | Runs |
|---|---|---|
| `run_sl01_rpi.py` | Hailo Ollama API (streaming) | 200 |
| `run_spo03_rpi.py` | Hailo Ollama API (non-streaming) | 20 |
| `run_sp03_rpi.py` | Ollama CPU cold-start (baseline) | 1 |

**Metrics:** TTFT, TTML, prefill time, decode throughput (tok/s), ITL, TPOT, CPU temperature/frequency, input voltage, core voltage, throttle status.

See [`rpi5-hailo/README.md`](rpi5-hailo/README.md) for full setup and usage instructions.

---

## Common Metrics Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time To First Token — latency from request send to first output token |
| **TTML** | Time To Model Load — time to load model weights into accelerator memory |
| **ITL** | Inter-Token Latency — wall-clock time between consecutive output tokens |
| **TPOT** | Time Per Output Token — decode latency in ms/token |
| **Throughput** | Decode tokens per second |
| **Prefill time** | Time to process the input prompt (KV cache construction) |
| **Energy/token** | Average energy consumed per decoded output token (mJ) |
