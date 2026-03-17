# Qwen 2.5 1.5B Laptop Inference Benchmark

Benchmark suite for measuring inference performance of **Qwen/Qwen2.5-1.5B-Instruct** on a laptop/desktop with an NVIDIA GPU. Four backends are covered: raw PyTorch (FP16 and Q4-NF4), GPTQ-Int4 via Transformers, vLLM, and Ollama.

Each script runs a fixed prompt (~500 input tokens, up to 4 000–15 000 output tokens), measures latency/throughput/power, and appends results to a CSV file under `logs/`.

---

## Hardware Requirements

| Requirement | Notes |
|---|---|
| NVIDIA GPU | `nvidia-smi` must be available |
| ≥ 8 GB VRAM | FP16 run needs ~6 GB; all others fit in 4–6 GB |
| Intel CPU | `sensors` + Intel RAPL (`/sys/class/powercap/`) used by the Ollama script; gracefully degrades if absent |
| Linux | Tested on Ubuntu 22.04 / 24.04 |

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd qwen-ws
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies per script

| Script | Extra packages |
|---|---|
| `run_qwen_1_5b_single.py` | `pip install torch transformers accelerate` |
| `run_spo02.py` (GPTQ) | `pip install torch transformers accelerate auto-gptq optimum` |
| `run_spo02_q4.py` (Q4 NF4) | `pip install torch transformers accelerate bitsandbytes` |
| `run_spo03_vllm.py` (vLLM) | `pip install vllm` |
| `run_sp04_ollama.py` (Ollama) | Install [Ollama](https://ollama.com) separately, then `pip install requests` |

> HuggingFace models are downloaded automatically on first run (~3 GB).
> For the Ollama script, pull the model first: `ollama pull qwen2.5:1.5b`

---

## Benchmark Scripts

### `run_qwen_1_5b_single.py` — Direct PyTorch, FP16

Loads `Qwen/Qwen2.5-1.5B-Instruct` in `float16` via Transformers and runs a single generation pass with a manual token-by-token decode loop. Captures per-token inter-token latency (ITL) and streams GPU metrics via `nvidia-smi --loop-ms=100`.

**Output:** `logs/sp02/sp02_user_metrics.csv`, `logs/sp02/sp02_gpu_log.csv`

```bash
python run_qwen_1_5b_single.py
```

**Metrics recorded:** `generated_tokens`, `decoded_tokens`, `decode_time_s`, `throughput_tok_s`, `avg/min/max_itl_ms`, `tpot_ms`

---

### `run_spo02.py` — GPTQ-Int4, 3 runs

Loads `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` and runs **3 consecutive generations** (model stays loaded). Measures TTFT, prefill time, decode throughput, and average ITL per run.

**Output:** `logs/spo02/spo02-_user_metrics.csv`, `logs/spo02/spo02-_gpu_log.csv`

```bash
python run_spo02.py
```

**Metrics recorded:** `run`, `ttft_s`, `prefill_s`, `decoded_tokens`, `decode_time_s`, `throughput_tok_s`, `avg_itl_ms`, `tpot_ms`

---

### `run_spo02_q4.py` — Q4 NF4 (BitsAndBytes), 20 runs

Loads `Qwen/Qwen2.5-1.5B-Instruct` quantized on-the-fly to **4-bit NF4** via BitsAndBytes. Runs **20 generations** and samples GPU power/temperature in a background thread every 100 ms.

**Output:** `logs/spo02_q4/spo02_q4_metrics.csv`

```bash
python run_spo02_q4.py
```

**Metrics recorded:** `run`, `load_time_s`, `ttft_s`, `prefill_s`, `decoded_tokens`, `decode_time_s`, `throughput_tok_s`, `avg_itl_ms`, `tpot_ms`, `avg/max_power_W`, `avg/max_temp_C`

---

### `run_spo03_vllm.py` — vLLM, GPTQ-Int4, 20 runs

Uses the **vLLM** engine with `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`. Runs **20 generations** and tracks energy per token, battery level (reads `/sys/class/power_supply/BAT1/capacity` — reports `-1` on desktop), CPU package temperature, and GPU temperature/power.

**Output:** `logs/spo03_vllm/spo03_vllm_metrics.csv`, `logs/spo03_vllm/spo03_vllm_gpu_log.csv`

```bash
python run_spo03_vllm.py
```

**Metrics recorded:** `run`, `total_tokens`, `decode_tokens`, `decode_ms`, `total_ms`, `decode_tok_s`, `battery_start/end_%`, `max_cpu/gpu_temp_c`, `initial/final/avg/peak_power_mw`, `energy_per_token_mj`

---

### `run_sp04_ollama.py` — Ollama cold start

Performs a **true cold-start** benchmark using the Ollama service (`qwen2.5:1.5b`). Before each run the script kills the Ollama process, restarts it, waits for the API to be live, then sends the prompt. Logs TTML (time-to-model-loaded), TTFT, prefill time, and ITL alongside GPU and CPU power via a background shell logger.

**Requirements:** `ollama` binary in `PATH`; `sensors` (lm-sensors) and Intel RAPL for CPU metrics.

**Output:** `logs/sp04/sp04_laptop_metrics.csv`, `logs/sp04/sp04_laptop_system_metrics.csv`

```bash
python run_sp04_ollama.py
```

**Metrics recorded:** `ttml_s`, `ttft_user_s`, `ttft_compute_s`, `ttft_after_load_s`, `first_decode_s`, `prompt_tokens`, `prefill_time_s`, `decoded_tokens`, `decode_time_s`, `throughput_tok_s`, `tpot_ms`, `avg/min/max_itl_ms`

---

## Prompt

All scripts read from `prompt.txt` — a multi-turn chat-formatted prompt (~500 tokens) asking for a comprehensive essay on consciousness. You can swap in any prompt; just ensure it follows the `<|im_start|>` / `<|im_end|>` chat format expected by Qwen Instruct models.

---

## Output Format

Results are appended to CSV files in `logs/<benchmark>/`. The `logs/` directory is tracked in git (via `.gitkeep` files) but generated CSVs are gitignored. Each row corresponds to one generation run, timestamped for easy correlation with external monitoring tools.

---

## Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time To First Token — latency from request send to receiving the first output token |
| **TTML** | Time To Model Load — time to load model weights into GPU VRAM |
| **ITL** | Inter-Token Latency — wall-clock time between consecutive output tokens |
| **TPOT** | Time Per Output Token — reciprocal of decode throughput (ms/token) |
| **Throughput** | Decode tokens per second |
| **Prefill time** | Time to process the input prompt (KV cache construction) |
| **Energy/token** | Average GPU energy consumed per decoded token (mJ) |
