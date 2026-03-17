# Raspberry Pi 5 + Hailo Inference Benchmark

Benchmark suite for running **Qwen 2.5 1.5B Instruct** on a **Raspberry Pi 5** equipped with a **Hailo AI accelerator**. Two backends are covered: the Hailo inference server (via its Ollama-compatible API) and stock Ollama (CPU-only, for comparison). Scripts log per-run latency, throughput, and platform-specific power/thermal metrics to CSV files under `logs/`.

---

## Hardware Requirements

| Requirement | Notes |
|---|---|
| Raspberry Pi 5 | Tested on RPi 5 (8 GB) |
| Hailo AI accelerator | Connected via M.2 HAT or HAT+; `hailo-ollama` service must be installed |
| Ollama | Required by `run_sp03_rpi.py` for CPU-baseline comparison |
| `vcgencmd` | Pre-installed on Raspberry Pi OS; used for temperature, frequency, and voltage readings |

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd rpi5-hailo
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests
```

### 3. Start the Hailo server (for Hailo scripts)

```bash
sudo systemctl start hailo-ollama
```

Verify it is reachable before running any Hailo benchmark:

```bash
curl http://localhost:8000/hailo/v1/list
```

### 4. Ollama setup (for `run_sp03_rpi.py` only)

```bash
# Install Ollama, then pull the model
ollama pull qwen2.5:1.5b
```

---

## Benchmark Scripts

### `run_sl01_rpi.py` — Hailo (streaming), 200 runs

Sends a streaming chat request to the Hailo Ollama-compatible API (`http://localhost:8000/api/chat`) and records per-token timing. Runs **200 consecutive iterations** with a 1-second inter-run sleep. Reads platform power and thermal metrics via `vcgencmd` after each run.

**Model:** `qwen2:1.5b`

**Output:** `logs/sl01/sl01_hailo_per_run_metrics.csv`

```bash
python run_sl01_rpi.py
```

**Metrics recorded:** `ttft_s`, `decode_time_s`, `total_latency_s`, `token_count`, `decode_throughput_tok_s`, `cpu_temp_c`, `cpu_freq_mhz`, `input_voltage_v`, `core_voltage_v`, `throttled`

---

### `run_spo03_rpi.py` — Hailo (non-streaming), 20 runs

Sends a non-streaming chat request to the Hailo Ollama-compatible API. Because the API returns the full response at once, TTFT is reported as the full round-trip latency (an approximation). Runs **20 iterations** with a 1-second inter-run sleep.

**Model:** `qwen2.5-instruct:1.5b`

**Output:** `logs/spo03/spo03_hailo_per_run_metrics.csv`

```bash
python run_spo03_rpi.py
```

**Metrics recorded:** `ttft_s` (approx), `total_latency_s`, `approx_tokens`, `throughput_tok_s`, `cpu_temp_c`

---

### `run_sp03_rpi.py` — Ollama cold-start (CPU baseline), 1 run

Performs a **true cold-start** benchmark using the system Ollama service on the RPi CPU. The script stops Ollama, restarts it, waits for the API to be live, then sends a single request. Uses Ollama's native `done` stats for server-truth TTML, prefill time, and decode time.

**Model:** `qwen2.5:1.5b`

**Output:** `logs/sp03/sp03_hailo_metrics.csv`

```bash
python run_sp03_rpi.py
```

> **Note:** Requires `sudo` for `systemctl stop/start ollama`.

**Metrics recorded:** `ttml_s`, `ttft_user_s`, `ttft_after_load_s`, `first_decode_latency_s`, `prompt_tokens`, `prefill_time_s`, `decoded_tokens`, `decode_time_s`, `throughput_tok_s`, `tpot_ms`, `avg/min/max_itl_ms`

---

## Prompt

All scripts read from `prompt.txt` — a chat-formatted prompt (~500 tokens) asking for a structured essay on consciousness. Follows the `<|im_start|>` / `<|im_end|>` format expected by Qwen Instruct models. You can replace it with any compatible prompt.

---

## Output Format

Results are appended to CSV files under `logs/<benchmark>/`. Each row corresponds to one generation run, timestamped for correlation with external monitoring.

---

## Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time To First Token — latency from request send to first output token |
| **TTML** | Time To Model Load — time to load model weights into accelerator memory |
| **ITL** | Inter-Token Latency — wall-clock time between consecutive output tokens |
| **TPOT** | Time Per Output Token — decode latency in ms/token |
| **Throughput** | Decode tokens per second |
| **Prefill time** | Time to process the input prompt (KV cache construction) |
| **Throttled** | `vcgencmd get_throttled` bitmask — non-zero indicates thermal or voltage throttling |
