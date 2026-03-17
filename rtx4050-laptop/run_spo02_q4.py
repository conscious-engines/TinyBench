import torch
import time
import threading
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_FILE = "prompt.txt"

NUM_RUNS = 20
SLEEP_BETWEEN = 1.0
MAX_NEW_TOKENS = 15000

LOG_DIR = Path("logs/spo02_q4")
LOG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV_PATH = LOG_DIR / "spo02_q4_metrics.csv"

DEVICE = "cuda"
GPU_SAMPLE_INTERVAL_MS = 100

# =========================
# GPU SAMPLER (background thread)
# =========================
class GPUSampler:
    """Samples nvidia-smi every GPU_SAMPLE_INTERVAL_MS ms in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._power_samples: list[float] = []
        self._temp_samples: list[float] = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=1,
                ).decode().strip()
                parts = out.split(",")
                power = float(parts[0].strip())
                temp = float(parts[1].strip())
                with self._lock:
                    self._power_samples.append(power)
                    self._temp_samples.append(temp)
            except Exception:
                pass
            time.sleep(GPU_SAMPLE_INTERVAL_MS / 1000.0)

    def reset(self):
        with self._lock:
            self._power_samples.clear()
            self._temp_samples.clear()

    def stats(self):
        with self._lock:
            p = list(self._power_samples)
            t = list(self._temp_samples)
        avg_pow = sum(p) / len(p) if p else float("nan")
        max_pow = max(p) if p else float("nan")
        avg_tmp = sum(t) / len(t) if t else float("nan")
        max_tmp = max(t) if t else float("nan")
        return avg_pow, max_pow, avg_tmp, max_tmp


gpu_sampler = GPUSampler()
gpu_sampler.start()

# =========================
# CSV INIT (single file)
# =========================
metrics_csv = open(METRICS_CSV_PATH, "a")
if metrics_csv.tell() == 0:
    metrics_csv.write(
        "timestamp,run,"
        "load_time_s,"
        "ttft_s,prefill_s,decoded_tokens,decode_time_s,"
        "throughput_tok_s,avg_itl_ms,tpot_ms,"
        "avg_power_W,max_power_W,avg_temp_C,max_temp_C\n"
    )
    metrics_csv.flush()

# =========================
# LOAD TOKENIZER + MODEL (Q4 NF4)
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Loading model (Q4 NF4)...")
gpu_sampler.reset()
load_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
torch.cuda.synchronize()
load_end = time.perf_counter()
load_time = load_end - load_start
print(f"Model load time: {load_time:.3f} s")

# =========================
# LOAD PROMPT
# =========================
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# =========================
# BENCHMARK LOOP
# =========================
for run_idx in range(NUM_RUNS):
    print(f"\n===== RUN {run_idx + 1}/{NUM_RUNS} =====")

    gpu_sampler.reset()
    generated_tokens = []

    with torch.no_grad():
        # -----------------
        # PREFILL
        # -----------------
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()

        outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        past_kv = outputs.past_key_values

        # -----------------
        # FIRST TOKEN (TTFT)
        # -----------------
        ttft_start = prefill_start

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        generated_tokens.append(next_token.item())

        torch.cuda.synchronize()
        outputs = model(
            input_ids=next_token,
            past_key_values=past_kv,
            use_cache=True,
        )
        torch.cuda.synchronize()

        ttft_end = time.perf_counter()
        ttft = ttft_end - ttft_start

        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        # -----------------
        # DECODE LOOP
        # -----------------
        torch.cuda.synchronize()
        decode_start = time.perf_counter()

        decode_token_count = 0
        inter_token_latencies = []

        for _ in range(MAX_NEW_TOKENS - 1):
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            decode_token_count += 1

            torch.cuda.synchronize()
            itl_start = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )

            torch.cuda.synchronize()
            itl_end = time.perf_counter()
            inter_token_latencies.append(itl_end - itl_start)

            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if next_token.item() == tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        decode_end = time.perf_counter()

    # =========================
    # METRICS
    # =========================
    decode_time = decode_end - decode_start
    throughput = decode_token_count / decode_time if decode_time > 0 else 0
    tpot = 1.0 / throughput if throughput > 0 else float("nan")
    avg_itl = sum(inter_token_latencies) / len(inter_token_latencies) if inter_token_latencies else 0

    avg_power, max_power, avg_temp, max_temp = gpu_sampler.stats()

    # =========================
    # PRINT
    # =========================
    print(f"Prefill time   : {prefill_time:.3f} s")
    print(f"TTFT           : {ttft:.3f} s")
    print(f"Decoded tokens : {decode_token_count}")
    print(f"Decode time    : {decode_time:.3f} s")
    print(f"Throughput     : {throughput:.2f} tok/s")
    print(f"Avg ITL        : {avg_itl*1000:.2f} ms")
    print(f"TPOT           : {tpot*1000:.2f} ms/token")
    print(f"Avg power      : {avg_power:.2f} W   Max power: {max_power:.2f} W")
    print(f"Avg temp       : {avg_temp:.1f} C   Max temp : {max_temp:.1f} C")

    # =========================
    # CSV LOG (single file, per run)
    # =========================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_csv.write(
        f"{timestamp},{run_idx+1},"
        f"{load_time:.6f},"
        f"{ttft:.6f},{prefill_time:.6f},{decode_token_count},"
        f"{decode_time:.6f},{throughput:.3f},"
        f"{avg_itl*1000:.3f},{tpot*1000:.3f},"
        f"{avg_power:.3f},{max_power:.3f},{avg_temp:.3f},{max_temp:.3f}\n"
    )
    metrics_csv.flush()

    time.sleep(SLEEP_BETWEEN)

# =========================
# CLEANUP
# =========================
metrics_csv.close()
gpu_sampler.stop()
print("\nDone. Results saved to:", METRICS_CSV_PATH)
