import torch
import time
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
PROMPT_FILE = "prompt.txt"

NUM_RUNS = 3
SLEEP_BETWEEN = 1.0
MAX_NEW_TOKENS = 4000

LOG_DIR = Path("logs/spo02")
LOG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV_PATH = LOG_DIR / "spo02-_user_metrics.csv"
GPU_LOG_PATH = LOG_DIR / "spo02-_gpu_log.csv"

DEVICE = "cuda"

# =========================
# CSV INIT
# =========================
metrics_csv = open(METRICS_CSV_PATH, "a")
if metrics_csv.tell() == 0:
    metrics_csv.write(
        "timestamp,run,"
        "ttft_s,prefill_s,decoded_tokens,decode_time_s,"
        "throughput_tok_s,avg_itl_ms,tpot_ms\n"
    )
    metrics_csv.flush()

# =========================
# GPU LOGGING
# =========================
gpu_log = open(GPU_LOG_PATH, "w")
gpu_log.write("timestamp,util_gpu,mem_MiB,power_W,temp_C\n")
gpu_log.flush()

nvidia_smi_proc = subprocess.Popen(
    [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms=100"
    ],
    stdout=gpu_log,
    stderr=subprocess.DEVNULL
)

# =========================
# LOAD TOKENIZER + MODEL
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Loading model...")
ttlm_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True
)
torch.cuda.synchronize()
ttlm_end = time.perf_counter()

print(f"Model load time (TTML): {ttlm_end - ttlm_start:.3f} s")

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
            use_cache=True
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
                use_cache=True
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
    throughput = decode_token_count / decode_time
    tpot = 1.0 / throughput
    avg_itl = sum(inter_token_latencies) / len(inter_token_latencies)

    # =========================
    # PRINT
    # =========================
    print(f"Prefill time: {prefill_time:.3f} s")
    print(f"TTFT: {ttft:.3f} s")
    print(f"Decoded tokens: {decode_token_count}")
    print(f"Decode time: {decode_time:.3f} s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"Avg ITL: {avg_itl*1000:.2f} ms")
    print(f"TPOT: {tpot*1000:.2f} ms/token")

    # =========================
    # CSV LOG (PER RUN)
    # =========================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_csv.write(
        f"{timestamp},{run_idx+1},"
        f"{ttft:.6f},{prefill_time:.6f},{decode_token_count},"
        f"{decode_time:.6f},{throughput:.3f},"
        f"{avg_itl*1000:.3f},{tpot*1000:.3f}\n"
    )
    metrics_csv.flush()

    time.sleep(SLEEP_BETWEEN)

# =========================
# CLEANUP
# =========================
metrics_csv.close()
nvidia_smi_proc.terminate()
gpu_log.close()
