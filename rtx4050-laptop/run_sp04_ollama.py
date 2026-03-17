import time
import json
import requests
import subprocess
import socket
from datetime import datetime
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"
PROMPT_FILE = "prompt.txt"

LOG_DIR = Path("logs/sp04")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = LOG_DIR / "sp04_laptop_metrics.csv"
SYS_CSV_PATH = LOG_DIR / "sp04_laptop_system_metrics.csv"

OLLAMA_STARTUP_TIMEOUT = 60  # seconds

# ---------------------------
# SYSTEM METRICS LOGGER
# ---------------------------
sys_csv = open(SYS_CSV_PATH, "w")
sys_csv.write(
    "timestamp,"
    "gpu_util_pct,gpu_mem_mb,gpu_power_w,gpu_temp_c,"
    "cpu_temp_c,cpu_energy_uj\n"
)
sys_csv.flush()

def start_system_logger():
    return subprocess.Popen(
        ["bash", "-c", r"""
while true; do
  TS=$(date +"%Y-%m-%d %H:%M:%S")

  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu \
        --format=csv,noheader,nounits)

  CPU_TEMP=$(sensors | grep -m1 'Package id 0:' | awk '{print $4}' | tr -d '+°C')

  CPU_PWR=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null)

  echo "$TS,$GPU,$CPU_TEMP,$CPU_PWR"
  sleep 0.2
done
"""],
        stdout=sys_csv,
        stderr=subprocess.DEVNULL
    )

# ---------------------------
# UTILS
# ---------------------------
def port_open(port=11434):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

# ---------------------------
# OLLAMA CONTROL (USER PROCESS)
# ---------------------------
def stop_ollama():
    print("Stopping Ollama (user process)...")
    subprocess.run(["pkill", "-x", "ollama"], check=False)

    start = time.perf_counter()
    while port_open():
        if time.perf_counter() - start > 10:
            raise RuntimeError("Port 11434 did not free")
        time.sleep(0.2)

def start_ollama():
    print("Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def wait_for_ollama():
    print("Waiting for Ollama API...")
    start = time.perf_counter()
    while time.perf_counter() - start < OLLAMA_STARTUP_TIMEOUT:
        try:
            r = requests.get("http://localhost:11434")
            if r.status_code == 200:
                print("Ollama API is live.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.3)
    raise RuntimeError("Ollama did not start in time")

# ---------------------------
# CSV INIT
# ---------------------------
new_file = not CSV_PATH.exists()
csv = open(CSV_PATH, "a")

if new_file:
    csv.write(
        "timestamp,"
        "ttml_s,ttft_user_s,ttft_compute_s,ttft_after_load_s,first_decode_s,"
        "prompt_tokens,prefill_time_s,"
        "decoded_tokens,decode_time_s,"
        "throughput_tok_s,tpot_ms,"
        "avg_itl_ms,min_itl_ms,max_itl_ms\n"
    )
    csv.flush()

# ---------------------------
# LOAD PROMPT
# ---------------------------
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read()

# ---------------------------
# START SYSTEM LOGGER
# ---------------------------
sys_logger = start_system_logger()

# ---------------------------
# TRUE COLD START
# ---------------------------
stop_ollama()
start_ollama()
wait_for_ollama()

# ---------------------------
# REQUEST
# ---------------------------
payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": True
}

headers = {"Content-Type": "application/json"}

request_start = time.perf_counter()

ttft_user = None
first_token_time = None
token_times = []
final_stats = None

print("Sending request to Ollama (cold start)...")

with requests.post(
    OLLAMA_URL,
    headers=headers,
    data=json.dumps(payload),
    stream=True,
    timeout=900
) as r:

    r.raise_for_status()

    for line in r.iter_lines():
        if not line:
            continue

        now = time.perf_counter()
        data = json.loads(line.decode())

        if data.get("response"):
            if first_token_time is None:
                first_token_time = now
                ttft_user = now - request_start
                # print(f"TTFT (user): {ttft_user:.3f} s")
            token_times.append(now)

        if data.get("done", False):
            final_stats = data
            break

# ---------------------------
# SERVER-TRUTH METRICS
# ---------------------------
ttml = final_stats["load_duration"] / 1e9
prefill_time = final_stats["prompt_eval_duration"] / 1e9
decode_time = final_stats["eval_duration"] / 1e9

prompt_tokens = final_stats["prompt_eval_count"]
decoded_tokens = final_stats["eval_count"]

# ---------------------------
# TTFT CALCULATIONS (CORRECT)
# ---------------------------
ttft_compute = first_token_time - (request_start + ttml)

first_decode_latency = (
    decode_time / decoded_tokens if decoded_tokens > 0 else 0.0
)

ttft_after_load = prefill_time + first_decode_latency

# ---------------------------
# THROUGHPUT
# ---------------------------
throughput = decoded_tokens / decode_time if decode_time > 0 else 0.0
tpot = (decode_time / decoded_tokens) * 1000 if decoded_tokens > 0 else 0.0

# ---------------------------
# INTER-TOKEN LATENCY
# ---------------------------
itls = [
    (t2 - t1) * 1000
    for t1, t2 in zip(token_times[:-1], token_times[1:])
]

avg_itl = sum(itls) / len(itls) if itls else 0.0
min_itl = min(itls) if itls else 0.0
max_itl = max(itls) if itls else 0.0

# ---------------------------
# PRINT
# ---------------------------
print(f"TTML (cold): {ttml:.3f} s")
print(f"Prefill time: {prefill_time:.3f} s")
print(f"First decode latency: {first_decode_latency:.3f} s")
print(f"TTFT (after model load): {ttft_after_load:.3f} s")
print(f"Decoded tokens: {decoded_tokens}")
print(f"Throughput: {throughput:.2f} tok/s")

# ---------------------------
# CSV LOG
# ---------------------------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

csv.write(
    f"{timestamp},"
    f"{ttml:.6f},{ttft_user:.6f},{ttft_compute:.6f},{ttft_after_load:.6f},{first_decode_latency:.6f},"
    f"{prompt_tokens},{prefill_time:.6f},"
    f"{decoded_tokens},{decode_time:.6f},"
    f"{throughput:.3f},{tpot:.3f},"
    f"{avg_itl:.3f},{min_itl:.3f},{max_itl:.3f}\n"
)

csv.flush()
csv.close()

# ---------------------------
# CLEANUP
# ---------------------------
sys_logger.terminate()
sys_csv.flush()
sys_csv.close()

stop_ollama()
print("Benchmark complete. Ollama stopped.")
