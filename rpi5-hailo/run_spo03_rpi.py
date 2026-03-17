import time
import json
import requests
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
HAILO_URL = "http://localhost:8000/api/chat"
MODEL = "qwen2.5-instruct:1.5b"
PROMPT_FILE = "prompt.txt"

NUM_RUNS = 20
RUN_INTERVAL_S = 1.0

LOG_DIR = Path("logs/spo03")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = LOG_DIR / "spo03_hailo_per_run_metrics.csv"

HAILO_SERVICE = "hailo-ollama"  # or hailo-genai, depending on your setup
STARTUP_TIMEOUT = 60

# ---------------------------
# UTILS
# ---------------------------

def read_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.split("=")[1].replace("'C", ""))
    except Exception:
        return None

# ---------------------------
# LOAD PROMPT
# ---------------------------
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read().strip()

# ---------------------------
# CSV INIT
# ---------------------------
new_file = not CSV_PATH.exists()
csv = open(CSV_PATH, "a")

if new_file:
    csv.write(
        "timestamp,run_id,"
        "ttft_s,"
        "total_latency_s,"
        "approx_tokens,"
        "throughput_tok_s,"
        "cpu_temp_c\n"
    )
    csv.flush()

print("Checking Hailo server...")
try:
    r = requests.get("http://localhost:8000/hailo/v1/list", timeout=2)
    r.raise_for_status()
except Exception as e:
    raise RuntimeError("Hailo server is not running. Start it manually first.") from e

print("Hailo server is reachable.\n")

# ---------------------------
# RUN LOOP
# ---------------------------
for i in range(NUM_RUNS):
    run_id = i + 1
    print(f"▶ Run {run_id}/{NUM_RUNS}")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    start_time = time.perf_counter()
    first_token_time = None

    r = requests.post(
        HAILO_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=900
    )

    end_time = time.perf_counter()

    data = r.json()

    text = data.get("message", {}).get("content", "")
    approx_tokens = max(len(text.split()), 1)

    # crude TTFT approximation (Hailo API is non-streaming)
    ttft = end_time - start_time
    total_latency = end_time - start_time
    throughput = approx_tokens / total_latency

    cpu_temp = read_cpu_temp()

    # ---------------------------
    # PRINT RESULTS
    # ---------------------------
    print(
        f"  TTFT: {ttft:.3f}s | "
        f"Total latency: {total_latency:.3f}s | "
        f"Approx tokens: {approx_tokens} | "
        f"Throughput: {throughput:.2f} tok/s | "
        f"CPU temp: {cpu_temp if cpu_temp else 'NA'}°C"
    )

    # print("\n--- Generated Output ---")
    # print(text)
    # print("--- End Output ---\n")

    # ---------------------------
    # CSV WRITE
    # ---------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv.write(
        f"{ts},{run_id},"
        f"{ttft:.6f},"
        f"{total_latency:.6f},"
        f"{approx_tokens},"
        f"{throughput:.3f},"
        f"{cpu_temp if cpu_temp else ''}\n"
    )
    csv.flush()

    if i < NUM_RUNS - 1:
        time.sleep(RUN_INTERVAL_S)

csv.close()
print("\nBenchmark complete.")
