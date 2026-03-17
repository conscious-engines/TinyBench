import time
import threading
import subprocess
from vllm import LLM, SamplingParams
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
PROMPT_FILE = "prompt.txt"

NUM_RUNS = 20
SLEEP_BETWEEN = 1.0
MAX_NEW_TOKENS = 4000

LOG_DIR = Path("logs/spo03_vllm")
LOG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV_PATH = LOG_DIR / "spo03_vllm_metrics.csv"
GPU_LOG_PATH     = LOG_DIR / "spo03_vllm_gpu_log.csv"

BATTERY_PATH  = Path("/sys/class/power_supply/BAT1/capacity")
CPU_TEMP_PATH = Path("/sys/class/thermal/thermal_zone9/temp")  # x86_pkg_temp

# =========================
# HELPERS
# =========================
def read_battery():
    try:
        return float(BATTERY_PATH.read_text().strip())
    except Exception:
        return -1.0

def read_cpu_temp():
    try:
        return float(CPU_TEMP_PATH.read_text().strip()) / 1000.0
    except Exception:
        return -1.0

def read_gpu_stats():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(",")
    return float(parts[0].strip()), float(parts[1].strip())  # power_W, temp_C

class HardwareMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self._running = False
        self._gpu_power = []
        self._gpu_temps = []
        self._cpu_temps = []

    def start(self):
        self._gpu_power = []
        self._gpu_temps = []
        self._cpu_temps = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            pwr, gt = read_gpu_stats()
            self._gpu_power.append(pwr)
            self._gpu_temps.append(gt)
            self._cpu_temps.append(read_cpu_temp())
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self._thread.join()
        return {
            "avg_power_w":  sum(self._gpu_power) / len(self._gpu_power) if self._gpu_power else 0,
            "peak_power_w": max(self._gpu_power) if self._gpu_power else 0,
            "max_gpu_temp": max(self._gpu_temps) if self._gpu_temps else 0,
            "max_cpu_temp": max(self._cpu_temps) if self._cpu_temps else 0,
        }

monitor = HardwareMonitor()

# =========================
# CSV INIT
# =========================
metrics_csv = open(METRICS_CSV_PATH, "w")
metrics_csv.write(
    "timestamp,run,"
    "total_tokens,decode_tokens,decode_ms,total_ms,decode_tok_s,"
    "battery_start_%,battery_end_%,"
    "max_battery_temp_c,max_cpu_temp_c,max_gpu_temp_c,"
    "initial_power_mw,final_power_mw,avg_power_mw,peak_power_mw,"
    "energy_per_token_mj\n"
)
metrics_csv.flush()

# =========================
# GPU LOG
# =========================
gpu_log = open(GPU_LOG_PATH, "w")
gpu_log.write("timestamp,util_gpu,mem_MiB,power_W,temp_C\n")
gpu_log.flush()

nvidia_smi_proc = subprocess.Popen(
    ["nvidia-smi",
     "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
     "--format=csv,noheader,nounits",
     "--loop-ms=100"],
    stdout=gpu_log,
    stderr=subprocess.DEVNULL
)

# =========================
# LOAD MODEL
# =========================
print("Loading model with vLLM...")
ttlm_start = time.perf_counter()
llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    max_model_len=MAX_NEW_TOKENS + 2048,
    gpu_memory_utilization=0.85,
    max_num_seqs=32,
    trust_remote_code=True,
)
ttlm_end = time.perf_counter()
print(f"Model load time (TTML): {ttlm_end - ttlm_start:.3f} s")

sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

tokenizer = llm.get_tokenizer()

# =========================
# LOAD PROMPT
# =========================
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read()

prefill_tokens = len(tokenizer.encode(prompt))

# =========================
# BENCHMARK LOOP
# =========================
for run_idx in range(NUM_RUNS):
    print(f"\n===== RUN {run_idx + 1}/{NUM_RUNS} =====")

    battery_start = read_battery()
    initial_power_w, _ = read_gpu_stats()

    monitor.start()
    t_start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    t_end = time.perf_counter()
    hw = monitor.stop()

    final_power_w, _ = read_gpu_stats()
    battery_end = read_battery()

    output = outputs[0]
    decode_tokens = len(output.outputs[0].token_ids)
    total_tokens = prefill_tokens + decode_tokens
    total_ms = (t_end - t_start) * 1000

    m = output.metrics
    if (m is not None
            and getattr(m, "first_token_time", None) is not None
            and getattr(m, "first_scheduled_time", None) is not None
            and getattr(m, "finished_time", None) is not None):
        ttft_ms   = (m.first_token_time - m.first_scheduled_time) * 1000
        decode_ms = (m.finished_time    - m.first_token_time)     * 1000
    else:
        # Fall back: estimate prefill as proportion of total based on token counts
        # prefill_ms ≈ total_ms * prefill_tokens / total_tokens
        ttft_ms   = total_ms * prefill_tokens / total_tokens
        decode_ms = total_ms - ttft_ms

    decode_tok_s = decode_tokens / (decode_ms / 1000) if decode_ms > 0 else 0

    avg_power_mw  = hw["avg_power_w"]  * 1000
    peak_power_mw = hw["peak_power_w"] * 1000
    initial_power_mw = initial_power_w * 1000
    final_power_mw   = final_power_w   * 1000

    energy_j = hw["avg_power_w"] * (total_ms / 1000)
    energy_per_token_mj = (energy_j * 1000) / decode_tokens if decode_tokens > 0 else 0

    print(f"Total tokens:    {total_tokens}  (prefill={prefill_tokens}, decode={decode_tokens})")
    print(f"Total time:      {total_ms:.1f} ms")
    print(f"Decode time:     {decode_ms:.1f} ms  ({decode_tok_s:.2f} tok/s)")
    print(f"Total:           {total_ms:.1f} ms")
    print(f"Battery:         {battery_start:.0f}% → {battery_end:.0f}%")
    print(f"CPU temp max:    {hw['max_cpu_temp']:.1f} °C")
    print(f"GPU temp max:    {hw['max_gpu_temp']:.1f} °C")
    print(f"Power:           init={initial_power_mw:.0f} mW  avg={avg_power_mw:.0f} mW  peak={peak_power_mw:.0f} mW  final={final_power_mw:.0f} mW")
    print(f"Energy/token:    {energy_per_token_mj:.4f} mJ")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_csv.write(
        f"{timestamp},{run_idx+1},"
        f"{total_tokens},{decode_tokens},{decode_ms:.3f},{total_ms:.3f},{decode_tok_s:.3f},"
        f"{battery_start:.1f},{battery_end:.1f},"
        f"-1,{hw['max_cpu_temp']:.1f},{hw['max_gpu_temp']:.1f},"
        f"{initial_power_mw:.1f},{final_power_mw:.1f},{avg_power_mw:.1f},{peak_power_mw:.1f},"
        f"{energy_per_token_mj:.4f}\n"
    )
    metrics_csv.flush()

    time.sleep(SLEEP_BETWEEN)

# =========================
# CLEANUP
# =========================
metrics_csv.close()
nvidia_smi_proc.terminate()
gpu_log.close()
