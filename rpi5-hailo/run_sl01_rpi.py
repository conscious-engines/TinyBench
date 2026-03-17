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
MODEL = "qwen2:1.5b"
PROMPT_FILE = "prompt.txt"

NUM_RUNS = 200
RUN_INTERVAL_S = 1.0

LOG_DIR = Path("logs/sl01")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = LOG_DIR / "sl01_hailo_per_run_metrics.csv"

HAILO_SERVICE = "hailo-ollama"
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

def read_power_metrics():
    """Collect comprehensive power and performance indicators"""
    metrics = {}
    
    try:
        # Throttling status
        throttled = subprocess.check_output(
            ["vcgencmd", "get_throttled"]
        ).decode().strip().split("=")[1]
        metrics['throttled'] = throttled
        
        # Input voltage (5V supply)
        try:
            ext5v = subprocess.check_output(
                ["vcgencmd", "pmic_read_adc", "EXT5V_V"]
            ).decode().strip()
            # Extract voltage from "EXT5V_V volt(24)=5.07726000V"
            voltage = float(ext5v.split("=")[1].replace("V", ""))
            metrics['input_voltage_v'] = voltage
        except:
            metrics['input_voltage_v'] = None
        
        # CPU frequency (MHz)
        freq = subprocess.check_output(
            ["vcgencmd", "measure_clock", "arm"]
        ).decode().strip().split("=")[1]
        metrics['cpu_freq_mhz'] = int(freq) / 1_000_000
        
        # CPU temperature
        temp = subprocess.check_output(
            ["vcgencmd", "measure_temp"]
        ).decode().strip().split("=")[1].replace("'C", "")
        metrics['cpu_temp_c'] = float(temp)
        
        # Core voltage
        volts = subprocess.check_output(
            ["vcgencmd", "measure_volts", "core"]
        ).decode().strip().split("=")[1].replace("V", "")
        metrics['core_voltage_v'] = float(volts)
        
    except Exception as e:
        print(f"Error reading power metrics: {e}")
    
    return metrics

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
        "decode_time_s,"
        "total_latency_s,"
        "token_count,"
        "decode_throughput_tok_s,"
        "cpu_temp_c,"
        "cpu_freq_mhz,"
        "input_voltage_v,"
        "core_voltage_v,"
        "throttled\n"
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
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = ""

    with requests.post(HAILO_URL, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            
            # Parse the streaming JSON response
            try:
                # Remove "data: " prefix if present
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                
                # Skip if it's [DONE]
                if line_str.strip() == "[DONE]":
                    continue
                    
                chunk = json.loads(line_str)
                
                # First token timing
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                # Extract content from the chunk
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_text += content
                    # Count tokens (rough approximation)
                    token_count += len(content.split())
                    
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                # Skip malformed lines
                continue
        
        end_time = time.perf_counter()

    # Calculate metrics
    ttft = first_token_time - start_time if first_token_time else 0
    decode_time = end_time - first_token_time if first_token_time else 0
    total_latency = end_time - start_time
    decode_throughput = token_count / decode_time if decode_time > 0 else 0

    # Get power metrics
    power_info = read_power_metrics()

    # ---------------------------
    # PRINT RESULTS
    # ---------------------------
    print(
        f"  TTFT: {ttft:.3f}s | "
        f"Decode Time: {decode_time:.3f}s | "
        f"Total Latency: {total_latency:.3f}s | "
        f"Tokens: {token_count} | "
        f"Decode Throughput: {decode_throughput:.2f} tok/s"
    )
    print(
        f"  CPU: {power_info.get('cpu_freq_mhz', 'NA'):.0f} MHz | "
        f"Temp: {power_info.get('cpu_temp_c', 'NA')}°C | "
        f"Input V: {power_info.get('input_voltage_v', 'NA'):.2f}V | "
        f"Throttled: {power_info.get('throttled', 'NA')}"
    )

    # print("\n--- Generated Output ---")
    # print(full_text)
    # print("--- End Output ---\n")

    # ---------------------------
    # CSV WRITE
    # ---------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv.write(
        f"{ts},{run_id},"
        f"{ttft:.6f},"
        f"{decode_time:.6f},"
        f"{total_latency:.6f},"
        f"{token_count},"
        f"{decode_throughput:.3f},"
        f"{power_info.get('cpu_temp_c', '')},"
        f"{power_info.get('cpu_freq_mhz', '')},"
        f"{power_info.get('input_voltage_v', '')},"
        f"{power_info.get('core_voltage_v', '')},"
        f"{power_info.get('throttled', '')}\n"
    )
    csv.flush()

    if i < NUM_RUNS - 1:
        time.sleep(RUN_INTERVAL_S)

csv.close()
print("\nBenchmark complete.")