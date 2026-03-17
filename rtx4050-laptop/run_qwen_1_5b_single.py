import torch
import time
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

metrics_csv = open("logs/sp02/sp02_user_metrics.csv", "a")

if metrics_csv.tell() == 0:
    metrics_csv.write(
        "timestamp,generated_tokens,decoded_tokens,decode_time_s,"
        "throughput_tok_s,avg_itl_ms,min_itl_ms,max_itl_ms,tpot_ms\n"
    )
    metrics_csv.flush()

gpu_log = open("logs/sp02/sp02_gpu_log.csv", "w")
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

model_id = "Qwen/Qwen2.5-1.5B-Instruct"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

torch.cuda.empty_cache()
torch.cuda.synchronize()

ttlm_start = time.perf_counter()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

torch.cuda.synchronize()

ttlm_end = time.perf_counter()

print(f"Model load time: {(ttlm_end - ttlm_start):.3f} seconds")

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

max_new_tokens = 15000

ttft_start = time.perf_counter()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated_tokens = []

with torch.no_grad():

    torch.cuda.synchronize()
    outputs = model(**inputs, use_cache=True)
    torch.cuda.synchronize()

    past_kv = outputs.past_key_values

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
    print(f"TTFT: {ttft:.3f} seconds")

    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    
    decoded_text = ""

    torch.cuda.synchronize()
    decode_start = time.perf_counter()

    decode_token_count = 0
    
    inter_token_latencies = []

    for _ in range(max_new_tokens - 1):
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

        generated_tokens.append(next_token.item())
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

        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    decode_end = time.perf_counter()

decode_time = decode_end - decode_start
throughput = decode_token_count / decode_time

if inter_token_latencies:
    avg_itl = sum(inter_token_latencies) / len(inter_token_latencies)
    min_itl = min(inter_token_latencies)
    max_itl = max(inter_token_latencies)

final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

tpot = 1 / throughput


print(f"Generated tokens: {len(generated_tokens)}")
print(f"Decoded tokens: {decode_token_count}")
print(f"Decode time: {decode_time:.3f} s")
print(f"Throughput: {throughput:.2f} tokens/sec")
print(f"Avg inter-token latency: {avg_itl*1000:.2f} ms")
print(f"Min inter-token latency: {min_itl*1000:.2f} ms")
print(f"Max inter-token latency: {max_itl*1000:.2f} ms")
print(f"Time per Output Token (TPOT): {tpot*1000:.2f} ms/token")

print("\n--- GENERATED TEXT ---")
print(final_text)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

metrics_csv.write(
    f"{timestamp},"
    f"{len(generated_tokens)},"
    f"{decode_token_count},"
    f"{decode_time:.6f},"
    f"{throughput:.3f},"
    f"{avg_itl*1000:.3f},"
    f"{min_itl*1000:.3f},"
    f"{max_itl*1000:.3f},"
    f"{tpot*1000:.3f}\n"
)
metrics_csv.flush()
metrics_csv.close()

nvidia_smi_proc.terminate()
gpu_log.close()


