# MLC-LLM Android Benchmarking — Samsung Galaxy S24 Ultra

A fork of [MLC-LLM](https://github.com/mlc-ai/mlc-llm) extended with an automated benchmarking service for measuring on-device LLM inference performance. Targets the **Samsung Galaxy S24 Ultra** (Snapdragon 8 Gen 3, Adreno 750 GPU, OpenCL) and exports per-iteration telemetry — tokens/sec, prefill latency, battery drain, CPU/GPU temperatures, and power consumption — to a CSV file on device.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone & Submodules](#step-1--clone--submodules)
4. [Step 2 — Python Environment & MLC-LLM](#step-2--python-environment--mlc-llm)
5. [Step 3 — Compile the Model](#step-3--compile-the-model)
6. [Step 4 — Prepare Model Assets](#step-4--prepare-model-assets)
7. [Step 5 — Build Native JNI Libraries](#step-5--build-native-jni-libraries)
8. [Step 6 — Build the APK](#step-6--build-the-apk)
9. [Step 7 — Install & Configure on S24 Ultra](#step-7--install--configure-on-s24-ultra)
10. [Step 8 — Run Benchmarks & Collect Results](#step-8--run-benchmarks--collect-results)
11. [CSV Output Schema](#csv-output-schema)
12. [Changing the Prompt or Model](#changing-the-prompt-or-model)
13. [Troubleshooting](#troubleshooting)

---

## How It Works

```
App Launch
   │
   ▼
BenchmarkService (foreground service, wake lock)
   │
   ├─ Copies bundled model from assets → internal storage (first run only)
   ├─ Loads model via MLCEngine (OpenCL, arm64-v8a)
   │
   ▼
MainActivity polls for model-ready (up to 30 s)
   │
   ├─ Reads prompt from /sdcard/Android/data/ai.mlc.mlcchat/files/prompt.txt
   │  (fallback: /sdcard/Download/prompt.txt, /sdcard/prompt.txt, or default)
   │
   ▼
runBenchmark(prompt)
   ├─ 1 warmup iteration  (discarded)
   └─ 20 timed iterations (1 s delay between each)
       ├─ per-iteration: inference + telemetry sampled every 200 ms
       └─ appends row to benchmark_log.csv
```

**Telemetry collected per iteration:**
- Token throughput (prefill time, decode tokens/sec)
- Battery level, temperature, health
- CPU temperature (`/sys/class/thermal/thermal_zone0/temp`)
- GPU temperature, current & max frequency (`/sys/class/kgsl/kgsl-3d0/…`) — Adreno-specific
- Instantaneous, average, and peak power draw (mW) and energy per token (mJ/tok)

The app uses `FLAG_KEEP_SCREEN_ON`, `setShowWhenLocked`, `setTurnScreenOn`, and `requestDismissKeyguard` to remain in the foreground (oom_score_adj −800) throughout the benchmark, preventing Android's low-memory killer from pausing or killing the process mid-run.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux or macOS | Windows works with adjustments (Ninja required) |
| Python | ≥ 3.9 | Conda strongly recommended |
| Java JDK | 17 | Android Gradle requires JDK 17 |
| Android Studio | Hedgehog (2023.1+) | or any 2023+ release |
| Android SDK | API 33–35 | install via Android Studio SDK Manager |
| Android NDK | r26+ (r27 recommended) | install via Android Studio SDK Manager |
| CMake | ≥ 3.18 | install via Android Studio SDK Manager |
| Rust | stable | with `aarch64-linux-android` target |
| ADB | any | included with Android SDK platform-tools |

**Device:** Samsung Galaxy S24 Ultra (SM-S928B or SM-S928U) with Developer Options and USB Debugging enabled. Charge to at least 30% before starting a benchmark run.

---

## Step 1 — Clone & Submodules

```bash
git clone https://github.com/<your-org>/<this-repo>.git
cd <this-repo>
git submodule update --init --recursive
```

This pulls in the `mlc-llm` submodule (upstream MLC-LLM engine) along with its own nested submodules, including `3rdparty/tvm`. The TVM submodule is required for the native library build.

Expected directory structure after clone:

```
<repo>/
├── mlc-llm/          ← git submodule (upstream MLC-LLM)
│   ├── 3rdparty/tvm  ← nested submodule inside mlc-llm
│   ├── python/
│   ├── cpp/
│   └── ...
├── android/
│   └── MLCChat/      ← Android benchmarking app
├── README.md
└── .gitignore
```

---

## Step 2 — Python Environment & MLC-LLM

### 2a — Create conda environment

```bash
conda create -n mlc-llm python=3.11 -y
conda activate mlc-llm
```

### 2b — Install MLC-LLM Python package

Install the pre-built nightly wheel (no GPU required on the host machine):

```bash
pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly-cpu
```

Verify:

```bash
mlc_llm --version
```

### 2c — Install Rust with Android cross-compile target

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup target add aarch64-linux-android
```

---

## Step 3 — Compile the Model

This produces `libmodel_android.a` — the model's computation graph compiled for `arm64-v8a + OpenCL`. This step downloads the model from Hugging Face and runs ML compilation, which takes **20–60 minutes** depending on your CPU.

### 3a — Set environment variables

```bash
# Find your NDK path in Android Studio → SDK Manager → SDK Tools → NDK → Show Package Details
export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.2.12479018    # Linux (adjust version)
# export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/27.2.12479018   # macOS

export MLC_LLM_SOURCE_DIR=$(pwd)/mlc-llm   # points to the submodule
```

Verify the NDK path is correct:
```bash
ls $ANDROID_NDK/build/cmake/android.toolchain.cmake
```

### 3b — Create output directories

```bash
mkdir -p dist/android
mkdir -p android/MLCChat/mlc4j/build/lib
mkdir -p android/MLCChat/app/src/main/assets/lib
```

### 3c — Compile Qwen2.5-1.5B for Android/OpenCL

```bash
mlc_llm compile "HF://Qwen/Qwen2.5-1.5B-Instruct" \
  --quantization q4f16_2 \
  --device android:opencl \
  --opt O3 \
  --output dist/android/libmodel_android.tar
```

On first run this downloads ~3 GB from Hugging Face. The output `.tar` contains the compiled static library for arm64-v8a.

### 3d — Extract and place the compiled library

```bash
cd dist/android
tar -xf libmodel_android.tar

# Place for native library build (Step 5)
cp libmodel_android.a ../../android/MLCChat/mlc4j/build/lib/

# Place inside assets (bundled with APK, loaded at runtime)
cp libmodel_android.a ../../android/MLCChat/app/src/main/assets/lib/

cd ../..
```

---

## Step 4 — Prepare Model Assets

These are the model weights and tokenizer files that get bundled into the APK and copied to internal storage on first app launch (~1–2 minutes on device).

### 4a — Create asset directories

```bash
mkdir -p android/MLCChat/app/src/main/assets/params
```

### 4b — Convert model weights to MLC format

```bash
mlc_llm convert_weight "HF://Qwen/Qwen2.5-1.5B-Instruct" \
  --quantization q4f16_2 \
  --output dist/android/qwen2.5-1.5b-weights
```

### 4c — Generate model configuration

```bash
mlc_llm gen_config "HF://Qwen/Qwen2.5-1.5B-Instruct" \
  --quantization q4f16_2 \
  --conv-template qwen2 \
  --output dist/android/qwen2.5-1.5b-weights
```

### 4d — Copy all assets into the app

```bash
WEIGHTS_DIR=dist/android/qwen2.5-1.5b-weights
ASSETS_DIR=android/MLCChat/app/src/main/assets

cp $WEIGHTS_DIR/mlc-chat-config.json   $ASSETS_DIR/
cp $WEIGHTS_DIR/tokenizer.json         $ASSETS_DIR/
cp $WEIGHTS_DIR/vocab.json             $ASSETS_DIR/
cp $WEIGHTS_DIR/merges.txt             $ASSETS_DIR/
cp $WEIGHTS_DIR/tokenizer_config.json  $ASSETS_DIR/
cp $WEIGHTS_DIR/params/*.bin           $ASSETS_DIR/params/

# Copy tensor cache index if generated
[ -f $WEIGHTS_DIR/tensor-cache.json ] && cp $WEIGHTS_DIR/tensor-cache.json $ASSETS_DIR/
```

> **Disk space:** Total asset size is ~1.2 GB. Ensure you have 10+ GB free for the full build.

---

## Step 5 — Build Native JNI Libraries

Compiles `tvm4j_runtime_packed.so` with the model statically linked in, and the TVM Java bindings JAR. The Gradle build picks these up automatically.

### 5a — Run the build script

```bash
cd android/MLCChat/mlc4j
python prepare_libs.py
cd ../../..
```

The script defaults to `../../../mlc-llm` (the submodule) as the MLC-LLM source directory. You can override with `--mlc-llm-source-dir <path>` if needed. This runs CMake for `arm64-v8a` and installs outputs to `mlc4j/build/output/`. Allow **10–30 minutes**.

### 5b — Copy outputs to the expected locations

```bash
BUILD_OUT=android/MLCChat/mlc4j/build/output

# JNI shared libraries (picked up by mlc4j/build.gradle)
mkdir -p android/MLCChat/mlc4j/output/jniLibs/arm64-v8a
cp $BUILD_OUT/arm64-v8a/*.so android/MLCChat/mlc4j/output/jniLibs/arm64-v8a/

# TVM Java core JAR (referenced by mlc4j/build.gradle dependencies)
mkdir -p android/MLCChat/mlc4j/libs
cp $BUILD_OUT/tvm4j_core.jar android/MLCChat/mlc4j/libs/tvm4j-core-0.0.1-SNAPSHOT.jar
```

---

## Step 6 — Build the APK

### Option A — Android Studio (recommended)

1. Open Android Studio → **File → Open** → select `android/MLCChat/`
2. Wait for Gradle sync to complete (downloads ~1 GB of dependencies on first run)
3. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
4. APK output: `app/build/outputs/apk/debug/app-debug.apk`

### Option B — Command line

```bash
cd android/MLCChat

# Create local SDK config (do not commit this file)
echo "sdk.dir=$HOME/Android/Sdk" > local.properties           # Linux
# echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties   # macOS

export GRADLE_OPTS="-Xmx8g"
./gradlew assembleDebug
cd ../..
```

APK output: `android/MLCChat/app/build/outputs/apk/debug/app-debug.apk`

> **Build time:** 5–15 minutes. The APK embeds 1.2 GB of model assets and will be >1 GB in size.

---

## Step 7 — Install & Configure on S24 Ultra

### 7a — Enable Developer Options on the device

1. **Settings → About phone → Software information**
2. Tap **Build number** 7 times until "Developer mode has been turned on" appears
3. **Settings → Developer options** → enable **USB debugging**

### 7b — Connect and verify ADB

```bash
adb devices
# Expected: <serial>    device
```

Accept the RSA fingerprint prompt on the phone if it shows `unauthorized`.

### 7c — Install the APK

```bash
adb install -r android/MLCChat/app/build/outputs/apk/debug/app-debug.apk
```

### 7d — Grant permissions

```bash
adb shell pm grant ai.mlc.mlcchat android.permission.POST_NOTIFICATIONS
```

On Android 13+ (API 33) the app uses scoped storage and writes to its own external files directory (`/sdcard/Android/data/ai.mlc.mlcchat/files/`) without requiring additional storage permissions.

### 7e — Push a prompt file

```bash
adb shell mkdir -p /sdcard/Android/data/ai.mlc.mlcchat/files/

echo "Explain how transformers work in machine learning." | \
  adb shell dd of=/sdcard/Android/data/ai.mlc.mlcchat/files/prompt.txt
```

If no prompt file is found, the app uses the default: *"Explain how transformers work in machine learning."*

---

## Step 8 — Run Benchmarks & Collect Results

### 8a — Launch the benchmark

```bash
adb shell am start -n ai.mlc.mlcchat/.MainActivity
```

The app will:
1. Start the foreground `BenchmarkService` and acquire a 60-minute wake lock
2. Copy the bundled model from assets to internal storage (first run only, ~1–2 min)
3. Load the model via MLCEngine (OpenCL backend)
4. Run 1 warmup iteration, then 20 timed iterations with 1 s between each
5. Write results to `/sdcard/Android/data/ai.mlc.mlcchat/files/benchmark_log.csv`

### 8b — Monitor progress via logcat

```bash
adb logcat -s BENCH:I MainActivity:I
```

You will see output like:
```
INITIALIZING MODEL
Model loaded in 18432ms
BENCHMARK START  prompt="Explain how transformers..."
--- WARMUP 1/1 ---
--- WARMUP COMPLETE, starting timed iterations ---
--- ITERATION 1/20 ---
  tokens=312  prefill=2341ms  decode=28190ms  total=30531ms  tput=10.955 tok/s
  batt: 82%→82%  temp: 35.2→36.1°C (max 36.4°C)
  cpu: 48.2°C (max 52.1°C)  gpu: 61.3°C (max 63.2°C)  freq: 900/900 MHz
  power: avg=4812.3mW  peak=5124.0mW  energy=14.432mJ/tok
...
BENCHMARK COMPLETE  results -> /data/user/0/ai.mlc.mlcchat/...
```

### 8c — Expected timeline

| Phase | Duration |
|---|---|
| Model copy to internal storage (first run) | ~1–2 min |
| Model load via MLCEngine / OpenCL | ~15–30 s |
| Warmup iteration | ~30–60 s |
| 20 benchmark iterations | ~10–20 min total |

### 8d — Pull results

```bash
mkdir -p results
adb pull /sdcard/Android/data/ai.mlc.mlcchat/files/benchmark_log.csv ./results/
```

### 8e — Optional: ADB broadcast trigger

Once the app is running with the model loaded, you can start a new benchmark with a different prompt without relaunching:

```bash
adb shell am broadcast -a ai.mlc.BENCH \
  --es prompt "Summarize the key milestones in the history of artificial intelligence."
```

---

## CSV Output Schema

Results are appended to `benchmark_log.csv` (one row per iteration, header written fresh each run):

| Column | Description |
|---|---|
| `iteration` | Iteration number (1–20) |
| `timestamp` | Unix timestamp (ms) |
| `date_time` | Human-readable datetime |
| `prompt` | First 100 chars of the prompt |
| `total_tokens` | Total tokens generated |
| `prefill_ms` | Time to first token (ms) |
| `decode_tokens` | Tokens generated after first token |
| `decode_ms` | Time spent decoding (ms) |
| `total_ms` | Total inference wall time (ms) |
| `decode_tok_s` | Decode throughput (tokens/sec) |
| `battery_start_%` | Battery level at start of iteration |
| `battery_end_%` | Battery level at end of iteration |
| `battery_delta_%` | Change in battery level |
| `temp_start_c` | Battery temperature at start (°C) |
| `temp_end_c` | Battery temperature at end (°C) |
| `temp_delta_c` | Battery temperature change (°C) |
| `battery_health` | Battery health string (Good / Overheat / …) |
| `cpu_temp_c` | CPU temperature at end (°C) |
| `gpu_temp_c` | GPU temperature at end (°C) — Adreno only |
| `gpu_freq_mhz` | GPU clock at end (MHz) — Adreno only |
| `gpu_max_freq_mhz` | GPU max clock (MHz) — Adreno only |
| `max_battery_temp_c` | Peak battery temp during iteration (°C) |
| `max_cpu_temp_c` | Peak CPU temp during iteration (°C) |
| `max_gpu_temp_c` | Peak GPU temp during iteration (°C) |
| `initial_power_mw` | Power draw at iteration start (mW) |
| `final_power_mw` | Power draw at iteration end (mW) |
| `avg_power_mw` | Average power during iteration (mW) |
| `peak_power_mw` | Peak power during iteration (mW) |
| `energy_per_token_mj` | Energy per decoded token (mJ/tok) |

> GPU columns read Qualcomm Adreno-specific sysfs paths (`/sys/class/kgsl/kgsl-3d0/…`) and will show 0 on non-Qualcomm devices. Power values are derived from `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW` (µA on Snapdragon) × voltage (mV).

---

## Changing the Prompt or Model

### Change the prompt

Push a new file before launching (or between runs with the broadcast trigger):

```bash
echo "Your prompt here." | \
  adb shell dd of=/sdcard/Android/data/ai.mlc.mlcchat/files/prompt.txt
```

### Use a different model

1. **Compile** the new model (Step 3) with its Hugging Face path and desired quantization
2. **Replace assets** (Step 4) with the new model's weights, config, and tokenizer files
3. **Update `BenchmarkService.kt`** — change the `modelDir` directory name:
   ```kotlin
   val modelDir = File(filesDir, "your-model-dir-name")
   ```
4. **Rebuild native libs** (Step 5) with the new `libmodel_android.a`
5. **Rebuild and reinstall** the APK (Steps 6–7)

Other tested model configurations (`mlc-package-config.json`):

| Model | Quantization | Approx. size |
|---|---|---|
| `HF://mlc-ai/Qwen3-0.6B-q0f16-MLC` | q0f16 | ~1.2 GB |
| `HF://mlc-ai/Qwen3-1.7B-q4f16_1-MLC` | q4f16_1 | ~1.1 GB |
| `HF://mlc-ai/gemma-2-2b-it-q4f16_1-MLC` | q4f16_1 | ~1.3 GB |
| `HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC` | q4f16_0 | ~2 GB |

---

## Troubleshooting

**Model load timeout after 30 s**
The app polls every 500 ms for up to 30 s. If the model copy from assets is still in progress, the app exits. Re-launch it — the copy step is skipped on subsequent launches. Check logcat for details:
```bash
adb logcat -s BENCH:E
```

**`libmodel_android.a` not found during CMake build**
Ensure `android/MLCChat/mlc4j/build/lib/libmodel_android.a` exists before running `prepare_libs.py`. The CMake config (`mlc4j/CMakeLists.txt`) imports it from exactly that path.

**`ANDROID_NDK` not set or wrong path**
The prepare script checks for `ANDROID_NDK` at startup. Verify:
```bash
echo $ANDROID_NDK
ls $ANDROID_NDK/build/cmake/android.toolchain.cmake
```

**CSV is empty (header row only)**
The benchmark runs asynchronously. Wait for the `BENCHMARK COMPLETE` logcat line before pulling:
```bash
adb logcat -s BENCH:I | grep "BENCHMARK COMPLETE"
```

**App killed mid-benchmark by Samsung power manager**
Disable aggressive battery optimization:
1. **Settings → Battery → Background usage limits → Never sleeping apps** → add MLCChat
2. **Settings → Apps → MLCChat → Battery** → set to **Unrestricted**

**`mlc_llm compile` hangs or errors**
- Use the nightly CPU wheel, not the stable pip release
- Try `--opt O0` first to verify the pipeline, then switch to `O3`
- Ensure `ANDROID_NDK` is exported before running the compile command
- Ensure `aarch64-linux-android` Rust target is installed (`rustup target list --installed`)

---

## Acknowledgements

Built on [MLC-LLM](https://github.com/mlc-ai/mlc-llm) by the MLC AI team. See the upstream project for documentation on iOS, WebGPU, CUDA, and other deployment targets, as well as the full citation list.
