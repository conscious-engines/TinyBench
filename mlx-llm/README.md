# MLX LLM

On-device LLM benchmarking on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-swift). A native SwiftUI iOS/macOS app that loads **Qwen 2.5 1.5B Instruct** locally using Apple's MLX framework, runs single-prompt generation and multi-iteration benchmark loops, and logs performance and device telemetry to the console.

---

## Repository Structure

```
mlx-llm/
└── MLX-app/               # Xcode project (LLMChat)
    ├── LLMChat/
    │   ├── LLMChatApp.swift       # App entry point
    │   ├── ContentView.swift      # Chat UI with benchmark controls
    │   ├── LLMEvaluator.swift     # Model loading, generation, and benchmarking logic
    │   └── DeviceHelpers.swift    # Cross-platform battery & thermal utilities
    ├── LLMChatTests/
    └── LLMChatUITests/
```

---

## Target Hardware

- **macOS:** Apple Silicon Mac (M1 or later)
- **iOS:** iPhone / iPad with Apple Neural Engine (A14+ / M1+)

The app uses the MLX backend which runs on Apple's GPU and Neural Engine — no external accelerator required.

---

## Model

| Property | Value |
|---|---|
| **Model** | Qwen 2.5 1.5B Instruct |
| **Framework** | MLX Swift (`MLXLLM`) |
| **Sampling** | Temperature 0.7 |
| **Max tokens** | 1,000 per generation |

The model is downloaded automatically from Hugging Face on first launch via `LLMModelFactory`.

---

## Metrics

Each generation run logs the following to the console:

| Metric | Description |
|---|---|
| **TTLM** | Time To Load Model — time to download (if needed) and load weights into memory |
| **TTFT** | Time To First Token — prefill latency |
| **Throughput** | Decode tokens per second (from `GenerateResult.summary()`) |
| **Battery Level** | Device battery percentage (0–100%) via IOKit (macOS) or UIKit (iOS) |
| **Thermal State** | Device thermal state: Normal, Warm, Hot, or Very Hot |

---

## Build Requirements

- **Xcode 15+** with Swift 5.9+
- **macOS 14+ / iOS 17+** deployment target
- Apple Silicon (MLX does not support Intel Macs)

### Dependencies (resolved via Swift Package Manager)

| Package | Version |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | 0.29.1 |
| [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (MLXLLM) | 2.29.1 |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.0.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.3.1 |

---

## Setup & Usage

1. **Open** `MLX-app/LLMChat.xcodeproj` in Xcode.
2. **Select** a target device (Mac, iPhone, or iPad).
3. **Build & Run** — the model downloads automatically on first launch (~3 GB).
4. Once the model is loaded, the chat interface appears with a pre-filled prompt.

### Single Generation

Type or edit a prompt and tap **Generate** (paper plane icon). Tokens stream to the screen in real time. Metrics are printed to the Xcode console after completion.

### Benchmark Loop

Tap **Run Benchmark Loop** to execute 25 consecutive generation runs with the same prompt. Each run logs TTFT, throughput, battery level, and thermal state to the console — useful for measuring sustained performance and thermal throttling over time.

---

## Prompt

The default benchmark prompt asks the model to write a multi-section essay on consciousness. It is designed to produce long outputs (~1,000 tokens) that exercise sustained decode throughput. The prompt can be edited in the text field before running.
