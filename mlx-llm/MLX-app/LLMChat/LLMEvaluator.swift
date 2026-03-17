//
//  LLMEvaluator.swift
//  LLMChat
//
//  Created by MD Sahil AK on 10/02/26.
//

import SwiftUI
import Combine
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Tokenizers

/// Manages on-device LLM inference using the MLX framework.
///
/// `LLMEvaluator` handles model loading, single-prompt text generation, and
/// multi-run benchmarking. It streams generated tokens to the UI in real time
/// and logs performance metrics (TTFT, throughput, battery, thermal state)
/// after each generation run.
///
/// This class is `@Observable` and `@MainActor`-isolated so its published
/// properties (`running`, `output`) can drive SwiftUI views directly.
@Observable
@MainActor
final class LLMEvaluator {

    /// Whether a generation or benchmark is currently in progress.
    var running = false

    /// The latest generated text displayed in the UI. Updated incrementally as tokens stream in.
    var output = ""

    /// Whether the model has been loaded into memory and is ready for inference.
    var isLoaded: Bool { modelContainer != nil }

    /// The model configuration to load (Qwen 2.5 1.5B).
    private let modelConfig = LLMRegistry.qwen2_5_1_5b

    /// Sampling parameters used for generation.
    private let parameters = GenerateParameters(temperature: 0.7)

    /// The loaded model container, or `nil` if the model hasn't been loaded yet.
    private var modelContainer: ModelContainer? = nil

    /// Tracks the total number of generation runs completed (across single and benchmark calls).
    private var runCount: Int = 0
    
    /// Downloads (if needed) and loads the model into memory.
    ///
    /// Logs battery and thermal state before and after loading, as well as
    /// the total time to load. This method is a no-op if the model is already loaded.
    func loadModel() async throws {
        guard modelContainer == nil else { return }
        
        logBatteryStats("BEFORE MODEL LOADING")
        
        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                print("Downloading model: \(Int(progress.fractionCompleted * 100))%")
            }
        }
        
        print("TTLM - Time TO LOAD MODEL(s): \(elapsed)")
        
        logBatteryStats("AFTER MODEL LOADING")
    }
    
    /// Generates a single response for the given prompt and streams tokens to ``output``.
    ///
    /// Tokens are capped at 1,000 per run. After generation completes, TTFT and
    /// throughput metrics are printed to the console alongside battery stats.
    ///
    /// - Parameters:
    ///   - prompt: The user's input text.
    ///   - systemPrompt: The system message providing context to the model.
    func generate(prompt: String, systemPrompt: String = "You are a helpful assistant.") async {
        guard isLoaded else {
            print("Model not loaded. Try loading it first.")
            return
        }
        
        guard !running else { return }
        
        running = true
        output = "Generating..."
        
        do {
            runCount += 1
            
            let result = try await modelContainer!.perform { context in
                let input = try await context.processor.prepare(
                    input: .init(messages: [
                        ["role": "system", "content": systemPrompt],
                        ["role": "user", "content": prompt]
                    ])
                )
                
                return try MLXLMCommon.generate(
                    input: input,
                    parameters: parameters,
                    context: context
                ) { tokens in
                    let partial = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in self.output = partial }
                    return tokens.count >= 1000 ? .stop : .more
                }
            }
            
            logBatteryStats("AFTER GENERATION - \(runCount)")
            
            print("TTFT - TIME TO FIRST TOKEN(s): \(result.promptTime)")
            print("Ingestion / Throughput: ")
            print(result.summary())
            
            print("------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------")
        } catch {
            output = "Error: \(error.localizedDescription)"
        }
        
        running = false
    }
    
    /// Runs the generation loop for a fixed number of iterations to collect performance data.
    ///
    /// Each iteration generates a full response (up to 1,000 tokens) and logs
    /// TTFT, throughput, battery level, and thermal state. The UI shows the
    /// current run number and streamed output.
    ///
    /// - Parameters:
    ///   - prompt: The user's input text (same prompt is reused each iteration).
    ///   - systemPrompt: The system message providing context to the model.
    ///   - iterations: Number of generation runs to perform (default: 25).
    func benchmark(prompt: String, systemPrompt: String = "You are a helpful assistant.", iterations: Int = 25) async {
        guard isLoaded, !running else { return }
        running = true

        for i in 1...iterations {
            output = "Run \(i)/\(iterations)..."
            
            do {
                let result = try await modelContainer!.perform { context in
                    let input = try await context.processor.prepare(
                        input: .init(messages: [
                            ["role": "system", "content": systemPrompt],
                            ["role": "user", "content": prompt]
                        ])
                    )
                    
                    return try MLXLMCommon.generate(
                        input: input,
                        parameters: self.parameters,
                        context: context
                    ) { tokens in
                        let partial = context.tokenizer.decode(tokens: tokens)
                        Task { @MainActor in self.output = "Run \(i)/\(iterations):\n\(partial)" }
                        return tokens.count >= 1000 ? .stop : .more
                    }
                }
                
                runCount += 1
                
                print("Throughput Stats for Run \(i): \nTTFT=\(result.promptTime)s")
                print(result.summary())
                
                logBatteryStats("Battery Stats for Run \(i):")
                
                print("------------------------------------------------------------------------------")
            } catch {
                print("Run \(i) failed: \(error)")
            }
        }

        running = false
    }

    /// Prints battery level and thermal state to the console under the given heading.
    private func logBatteryStats(_ heading: String) {
        print("------------------------------------------------------------------------------")
        print(heading)
        
        print("Thermal State: \(DeviceHelpers.getBatteryThermalState())")
        print("Battery Level: \(DeviceHelpers.getBatteryLevel())")
        
        
        print("------------------------------------------------------------------------------")
    }
    
}

