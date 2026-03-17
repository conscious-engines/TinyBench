//
//  ContentView.swift
//  LLMChat
//
//  Created by MD Sahil AK on 09/02/26.
//

import SwiftUI
import Combine

/// The main chat interface for interacting with the on-device LLM.
///
/// Displays the model's streamed output in a scrollable area and provides
/// controls for sending a prompt, clearing output, and running a benchmark loop.
/// While the model is loading, a progress indicator is shown instead.
struct ContentView: View {
    /// The shared evaluator that manages model loading and text generation.
    let evaluator = LLMEvaluator()

    /// The editable prompt text. Pre-filled with a long-form essay prompt for benchmarking.
    @State private var userPrompt: String = """
        Write an in-depth, structured, and self-contained essay explaining the concept of consciousness from multiple perspectives.
    
            Begin by clearly defining consciousness and why it is a difficult concept to study. Then explore the topic from the following viewpoints, dedicating multiple detailed paragraphs to each:
            1. Philosophical perspectives, including classical and modern views, major debates, and unresolved questions.
            2. Neuroscientific perspectives, covering brain structures, neural correlates, current theories, and experimental approaches.
            3. Cognitive science and psychology perspectives, including perception, attention, self-awareness, and consciousness disorders.
            4. Artificial intelligence and machine consciousness, discussing whether machines can be conscious, functional vs phenomenal consciousness, and current limitations of AI systems.
            5. Ethical and societal implications of understanding or engineering consciousness.
    
            Throughout the essay:
            - Use clear section headings
            - Explain all technical terms in plain language
            - Provide illustrative examples where helpful
            - Compare and contrast different viewpoints
            - Avoid bullet lists unless they are necessary for clarity
    
        Conclude with a reflective summary that discusses what remains unknown about consciousness and what future research directions may look like.
    
        The response should be detailed, continuous, and written in a neutral, academic tone.
    """
    
    var body: some View {
        VStack {
            if evaluator.isLoaded {
                ScrollView(.vertical) {
                    Text(LocalizedStringKey(evaluator.output))
                        .multilineTextAlignment(.leading)
                        .animation(.easeInOut, value: evaluator.output)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                
                VStack {
                    HStack {
                        Button {
                            evaluator.output = ""
                        } label: {
                            Label("Clear", systemImage: "xmark.circle")
                                .labelStyle(.iconOnly)
                        }
                        .disabled(evaluator.running)
                        .tint(.red)
                        
                        TextField("Ask something...", text: $userPrompt)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .onSubmit { Task { await evaluator.generate(prompt: userPrompt) } }
                            .disabled(evaluator.running)
                            .padding()
                        
                        Button {
                            Task {
                                await evaluator.generate(prompt: userPrompt)
                            }
                        } label: {
                            Label("Generate", systemImage: "paperplane.fill")
                                .labelStyle(.iconOnly)
                        }
                        .disabled(evaluator.running)
                        .tint(.blue)
                    }
                    
                    Button {
                        Task {
                            await evaluator.benchmark(prompt: userPrompt)
                        }
                    } label: {
                        Label("Run Benchmark Loop", systemImage: "testtube.2")
                    }
                    .disabled(evaluator.running)
                    .tint(.orange)
                }
                .padding()
            } else {
                loadingView
            }
        }
        .task {
            do {
                try await evaluator.loadModel()
            } catch {
                print("Failed to load model: \(error)")
            }
        }
    }
    
    /// A centered loading indicator shown while the model is being downloaded or loaded into memory.
    private var loadingView: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                ProgressView()
                Text("Loading Model...")
                Spacer()
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

