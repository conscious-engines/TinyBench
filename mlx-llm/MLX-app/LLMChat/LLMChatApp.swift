//
//  LLMChatApp.swift
//  LLMChat
//
//  Created by MD Sahil AK on 09/02/26.
//

import SwiftUI

/// The main entry point for the LLMChat application.
///
/// LLMChat is an on-device language model chat app powered by Apple's MLX framework.
/// It loads and runs a Qwen 2.5 1.5B model locally, with support for single-prompt
/// generation and multi-run benchmarking with performance metrics.
@main
struct LLMChatApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}  
