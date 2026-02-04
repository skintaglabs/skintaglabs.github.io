//
//  SkinTagApp.swift
//  SkinTag
//
//  Skin lesion triage application using Core ML
//

import SwiftUI

@main
struct SkinTagApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

/// Global application state
class AppState: ObservableObject {
    @Published var hasAcceptedDisclaimer = false
    @Published var modelLoaded = false
    @Published var modelLoadError: String?

    private var modelInference: ModelInference?

    init() {
        loadModel()
    }

    func loadModel() {
        do {
            modelInference = try ModelInference()
            modelLoaded = true
        } catch {
            modelLoadError = error.localizedDescription
            print("Failed to load model: \(error)")
        }
    }

    func getModelInference() -> ModelInference? {
        return modelInference
    }
}
