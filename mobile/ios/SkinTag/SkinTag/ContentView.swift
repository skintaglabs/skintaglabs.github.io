//
//  ContentView.swift
//  SkinTag
//
//  Main content view handling navigation
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        if !appState.hasAcceptedDisclaimer {
            DisclaimerView()
        } else if let error = appState.modelLoadError {
            ModelErrorView(error: error)
        } else if !appState.modelLoaded {
            LoadingView()
        } else {
            MainCameraView()
        }
    }
}

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading model...")
                .font(.headline)
                .foregroundColor(.secondary)
        }
    }
}

struct ModelErrorView: View {
    let error: String

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)

            Text("Model Load Error")
                .font(.title)
                .fontWeight(.bold)

            Text(error)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Text("Please reinstall the app or contact support.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
}
