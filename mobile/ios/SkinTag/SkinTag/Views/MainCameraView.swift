//
//  MainCameraView.swift
//  SkinTag
//
//  Main camera view for capturing skin lesion images
//

import SwiftUI
import AVFoundation

struct MainCameraView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var cameraManager = CameraManager()
    @State private var showingResults = false
    @State private var currentResult: TriageResult?
    @State private var capturedImage: UIImage?
    @State private var isProcessing = false

    var body: some View {
        ZStack {
            // Camera preview
            CameraPreviewView(cameraManager: cameraManager)
                .ignoresSafeArea()

            // Overlay UI
            VStack {
                // Top bar
                TopBarView()

                Spacer()

                // Capture guide
                CaptureGuideView()

                Spacer()

                // Bottom controls
                BottomControlsView(
                    isProcessing: isProcessing,
                    onCapture: capturePhoto
                )
            }

            // Processing overlay
            if isProcessing {
                ProcessingOverlay()
            }
        }
        .onAppear {
            cameraManager.startSession()
        }
        .onDisappear {
            cameraManager.stopSession()
        }
        .sheet(isPresented: $showingResults) {
            if let result = currentResult, let image = capturedImage {
                ResultsView(result: result, image: image, onDismiss: {
                    showingResults = false
                    currentResult = nil
                    capturedImage = nil
                })
            }
        }
    }

    private func capturePhoto() {
        guard !isProcessing else { return }
        isProcessing = true

        cameraManager.capturePhoto { image in
            guard let image = image else {
                isProcessing = false
                return
            }

            self.capturedImage = image

            // Run inference
            if let modelInference = appState.getModelInference() {
                DispatchQueue.global(qos: .userInitiated).async {
                    let result = modelInference.predict(image: image)

                    DispatchQueue.main.async {
                        self.currentResult = result
                        self.isProcessing = false
                        self.showingResults = true
                    }
                }
            } else {
                isProcessing = false
            }
        }
    }
}

struct TopBarView: View {
    var body: some View {
        HStack {
            Text("SkinTag")
                .font(.headline)
                .foregroundColor(.white)

            Spacer()

            NavigationLink(destination: InfoView()) {
                Image(systemName: "info.circle")
                    .font(.title2)
                    .foregroundColor(.white)
            }
        }
        .padding()
        .background(Color.black.opacity(0.5))
    }
}

struct CaptureGuideView: View {
    var body: some View {
        VStack(spacing: 10) {
            // Circular guide overlay
            Circle()
                .stroke(Color.white, lineWidth: 3)
                .frame(width: 200, height: 200)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.3), lineWidth: 1)
                        .frame(width: 180, height: 180)
                )

            Text("Center the lesion in the circle")
                .font(.subheadline)
                .foregroundColor(.white)
                .padding(.horizontal, 20)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.6))
                .cornerRadius(20)
        }
    }
}

struct BottomControlsView: View {
    let isProcessing: Bool
    let onCapture: () -> Void

    var body: some View {
        VStack(spacing: 15) {
            // Tips
            HStack(spacing: 20) {
                TipBadge(icon: "sun.max", text: "Good lighting")
                TipBadge(icon: "hand.raised", text: "Hold steady")
                TipBadge(icon: "arrow.up.left.and.arrow.down.right", text: "Fill frame")
            }
            .padding(.horizontal)

            // Capture button
            Button(action: onCapture) {
                ZStack {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 70, height: 70)

                    Circle()
                        .stroke(Color.white, lineWidth: 4)
                        .frame(width: 80, height: 80)
                }
            }
            .disabled(isProcessing)
            .opacity(isProcessing ? 0.5 : 1)

            // Disclaimer reminder
            Text("For educational purposes only - not a medical diagnosis")
                .font(.caption2)
                .foregroundColor(.white.opacity(0.8))
                .padding(.bottom, 5)
        }
        .padding()
        .background(Color.black.opacity(0.5))
    }
}

struct TipBadge: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
            Text(text)
                .font(.caption)
        }
        .foregroundColor(.white.opacity(0.9))
    }
}

struct ProcessingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)

                Text("Analyzing image...")
                    .font(.headline)
                    .foregroundColor(.white)
            }
        }
    }
}

#Preview {
    MainCameraView()
        .environmentObject(AppState())
}
