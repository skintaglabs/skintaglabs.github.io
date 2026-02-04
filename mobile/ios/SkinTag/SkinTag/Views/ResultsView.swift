//
//  ResultsView.swift
//  SkinTag
//
//  Display triage results with medical disclaimer
//

import SwiftUI

struct ResultsView: View {
    let result: TriageResult
    let image: UIImage
    let onDismiss: () -> Void

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Captured image
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 200)
                        .cornerRadius(12)
                        .shadow(radius: 5)

                    // Risk indicator
                    RiskGaugeView(result: result)

                    // Triage tier
                    TriageTierCard(result: result)

                    // Medical disclaimer
                    DisclaimerCard()

                    // Action buttons
                    ActionButtonsView(onDismiss: onDismiss)
                }
                .padding()
            }
            .navigationTitle("Analysis Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        onDismiss()
                    }
                }
            }
        }
    }
}

struct RiskGaugeView: View {
    let result: TriageResult

    var body: some View {
        VStack(spacing: 12) {
            // Gauge
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.2), lineWidth: 20)
                    .frame(width: 150, height: 150)

                Circle()
                    .trim(from: 0, to: CGFloat(result.malignantProbability))
                    .stroke(
                        result.riskColor,
                        style: StrokeStyle(lineWidth: 20, lineCap: .round)
                    )
                    .frame(width: 150, height: 150)
                    .rotationEffect(.degrees(-90))

                VStack(spacing: 4) {
                    Text("\(Int(result.malignantProbability * 100))%")
                        .font(.system(size: 36, weight: .bold))
                        .foregroundColor(result.riskColor)

                    Text("Risk Score")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Text("Probability of requiring medical attention")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
    }
}

struct TriageTierCard: View {
    let result: TriageResult

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: result.tierIcon)
                    .font(.title)
                    .foregroundColor(result.riskColor)

                Text(result.tierName)
                    .font(.title2)
                    .fontWeight(.bold)
            }

            Text(result.tierDescription)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            // Recommendation
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)

                Text(result.recommendation)
                    .font(.subheadline)
            }
            .padding()
            .background(Color.yellow.opacity(0.1))
            .cornerRadius(12)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
    }
}

struct DisclaimerCard: View {
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                Text("Important Reminder")
                    .font(.headline)
                    .foregroundColor(.red)
            }

            Text("This result is NOT a medical diagnosis. It is an AI-generated screening suggestion only. Always consult a qualified dermatologist or healthcare provider for proper evaluation of any skin concern.")
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.red.opacity(0.3), lineWidth: 1)
        )
    }
}

struct ActionButtonsView: View {
    let onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Button(action: onDismiss) {
                HStack {
                    Image(systemName: "camera.fill")
                    Text("Take Another Photo")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
            }

            Button(action: {
                // Could open health app or dermatologist finder
            }) {
                HStack {
                    Image(systemName: "stethoscope")
                    Text("Find a Dermatologist")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color(UIColor.secondarySystemBackground))
                .foregroundColor(.primary)
                .cornerRadius(12)
            }
        }
    }
}

#Preview {
    ResultsView(
        result: TriageResult(
            benignProbability: 0.3,
            malignantProbability: 0.7
        ),
        image: UIImage(systemName: "photo")!,
        onDismiss: {}
    )
}
