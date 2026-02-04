//
//  DisclaimerView.swift
//  SkinTag
//
//  Medical disclaimer that must be accepted before using the app
//

import SwiftUI

struct DisclaimerView: View {
    @EnvironmentObject var appState: AppState
    @State private var hasScrolledToBottom = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {
                            // Header
                            VStack(spacing: 12) {
                                Image(systemName: "cross.case.fill")
                                    .font(.system(size: 50))
                                    .foregroundColor(.red)

                                Text("Important Medical Disclaimer")
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .multilineTextAlignment(.center)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.top)

                            Divider()

                            // Disclaimer content
                            Group {
                                DisclaimerSection(
                                    icon: "exclamationmark.triangle.fill",
                                    iconColor: .orange,
                                    title: "This Is NOT a Medical Diagnosis",
                                    content: "SkinTag is an educational screening tool only. It does NOT provide medical diagnoses, medical advice, or treatment recommendations. The results should NEVER be used as a substitute for professional medical evaluation."
                                )

                                DisclaimerSection(
                                    icon: "stethoscope",
                                    iconColor: .blue,
                                    title: "Always Consult a Doctor",
                                    content: "Any skin lesion that concerns you should be evaluated by a qualified dermatologist or healthcare provider. Early detection by medical professionals saves lives."
                                )

                                DisclaimerSection(
                                    icon: "chart.bar.fill",
                                    iconColor: .purple,
                                    title: "AI Limitations",
                                    content: "This AI model has limitations and can make errors. It was trained on specific datasets and may not perform equally well on all skin types, lighting conditions, or lesion types. False positives and false negatives are possible."
                                )

                                DisclaimerSection(
                                    icon: "person.fill.questionmark",
                                    iconColor: .teal,
                                    title: "Not for Emergency Use",
                                    content: "If you notice rapid changes in a skin lesion, bleeding, or other concerning symptoms, seek immediate medical attention. Do not rely on this app for urgent medical decisions."
                                )

                                DisclaimerSection(
                                    icon: "checkmark.shield.fill",
                                    iconColor: .green,
                                    title: "Intended Use",
                                    content: "This tool is intended to help raise awareness about skin health and encourage users to seek professional evaluation when appropriate. Use it as one of many tools in your health awareness toolkit."
                                )
                            }

                            // Agreement text
                            Text("By tapping 'I Understand and Accept' below, you acknowledge that you have read and understood this disclaimer, and agree that SkinTag is not a substitute for professional medical care.")
                                .font(.footnote)
                                .foregroundColor(.secondary)
                                .padding(.vertical)
                                .id("bottom")
                        }
                        .padding()
                    }
                    .onAppear {
                        // Auto-scroll check could be added here
                        hasScrolledToBottom = true
                    }
                }

                // Accept button
                VStack(spacing: 12) {
                    Divider()

                    Button(action: {
                        withAnimation {
                            appState.hasAcceptedDisclaimer = true
                        }
                    }) {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                            Text("I Understand and Accept")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                }
                .background(Color(UIColor.systemBackground))
            }
            .navigationBarHidden(true)
        }
    }
}

struct DisclaimerSection: View {
    let icon: String
    let iconColor: Color
    let title: String
    let content: String

    var body: some View {
        HStack(alignment: .top, spacing: 15) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(iconColor)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)

                Text(content)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

#Preview {
    DisclaimerView()
        .environmentObject(AppState())
}
