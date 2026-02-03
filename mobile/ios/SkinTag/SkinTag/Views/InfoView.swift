//
//  InfoView.swift
//  SkinTag
//
//  Information about the app and how it works
//

import SwiftUI

struct InfoView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)

                    Text("About SkinTag")
                        .font(.title)
                        .fontWeight(.bold)

                    Text("AI-Powered Skin Lesion Screening")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical)

                Divider()

                // How it works
                InfoSection(
                    title: "How It Works",
                    content: """
                    SkinTag uses a MobileNetV3 deep learning model trained on thousands of dermatology images to analyze skin lesions. When you take a photo, the app:

                    1. Preprocesses the image to match training conditions
                    2. Runs the AI model to extract visual features
                    3. Classifies the lesion as low, moderate, or high concern
                    4. Provides a recommendation for next steps

                    The analysis happens entirely on your device - no images are uploaded to any server.
                    """
                )

                InfoSection(
                    title: "Training Data",
                    content: """
                    The model was trained on a combination of:
                    - Fitzpatrick17k: Diverse skin tones dataset
                    - PAD-UFES-20: Brazilian dermatology images
                    - ISIC Archive: International skin imaging collaboration

                    This diverse training helps the model perform across different skin types and conditions.
                    """
                )

                InfoSection(
                    title: "Limitations",
                    content: """
                    - Image quality affects results (lighting, focus, angle)
                    - The model may not recognize rare conditions
                    - Performance varies across skin types
                    - Cannot detect non-visible characteristics
                    - Should not replace professional dermoscopy

                    Always seek professional medical advice for any skin concerns.
                    """
                )

                InfoSection(
                    title: "Privacy",
                    content: """
                    Your privacy is important to us:
                    - All analysis is done on-device
                    - No images are uploaded to servers
                    - No personal data is collected
                    - The app requires camera access only when actively taking photos
                    """
                )

                // Version info
                HStack {
                    Text("Version")
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("1.0.0")
                        .foregroundColor(.secondary)
                }
                .padding(.top)
            }
            .padding()
        }
        .navigationTitle("Information")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct InfoSection: View {
    let title: String
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)

            Text(content)
                .font(.body)
                .foregroundColor(.secondary)
        }
    }
}

#Preview {
    NavigationView {
        InfoView()
    }
}
