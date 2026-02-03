//
//  ModelInference.swift
//  SkinTag
//
//  Core ML model inference for skin lesion classification
//

import CoreML
import Vision
import UIKit

/// Handles loading and running the Core ML model
class ModelInference {
    private let model: VNCoreMLModel
    private let imageSize = CGSize(width: 224, height: 224)

    // ImageNet normalization values used by MobileNet
    private let meanRGB: [Float] = [0.485, 0.456, 0.406]
    private let stdRGB: [Float] = [0.229, 0.224, 0.225]

    init() throws {
        // Try to load the model
        // In production, the model would be bundled with the app
        guard let modelURL = Bundle.main.url(forResource: "SkinTagModel", withExtension: "mlmodelc") else {
            // Fall back to trying .mlmodel
            if let mlmodelURL = Bundle.main.url(forResource: "SkinTagModel", withExtension: "mlmodel") {
                let compiledURL = try MLModel.compileModel(at: mlmodelURL)
                let mlModel = try MLModel(contentsOf: compiledURL)
                self.model = try VNCoreMLModel(for: mlModel)
                return
            }
            throw ModelError.modelNotFound
        }

        let mlModel = try MLModel(contentsOf: modelURL)
        self.model = try VNCoreMLModel(for: mlModel)
    }

    /// Run inference on an image
    /// - Parameter image: The captured UIImage
    /// - Returns: TriageResult with probabilities
    func predict(image: UIImage) -> TriageResult {
        // Preprocess image
        guard let processedImage = preprocessImage(image) else {
            // Return neutral result on preprocessing failure
            return TriageResult(benignProbability: 0.5, malignantProbability: 0.5)
        }

        // Create request
        var result = TriageResult(benignProbability: 0.5, malignantProbability: 0.5)
        let semaphore = DispatchSemaphore(value: 0)

        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("Inference error: \(error)")
                semaphore.signal()
                return
            }

            guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = observations.first?.featureValue.multiArrayValue else {
                // Try classification observations
                if let classificationResults = request.results as? [VNClassificationObservation] {
                    result = self.processClassificationResults(classificationResults)
                }
                semaphore.signal()
                return
            }

            result = self.processMultiArrayResult(multiArray)
            semaphore.signal()
        }

        // Configure request
        request.imageCropAndScaleOption = .centerCrop

        // Run inference
        let handler = VNImageRequestHandler(cgImage: processedImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform inference: \(error)")
        }

        semaphore.wait()
        return result
    }

    /// Preprocess image for model input
    private func preprocessImage(_ image: UIImage) -> CGImage? {
        // Resize to model input size
        let size = imageSize
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return resizedImage?.cgImage
    }

    /// Process MLMultiArray output (for models that output logits/probabilities)
    private func processMultiArrayResult(_ multiArray: MLMultiArray) -> TriageResult {
        // Assuming output shape is [1, 2] for [benign, malignant] probabilities
        let count = multiArray.count
        guard count >= 2 else {
            return TriageResult(benignProbability: 0.5, malignantProbability: 0.5)
        }

        // Extract raw values
        let val0 = multiArray[0].doubleValue
        let val1 = multiArray[1].doubleValue

        // Check if values are already probabilities (sum close to 1)
        if abs((val0 + val1) - 1.0) < 0.01 {
            return TriageResult(
                benignProbability: val0,
                malignantProbability: val1
            )
        }

        // Apply softmax if logits
        let maxVal = max(val0, val1)
        let exp0 = exp(val0 - maxVal)
        let exp1 = exp(val1 - maxVal)
        let sum = exp0 + exp1

        return TriageResult(
            benignProbability: exp0 / sum,
            malignantProbability: exp1 / sum
        )
    }

    /// Process classification observation output
    private func processClassificationResults(_ results: [VNClassificationObservation]) -> TriageResult {
        var benignProb = 0.5
        var malignantProb = 0.5

        for observation in results {
            let label = observation.identifier.lowercased()
            if label.contains("benign") || label == "0" {
                benignProb = Double(observation.confidence)
            } else if label.contains("malignant") || label == "1" {
                malignantProb = Double(observation.confidence)
            }
        }

        // Normalize if needed
        let total = benignProb + malignantProb
        if total > 0 && abs(total - 1.0) > 0.01 {
            benignProb /= total
            malignantProb /= total
        }

        return TriageResult(
            benignProbability: benignProb,
            malignantProbability: malignantProb
        )
    }
}

enum ModelError: Error, LocalizedError {
    case modelNotFound
    case preprocessingFailed
    case inferenceFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "The AI model could not be found. Please reinstall the app."
        case .preprocessingFailed:
            return "Failed to process the image."
        case .inferenceFailed:
            return "The AI model failed to analyze the image."
        }
    }
}
