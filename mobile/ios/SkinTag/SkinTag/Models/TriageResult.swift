//
//  TriageResult.swift
//  SkinTag
//
//  Model representing the triage classification result
//

import SwiftUI

/// Represents the result of skin lesion analysis
struct TriageResult {
    let benignProbability: Double
    let malignantProbability: Double

    /// Risk tier based on malignant probability
    var tier: TriageTier {
        if malignantProbability < 0.3 {
            return .low
        } else if malignantProbability < 0.7 {
            return .moderate
        } else {
            return .high
        }
    }

    var tierName: String {
        tier.rawValue
    }

    var tierIcon: String {
        switch tier {
        case .low:
            return "checkmark.circle.fill"
        case .moderate:
            return "exclamationmark.circle.fill"
        case .high:
            return "exclamationmark.triangle.fill"
        }
    }

    var tierDescription: String {
        switch tier {
        case .low:
            return "The analysis suggests this lesion has characteristics typically associated with benign conditions."
        case .moderate:
            return "The analysis shows some features that warrant professional evaluation. Consider scheduling a dermatology appointment."
        case .high:
            return "The analysis indicates features that should be evaluated by a medical professional promptly."
        }
    }

    var recommendation: String {
        switch tier {
        case .low:
            return "Continue monitoring for changes. Take photos periodically to track any evolution. Consult a doctor if you notice changes in size, shape, color, or texture."
        case .moderate:
            return "We recommend scheduling an appointment with a dermatologist for professional evaluation within the next few weeks."
        case .high:
            return "Please consult a dermatologist or healthcare provider soon for a thorough examination. Early professional evaluation is important."
        }
    }

    var riskColor: Color {
        switch tier {
        case .low:
            return .riskLow
        case .moderate:
            return .riskMedium
        case .high:
            return .riskHigh
        }
    }
}

enum TriageTier: String {
    case low = "Low Concern"
    case moderate = "Moderate Concern"
    case high = "Higher Concern"
}
