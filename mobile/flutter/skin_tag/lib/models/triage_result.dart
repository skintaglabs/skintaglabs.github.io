import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Triage risk tier classification
enum TriageTier {
  low,
  moderate,
  high,
}

/// Represents the result of skin lesion analysis
class TriageResult {
  final double benignProbability;
  final double malignantProbability;

  TriageResult({
    required this.benignProbability,
    required this.malignantProbability,
  });

  /// Risk tier based on malignant probability
  TriageTier get tier {
    if (malignantProbability < 0.3) return TriageTier.low;
    if (malignantProbability < 0.7) return TriageTier.moderate;
    return TriageTier.high;
  }

  /// Human-readable tier name
  String get tierName {
    switch (tier) {
      case TriageTier.low:
        return 'Low Concern';
      case TriageTier.moderate:
        return 'Moderate Concern';
      case TriageTier.high:
        return 'Higher Concern';
    }
  }

  /// Icon for the tier
  IconData get tierIcon {
    switch (tier) {
      case TriageTier.low:
        return Icons.check_circle;
      case TriageTier.moderate:
        return Icons.error;
      case TriageTier.high:
        return Icons.warning;
    }
  }

  /// Description of the tier
  String get tierDescription {
    switch (tier) {
      case TriageTier.low:
        return 'The analysis suggests this lesion has characteristics typically associated with benign conditions.';
      case TriageTier.moderate:
        return 'The analysis shows some features that warrant professional evaluation. Consider scheduling a dermatology appointment.';
      case TriageTier.high:
        return 'The analysis indicates features that should be evaluated by a medical professional promptly.';
    }
  }

  /// Recommendation based on tier
  String get recommendation {
    switch (tier) {
      case TriageTier.low:
        return 'Continue monitoring for changes. Take photos periodically to track any evolution. Consult a doctor if you notice changes in size, shape, color, or texture.';
      case TriageTier.moderate:
        return 'We recommend scheduling an appointment with a dermatologist for professional evaluation within the next few weeks.';
      case TriageTier.high:
        return 'Please consult a dermatologist or healthcare provider soon for a thorough examination. Early professional evaluation is important.';
    }
  }

  /// Color for the risk tier
  Color get riskColor => AppColors.getRiskColor(malignantProbability);

  /// Risk percentage (0-100)
  int get riskPercentage => (malignantProbability * 100).round();
}
