"""Triage output system for clinical decision support.

Converts model probability scores into actionable urgency tiers with
recommendations and mandatory medical disclaimers.

IMPORTANT: This is a screening aid, NOT a diagnostic tool. All outputs
include disclaimers directing users to qualified medical professionals.
"""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

from dataclasses import dataclass
from typing import Optional


@dataclass
class TriageResult:
    """Result of a triage assessment."""
    risk_score: float  # 0.0 to 1.0, probability of malignancy
    urgency_tier: str  # "low", "moderate", "high"
    recommendation: str
    confidence: str  # "low", "moderate", "high" based on model certainty
    disclaimer: str


class TriageSystem:
    """Configurable triage system for skin lesion risk assessment.

    Converts classifier output probabilities into urgency tiers
    with recommendations. Thresholds are configurable via config.yaml.
    """

    DEFAULT_THRESHOLDS = {
        "low_max": 0.3,
        "moderate_max": 0.6,
    }

    DEFAULT_RECOMMENDATIONS = {
        "low": (
            "Lower malignancy risk. This lesion has features more consistent with benign "
            "or non-cancerous skin conditions. We still recommend consulting a board-certified "
            "dermatologist for a professional evaluation. Monitor using the ABCDE method "
            "(Asymmetry, Border, Color, Diameter, Evolution) and photograph monthly."
        ),
        "moderate": (
            "Moderate concern. This lesion has some features that warrant professional review. "
            "Schedule an appointment with a dermatologist within 2-4 weeks. Photograph the "
            "area now for comparison at your visit. Note any recent changes in size, color, "
            "or shape."
        ),
        "high": (
            "High concern. This lesion has characteristics that should be evaluated promptly. "
            "Schedule a dermatology appointment as soon as possible, ideally within 1-2 weeks. "
            "Do not delay seeking care. Early detection significantly improves outcomes."
        ),
    }

    # Context-aware recommendations when triage category is known
    CATEGORY_RECOMMENDATIONS = {
        "inflammatory": (
            "This lesion has features consistent with an inflammatory or reactive skin condition. "
            "While less likely to be cancerous, inflammatory conditions can cause discomfort and "
            "may require treatment. We recommend seeing a dermatologist for proper diagnosis "
            "and management. Some inflammatory conditions can mimic or mask more serious lesions."
        ),
        "benign": (
            "This lesion has features most consistent with a benign skin growth. While unlikely "
            "to be harmful, we recommend confirming this with a dermatologist, especially if the "
            "lesion is new, changing, or causing concern. Regular skin checks are important "
            "for early detection of any changes."
        ),
    }

    DEFAULT_DISCLAIMER = (
        "IMPORTANT MEDICAL DISCLAIMER: This tool is an AI-powered screening aid and is "
        "NOT a medical diagnosis. It cannot replace examination by a qualified healthcare "
        "professional. False positives and false negatives are possible. If you have any "
        "concerns about a skin lesion, please consult a board-certified dermatologist "
        "regardless of this tool's output. In case of rapid changes, bleeding, or pain, "
        "seek immediate medical attention."
    )

    def __init__(self, config: dict = None):
        """Initialize from config dict (typically from config.yaml triage section)."""
        config = config or {}
        thresholds = config.get("thresholds", {})
        self.low_max = thresholds.get("low_max", self.DEFAULT_THRESHOLDS["low_max"])
        self.moderate_max = thresholds.get("moderate_max", self.DEFAULT_THRESHOLDS["moderate_max"])

        self.recommendations = {
            tier: config.get("recommendations", {}).get(tier, default)
            for tier, default in self.DEFAULT_RECOMMENDATIONS.items()
        }
        self.disclaimer = config.get("disclaimer", self.DEFAULT_DISCLAIMER)

    def assess(self, probability: float, dominant_category: str = None) -> TriageResult:
        """Assess a single malignancy probability score.

        Args:
            probability: Model output probability of malignancy (0.0 to 1.0)
            dominant_category: Optional triage category ("malignant", "inflammatory",
                "benign") to provide context-aware recommendations.

        Returns:
            TriageResult with urgency tier, recommendation, confidence, disclaimer
        """
        probability = float(probability)

        # Determine urgency tier
        if probability < self.low_max:
            tier = "low"
        elif probability < self.moderate_max:
            tier = "moderate"
        else:
            tier = "high"

        # Confidence based on distance from decision boundaries
        dist_to_boundary = min(
            abs(probability - self.low_max),
            abs(probability - self.moderate_max),
        )
        if dist_to_boundary > 0.2:
            confidence = "high"
        elif dist_to_boundary > 0.1:
            confidence = "moderate"
        else:
            confidence = "low"

        # Use category-aware recommendation for low/moderate tiers
        recommendation = self.recommendations[tier]
        if dominant_category in self.CATEGORY_RECOMMENDATIONS:
            if tier in ("low", "moderate"):
                recommendation = self.CATEGORY_RECOMMENDATIONS[dominant_category]

        # Inflammatory conditions still need medical attention -- don't show green "Low Risk"
        if tier == "low" and dominant_category == "inflammatory":
            tier = "moderate"
            confidence = "moderate"

        return TriageResult(
            risk_score=probability,
            urgency_tier=tier,
            recommendation=recommendation,
            confidence=confidence,
            disclaimer=self.disclaimer,
        )

    def assess_batch(self, probabilities) -> list[TriageResult]:
        """Assess multiple probability scores."""
        return [self.assess(p) for p in probabilities]
