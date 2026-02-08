"""Unified condition taxonomy for multi-dataset skin lesion classification.

Defines 10 condition categories that span all 5 datasets (HAM10000, DDI,
Fitzpatrick17k, PAD-UFES-20, BCN20000). Each dataset's raw labels are mapped
to this shared taxonomy, enabling both:
  - Binary triage (benign vs malignant)
  - Multi-class condition estimation (10 categories)
"""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

from enum import IntEnum


class Condition(IntEnum):
    """Unified skin lesion condition taxonomy (10 categories)."""
    MELANOMA = 0
    BASAL_CELL_CARCINOMA = 1
    SQUAMOUS_CELL_CARCINOMA = 2
    ACTINIC_KERATOSIS = 3
    MELANOCYTIC_NEVUS = 4
    SEBORRHEIC_KERATOSIS = 5
    DERMATOFIBROMA = 6
    VASCULAR_LESION = 7
    NON_NEOPLASTIC = 8
    OTHER_UNKNOWN = 9


# Display names for UI / reports
CONDITION_NAMES = {
    Condition.MELANOMA: "Melanoma",
    Condition.BASAL_CELL_CARCINOMA: "Basal Cell Carcinoma",
    Condition.SQUAMOUS_CELL_CARCINOMA: "Squamous Cell Carcinoma",
    Condition.ACTINIC_KERATOSIS: "Actinic Keratosis",
    Condition.MELANOCYTIC_NEVUS: "Melanocytic Nevus",
    Condition.SEBORRHEIC_KERATOSIS: "Seborrheic Keratosis",
    Condition.DERMATOFIBROMA: "Dermatofibroma",
    Condition.VASCULAR_LESION: "Vascular Lesion",
    Condition.NON_NEOPLASTIC: "Non-Neoplastic",
    Condition.OTHER_UNKNOWN: "Other/Unknown",
}

# Three-category triage mapping for clinical display
class TriageCategory:
    MALIGNANT = "malignant"
    INFLAMMATORY = "inflammatory"
    BENIGN = "benign"

CONDITION_TRIAGE = {
    Condition.MELANOMA: TriageCategory.MALIGNANT,
    Condition.BASAL_CELL_CARCINOMA: TriageCategory.MALIGNANT,
    Condition.SQUAMOUS_CELL_CARCINOMA: TriageCategory.MALIGNANT,
    Condition.ACTINIC_KERATOSIS: TriageCategory.MALIGNANT,
    Condition.DERMATOFIBROMA: TriageCategory.INFLAMMATORY,
    Condition.VASCULAR_LESION: TriageCategory.INFLAMMATORY,
    Condition.NON_NEOPLASTIC: TriageCategory.INFLAMMATORY,
    Condition.MELANOCYTIC_NEVUS: TriageCategory.BENIGN,
    Condition.SEBORRHEIC_KERATOSIS: TriageCategory.BENIGN,
    Condition.OTHER_UNKNOWN: TriageCategory.BENIGN,
}

TRIAGE_CATEGORY_NAMES = {
    TriageCategory.MALIGNANT: "Malignant / Pre-cancerous",
    TriageCategory.INFLAMMATORY: "Inflammatory / Reactive",
    TriageCategory.BENIGN: "Benign / Harmless",
}

# Binary mapping: Condition -> 0 (benign) or 1 (malignant)
CONDITION_BINARY = {
    Condition.MELANOMA: 1,
    Condition.BASAL_CELL_CARCINOMA: 1,
    Condition.SQUAMOUS_CELL_CARCINOMA: 1,
    Condition.ACTINIC_KERATOSIS: 1,  # pre-cancerous, treated as malignant for triage
    Condition.MELANOCYTIC_NEVUS: 0,
    Condition.SEBORRHEIC_KERATOSIS: 0,
    Condition.DERMATOFIBROMA: 0,
    Condition.VASCULAR_LESION: 0,
    Condition.NON_NEOPLASTIC: 0,
    Condition.OTHER_UNKNOWN: 0,  # conservative: unknown defaults to benign
}


# ---------------------------------------------------------------------------
# Per-dataset mapping dictionaries
# ---------------------------------------------------------------------------

# HAM10000: dx column values -> Condition
HAM10000_CONDITION_MAP = {
    "mel": Condition.MELANOMA,
    "bcc": Condition.BASAL_CELL_CARCINOMA,
    "akiec": Condition.ACTINIC_KERATOSIS,
    "nv": Condition.MELANOCYTIC_NEVUS,
    "bkl": Condition.SEBORRHEIC_KERATOSIS,
    "df": Condition.DERMATOFIBROMA,
    "vasc": Condition.VASCULAR_LESION,
}

# PAD-UFES-20: diagnostic column values -> Condition
PAD_UFES_CONDITION_MAP = {
    "MEL": Condition.MELANOMA,
    "BCC": Condition.BASAL_CELL_CARCINOMA,
    "SCC": Condition.SQUAMOUS_CELL_CARCINOMA,
    "ACK": Condition.ACTINIC_KERATOSIS,
    "NEV": Condition.MELANOCYTIC_NEVUS,
    "SEK": Condition.SEBORRHEIC_KERATOSIS,
}

# BCN20000: diagnosis column values -> Condition
BCN20000_CONDITION_MAP = {
    "melanoma": Condition.MELANOMA,
    "basal cell carcinoma": Condition.BASAL_CELL_CARCINOMA,
    "squamous cell carcinoma": Condition.SQUAMOUS_CELL_CARCINOMA,
    "actinic keratosis": Condition.ACTINIC_KERATOSIS,
    "melanocytic nevus": Condition.MELANOCYTIC_NEVUS,
    "nevus": Condition.MELANOCYTIC_NEVUS,
    "seborrheic keratosis": Condition.SEBORRHEIC_KERATOSIS,
    "dermatofibroma": Condition.DERMATOFIBROMA,
    "vascular lesion": Condition.VASCULAR_LESION,
    # BCN20000 has additional labels that map to OTHER_UNKNOWN
    "solar lentigo": Condition.OTHER_UNKNOWN,
    "lichenoid keratosis": Condition.OTHER_UNKNOWN,
    "atypical melanocytic proliferation": Condition.OTHER_UNKNOWN,
}


# ---------------------------------------------------------------------------
# Keyword-based matchers for DDI and Fitzpatrick17k
# ---------------------------------------------------------------------------

# DDI: disease column has free-text condition names
# Keywords checked in order (first match wins)
_DDI_KEYWORD_MAP = [
    ("melanoma", Condition.MELANOMA),
    ("basal cell", Condition.BASAL_CELL_CARCINOMA),
    ("squamous cell", Condition.SQUAMOUS_CELL_CARCINOMA),
    ("actinic keratosis", Condition.ACTINIC_KERATOSIS),
    ("nevus", Condition.MELANOCYTIC_NEVUS),
    ("nevi", Condition.MELANOCYTIC_NEVUS),
    ("seborrheic keratosis", Condition.SEBORRHEIC_KERATOSIS),
    ("dermatofibroma", Condition.DERMATOFIBROMA),
    ("vascular", Condition.VASCULAR_LESION),
    ("hemangioma", Condition.VASCULAR_LESION),
    ("angioma", Condition.VASCULAR_LESION),
    ("pyogenic granuloma", Condition.VASCULAR_LESION),
    ("dermatitis", Condition.NON_NEOPLASTIC),
    ("eczema", Condition.NON_NEOPLASTIC),
    ("psoriasis", Condition.NON_NEOPLASTIC),
    ("lichen", Condition.NON_NEOPLASTIC),
    ("infection", Condition.NON_NEOPLASTIC),
    ("fungal", Condition.NON_NEOPLASTIC),
    ("wart", Condition.NON_NEOPLASTIC),
    ("verruca", Condition.NON_NEOPLASTIC),
]


def map_ddi_condition(disease_name: str) -> Condition:
    """Map a DDI disease name to a Condition using keyword matching."""
    if not disease_name:
        return Condition.OTHER_UNKNOWN
    lower = str(disease_name).lower().strip()
    for keyword, condition in _DDI_KEYWORD_MAP:
        if keyword in lower:
            return condition
    return Condition.OTHER_UNKNOWN


# Fitzpatrick17k: label column has 114+ condition names
_FITZ17K_KEYWORD_MAP = [
    ("melanoma", Condition.MELANOMA),
    ("basal cell", Condition.BASAL_CELL_CARCINOMA),
    ("squamous cell carcinoma", Condition.SQUAMOUS_CELL_CARCINOMA),
    ("actinic keratosis", Condition.ACTINIC_KERATOSIS),
    ("nevus", Condition.MELANOCYTIC_NEVUS),
    ("nevi", Condition.MELANOCYTIC_NEVUS),
    ("seborrheic keratosis", Condition.SEBORRHEIC_KERATOSIS),
    ("dermatofibroma", Condition.DERMATOFIBROMA),
    ("vascular", Condition.VASCULAR_LESION),
    ("hemangioma", Condition.VASCULAR_LESION),
    ("angioma", Condition.VASCULAR_LESION),
    ("pyogenic granuloma", Condition.VASCULAR_LESION),
    ("cherry angioma", Condition.VASCULAR_LESION),
    # Non-neoplastic (inflammatory, infectious, etc.)
    ("dermatitis", Condition.NON_NEOPLASTIC),
    ("eczema", Condition.NON_NEOPLASTIC),
    ("psoriasis", Condition.NON_NEOPLASTIC),
    ("lichen", Condition.NON_NEOPLASTIC),
    ("rosacea", Condition.NON_NEOPLASTIC),
    ("acne", Condition.NON_NEOPLASTIC),
    ("folliculitis", Condition.NON_NEOPLASTIC),
    ("impetigo", Condition.NON_NEOPLASTIC),
    ("tinea", Condition.NON_NEOPLASTIC),
    ("fungal", Condition.NON_NEOPLASTIC),
    ("wart", Condition.NON_NEOPLASTIC),
    ("verruca", Condition.NON_NEOPLASTIC),
    ("lupus", Condition.NON_NEOPLASTIC),
    ("vitiligo", Condition.NON_NEOPLASTIC),
    ("urticaria", Condition.NON_NEOPLASTIC),
    ("alopecia", Condition.NON_NEOPLASTIC),
    ("scabies", Condition.NON_NEOPLASTIC),
    ("herpes", Condition.NON_NEOPLASTIC),
    ("cellulitis", Condition.NON_NEOPLASTIC),
    ("keloid", Condition.NON_NEOPLASTIC),
    ("granuloma annulare", Condition.NON_NEOPLASTIC),
    ("pityriasis", Condition.NON_NEOPLASTIC),
    ("prurigo", Condition.NON_NEOPLASTIC),
    ("morphea", Condition.NON_NEOPLASTIC),
    ("scleroderma", Condition.NON_NEOPLASTIC),
    ("pemphigus", Condition.NON_NEOPLASTIC),
    ("bullous", Condition.NON_NEOPLASTIC),
]


def map_fitzpatrick17k_condition(label_name: str) -> Condition:
    """Map a Fitzpatrick17k condition label to a Condition using keyword matching."""
    if not label_name:
        return Condition.OTHER_UNKNOWN
    lower = str(label_name).lower().strip()
    for keyword, condition in _FITZ17K_KEYWORD_MAP:
        if keyword in lower:
            return condition
    return Condition.OTHER_UNKNOWN
