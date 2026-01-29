# SkinTag: Multi-Dataset, Domain-Robust, Fairness-Aware Skin Lesion Triage

## Vision

An AI-powered screening tool for low-resource dermatological settings — helping people with limited access to dermatologists get early risk assessment from consumer phone photos. Not a diagnostic tool; a triage aid that tells users when to seek professional care urgently.

**Target venue**: NeurIPS (with explicit caveats about screening vs. diagnosis).

---

## Problem

1. **Domain bias**: Most dermatology AI is trained on dermoscopic images. Consumer phone photos look completely different. Models learn "dermoscope artifacts = pathological" instead of actual lesion features.
2. **Skin tone bias**: Training data is overwhelmingly Fitzpatrick skin types I-III (lighter skin). Models have lower sensitivity on types IV-VI (darker skin) — exactly the populations with worst access to dermatologists.
3. **Binary triage gap**: Users don't need a 114-class differential diagnosis. They need: "Is this urgent enough to see a doctor?"

## Approach

- **Multi-dataset training** across four complementary datasets spanning dermoscopic, clinical, and smartphone domains
- **Domain-bridging augmentations** that add/remove dermoscope artifacts so the model can't cheat
- **Fitzpatrick-balanced sampling** that explicitly upweights under-represented darker skin tones
- **Three modeling approaches** (naive baseline, classical ML, deep learning) for rigorous comparison
- **Triage output** with urgency tiers, not raw probabilities
- **Polished web app** for consumer use with prominent medical disclaimers

---

## Datasets

| Dataset | Images | Domain | Fitzpatrick | Labels | Source URL |
|---------|--------|--------|-------------|--------|------------|
| **HAM10000** | 10,015 | Dermoscopic | No | 7 classes -> binary | [Kaggle](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) |
| **DDI** (Stanford) | 656 | Clinical | Yes (grouped I-II, III-IV, V-VI) | Biopsy-proven benign/malignant | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) |
| **Fitzpatrick17k** | ~16,577 | Clinical | Yes (1-6) | 114 conditions -> three_partition_label (benign/malignant/non-neoplastic) | [GitHub](https://github.com/mattgroh/fitzpatrick17k) |
| **PAD-UFES-20** | 2,298 | Smartphone | Yes (via `fitspatrick` column) | 6 diagnostic categories -> binary | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) |

### Why these four

- **HAM10000**: Large, well-labeled dermoscopic dataset. The baseline.
- **DDI**: Only dataset with balanced representation across all Fitzpatrick types. Gold standard for fairness evaluation. Biopsy-proven labels.
- **Fitzpatrick17k**: Adds volume of clinical images with per-image Fitzpatrick annotation. The `three_partition_label` column provides reliable binary mapping without fuzzy condition name matching. **Non-neoplastic conditions are included as benign** (see Label Taxonomy Decision below).
- **PAD-UFES-20**: The critical smartphone-domain dataset. Breaks the dermoscope-pathology correlation. Also provides Fitzpatrick annotations, age, gender, and rich clinical metadata.

### Potential Future Dataset

| Dataset | Images | Domain | Fitzpatrick | Labels | Source URL |
|---------|--------|--------|-------------|--------|------------|
| **BCN20000** | 18,946 (12,413 labeled) | Dermoscopic | No | 8 classes (adds SCC) | [ISIC Archive](https://api.isic-archive.com/collections/249/) |

BCN20000 (Hospital Clinic Barcelona, 2010-2016) adds SCC as an explicit class and more dermoscopic volume, but does not improve our weakest areas (skin tone coverage, smartphone domain, clinical diversity). It shares the same domain as HAM10000. Publicly available via the ISIC Archive with no access request. Consider adding if more dermoscopic volume or SCC representation is needed after initial experiments.

**Reference**: Hernández-Pérez et al., "BCN20000: Dermoscopic Lesions in the Wild." *Scientific Data* (2024). [DOI: 10.1038/s41597-024-03387-w](https://www.nature.com/articles/s41597-024-03387-w)

### Dataset CSV Column Reference

**HAM10000** (`HAM10000_metadata.csv`):
- `image_id`, `dx` (diagnosis: akiec/bcc/bkl/df/mel/nv/vasc), `dx_type`, `age`, `sex`, `localization`

**DDI** (`ddi_metadata.csv`):
- `DDI_file` (image filename), `skin_tone` (values: 12, 34, 56 = grouped FST), `malignant` (bool True/False), `disease` (condition name)
- Note: actual column is `malignant` (bool), not `malignancy(malig=1)` as some docs suggest. Loader handles both.

**Fitzpatrick17k** (`fitzpatrick17k.csv`):
- `md5hash` (image identifier/filename), `label` (114 conditions), `three_partition_label` (benign/malignant/non-neoplastic), `fitzpatrick` (1-6)

**PAD-UFES-20** (`metadata.csv`):
- `img_id`, `diagnostic` (ACK/BCC/MEL/NEV/SCC/SEK), `fitspatrick` (note: misspelled in original), `age`, `gender`, `region`

### Label Taxonomy Decision

**Decision: Binary (benign/malignant) with non-neoplastic included as benign.**

Evaluated in `notebooks/label_taxonomy_eda.ipynb`. Four options were considered:

| Option | Description | Total Data | Verdict |
|--------|-------------|-----------|---------|
| A: Binary (original) | Exclude Fitz17k non-neoplastic | ~21,500 | Loses ~48% of Fitz17k |
| B: Ternary | Benign/malignant/non-neoplastic | ~29,500 | Only Fitz17k has 3rd class |
| **C: Binary + non-neo as benign** | **Include non-neoplastic as benign** | **~29,500** | **Selected** |
| D: Binary + normal skin | Add healthy skin class | ~21,500 + extra | No dataset available |

**Rationale**:
- Non-neoplastic conditions (eczema, psoriasis, infections) are **not cancer** — for a triage system asking "is this urgent?", they belong in the benign bucket
- Recovers ~8,000 additional images from Fitzpatrick17k (48% of that dataset)
- All four datasets already have lesion-only images; no "normal skin" dataset exists for a 3rd class
- Confidence-based triage tiers (already in `triage.py`) handle ambiguity without extra classes
- HAM10000's large `nv` class (6,705 melanocytic nevi) already serves as "normal mole" representation

**Normal skin class not needed** because:
- Users upload photos of specific lesions they're concerned about (pre-selected input)
- No major public dataset includes "normal skin" images
- An OOD detector can flag non-lesion inputs without requiring a trained class

**Key references**: Daneshjou et al. (DDI, 2022), Groh et al. (Fitz17k, 2021), ISIC 2024 Challenge (binary pAUC metric validates binary triage approach).

### Downloading Data Locally

**Prerequisites**: Set Kaggle credentials as environment variables (or in `.env`):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**HAM10000** (automated via Kaggle):
```bash
pip install kaggle
kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip
# Or use: make data
```

**DDI** (requires Stanford AIMI access):
```bash
# 1. Request access at: https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965
# 2. Once approved, download and extract to:
mkdir -p data/ddi/images
# Place ddi_metadata.csv in data/ddi/
# Place all images in data/ddi/images/
```

**Fitzpatrick17k** (CSV from GitHub, images via URLs in CSV):
```bash
mkdir -p data/fitzpatrick17k/images
# Download CSV:
curl -L https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv \
  -o data/fitzpatrick17k/fitzpatrick17k.csv
# Download images using the URLs in the CSV (see scripts/download_datasets.py)
python download_datasets.py --dataset fitzpatrick17k
```

**PAD-UFES-20** (Mendeley Data):
```bash
mkdir -p data/pad_ufes/images
# 1. Download from: https://data.mendeley.com/datasets/zr7vgbcyr2/1
# 2. Also available on Kaggle:
kaggle datasets download -d mahdavi1202/skin-cancer -p data/pad_ufes/ --unzip
# Place metadata.csv in data/pad_ufes/
# Place all images in data/pad_ufes/images/
```

### Downloading the SigLIP Model Locally

The SigLIP backbone (`google/siglip-so400m-patch14-384`) is ~1.6GB and downloaded automatically from HuggingFace on first training run. To pre-download:

```bash
python -c "
from transformers import AutoModel, AutoImageProcessor
model_name = 'google/siglip-so400m-patch14-384'
print('Downloading SigLIP processor...')
AutoImageProcessor.from_pretrained(model_name)
print('Downloading SigLIP model...')
AutoModel.from_pretrained(model_name)
print('Done — model cached in ~/.cache/huggingface/')
"
```

The model is cached at `~/.cache/huggingface/hub/` and is gitignored. To set a custom cache location:
```bash
export HF_HOME=/path/to/your/cache
```

### Local Data Layout

```
data/                                  # gitignored — all datasets stored locally
├── HAM10000_metadata.csv
├── Skin Cancer/
│   └── Skin Cancer/                   # Kaggle nests one level deep
│       └── *.jpg                      # 10,015 dermoscopic images
├── ddi/
│   ├── ddi_metadata.csv
│   └── images/
│       └── *.jpg                      # 656 clinical images
├── fitzpatrick17k/
│   ├── fitzpatrick17k.csv
│   └── images/
│       └── *.jpg                      # ~16,577 clinical images
└── pad_ufes/
    ├── metadata.csv
    └── images/
        └── *.png                      # 2,298 smartphone images
```

---

## Three Required Models

| Model | Type | Architecture | How it works |
|-------|------|-------------|-------------|
| **Naive baseline** | Majority class | Always predicts "benign" | No training. Sets floor for accuracy. |
| **Classical ML** | Logistic Regression | StandardScaler -> LogisticRegression on 1152-d SigLIP embeddings | sklearn, sub-second training |
| **Deep learning** | Fine-tuned MLP head | 2-layer MLP (1152->256->2) with dropout+BatchNorm on SigLIP embeddings | PyTorch, early stopping, class-weighted loss |
| **End-to-end** (optional) | Fine-tuned SigLIP | Unfreezes last N transformer layers + classification head | GPU required, best accuracy for deployment |

The **deployed model** will be whichever performs best on the cross-domain evaluation, ranked by **F1 macro** (the primary metric for imbalanced dermatology data).

---

## Focused Experiment: Domain Shift & Fairness Sensitivity

**Question**: Does training with domain-bridging augmentations and multi-dataset balancing improve cross-domain generalization and fairness across skin tones?

**Protocol**:
1. Train all 3 models on HAM10000 only (baseline condition)
2. Train all 3 models on multi-dataset (no augmentation, no balancing)
3. Train all 3 models on multi-dataset with domain+Fitzpatrick balanced weights
4. For each: evaluate per-Fitzpatrick-type F1, per-domain F1, fairness gaps

**Primary metrics**: F1 macro, F1 (malignant), per-Fitzpatrick-type sensitivity, equalized odds gap, cross-domain F1 gap

---

## Fairness & Skin Tone Analysis

### The Problem

Dermatology datasets are overwhelmingly light-skinned:
- HAM10000: No Fitzpatrick annotations at all
- Fitzpatrick17k: Skewed toward types I-III
- Most clinical training data: 80%+ Fitzpatrick I-III

This means models trained naively will have significantly lower sensitivity (miss more cancers) on darker skin tones — precisely the populations with worst access to dermatologists.

### Our Approach

1. **Fitzpatrick-balanced sampling**: Upweights under-represented (Fitzpatrick type, label) pairs so each skin tone contributes equally to training loss
2. **DDI as fairness benchmark**: The only dataset with deliberate balanced representation across all skin tone groups
3. **Per-Fitzpatrick evaluation**: Report F1, sensitivity, specificity, and AUC broken down by Fitzpatrick type
4. **Equalized odds gap**: Measure the maximum difference in sensitivity and specificity across Fitzpatrick groups — the key fairness metric

---

## Project Structure

```
SkinTag/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # CI pipeline
│       └── train.yml                  # Training workflow
├── app/
│   ├── main.py                        # FastAPI backend (upload -> SigLIP -> triage)
│   └── templates/
│       └── index.html                 # Polished dark-theme frontend with risk gauge
├── configs/
│   └── config.yaml                    # All configuration (data, training, triage thresholds)
├── data/                              # gitignored — local datasets
├── notebooks/
│   ├── colab_demo.ipynb               # Google Colab demo notebook
│   ├── demo.ipynb                     # Local demo notebook
│   ├── label_taxonomy_eda.ipynb       # Label taxonomy & data enrichment EDA
│   ├── quick_start_with_hugging_face.ipynb
│   ├── skin_tone_eda.ipynb            # Skin tone exploratory analysis
│   └── skin_tone_eda_executed.ipynb
├── results/                           # gitignored — cached embeddings, models, metrics
│   └── cache/
│       ├── embeddings.pt              # Cached SigLIP embeddings
│       ├── classifier.pkl             # Default trained model
│       ├── classifier_logistic.pkl    # Logistic regression model
│       ├── classifier_deep.pkl        # Deep MLP head model
│       ├── classifier_baseline.pkl    # Majority class baseline
│       ├── finetuned_model/           # End-to-end fine-tuned SigLIP (if trained)
│       ├── metadata.csv               # Full dataset metadata
│       ├── test_metadata.csv          # Test split metadata
│       └── training_results.json      # Model comparison metrics
├── scripts/
│   ├── train.py                       # Main training (--multi-dataset --model all --domain-balance)
│   ├── train_all_models.py            # Train + compare all model types
│   ├── evaluate.py                    # Full fairness evaluation report
│   └── evaluate_cross_domain.py       # Leave-one-domain-out experiment
├── src/
│   ├── data/
│   │   ├── schema.py                  # SkinSample dataclass (unified schema for all datasets)
│   │   ├── loader.py                  # load_ham10000(), load_multi_dataset() orchestrator
│   │   ├── datasets/
│   │   │   ├── __init__.py            # DATASET_LOADERS registry
│   │   │   ├── ham10000.py            # HAM10000 adapter -> SkinSample
│   │   │   ├── ddi.py                 # DDI adapter (skin_tone -> Fitzpatrick mapping)
│   │   │   ├── fitzpatrick17k.py      # Fitz17k adapter (three_partition_label -> binary)
│   │   │   └── pad_ufes.py            # PAD-UFES adapter (smartphone domain)
│   │   ├── sampler.py                 # Domain + Fitzpatrick balanced sampling weights
│   │   ├── augmentations.py           # Training/eval transforms + domain bridging
│   │   └── dermoscope_aug.py          # Custom dermoscope artifact augmentations
│   ├── model/
│   │   ├── embeddings.py              # SigLIP embedding extraction + caching
│   │   ├── classifier.py              # SklearnClassifier (logistic regression)
│   │   ├── baseline.py                # MajorityClassBaseline + RandomWeightedBaseline
│   │   ├── deep_classifier.py         # DeepClassifier (MLP head) + EndToEndClassifier
│   │   └── triage.py                  # TriageSystem with urgency tiers + disclaimers
│   └── evaluation/
│       └── metrics.py                 # F1, accuracy, AUC, per-group, equalized odds
├── .env                               # gitignored — Kaggle credentials, env vars
├── .gitignore
├── Dockerfile                         # Containerized deployment
├── LICENSE
├── Makefile                           # Build targets (install, data, train, app)
├── PLAN.md                            # This file
├── README.md
└── requirements.txt
```

---

## Usage

### Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
# Or:
make install

# Download HAM10000 (requires Kaggle API credentials)
make data

# Download other datasets (see "Downloading Data Locally" above)
# DDI: requires Stanford AIMI access request
# Fitzpatrick17k: CSV from GitHub, images from URLs
# PAD-UFES-20: from Mendeley Data or Kaggle
```

### Training
```bash
# Single dataset (HAM10000), logistic regression
python scripts/train.py

# All 3 models on HAM10000
python scripts/train.py --model all

# Multi-dataset, all models, domain+Fitzpatrick balanced
python scripts/train.py --multi-dataset --domain-balance --model all

# Train + compare all model types
python scripts/train_all_models.py

# Quick smoke test (500 samples)
python scripts/train.py --sample 500
```

### Evaluation
```bash
# Full fairness report
python scripts/evaluate.py --models logistic deep baseline

# Cross-domain experiment
python scripts/evaluate_cross_domain.py
```

### Web App
```bash
# Local development
make app                           # http://localhost:8000

# Docker
make app-docker
```

---

## Key Design Decisions

1. **F1 macro as primary metric** — accuracy is misleading on class-imbalanced data (70% benign -> 70% accuracy by always predicting benign). F1 macro treats both classes equally.

2. **Fitzpatrick-balanced sampling, not oversampling** — oversampling duplicates minority images (overfitting risk). Balanced sample weights upweight minority samples in the loss function without duplication.

3. **Domain-bridging augmentations** — randomly add dermoscope artifacts to phone photos and remove them from dermoscopic images. The model sees all label-domain combinations, breaking spurious correlations.

4. **three_partition_label for Fitzpatrick17k** — the dataset provides a reliable benign/malignant/non-neoplastic partition. Using this instead of fuzzy string matching against 114 condition names eliminates misclassification risk. **Non-neoplastic is included as benign** (not cancer = benign for triage). This recovers ~8,000 images that were previously excluded.

5. **DDI skin_tone as grouped Fitzpatrick** — DDI uses groups (I-II, III-IV, V-VI) not individual types. We map to midpoints (1, 3, 5) for consistency but note the grouping in analysis.

6. **Triage tiers, not probabilities** — users don't understand "0.47 probability of malignancy." They understand "moderate concern — schedule a dermatology appointment within 2-4 weeks."

7. **Medical disclaimer everywhere** — this is a screening aid, not a diagnosis. Every output includes a prominent disclaimer. The app makes this impossible to miss.

8. **SigLIP (google/siglip-so400m-patch14-384) as backbone** — 400M parameter vision-language model producing 1152-d embeddings. Strong zero-shot medical image understanding. Model is cached locally via HuggingFace Hub (~1.6GB) and gitignored.
