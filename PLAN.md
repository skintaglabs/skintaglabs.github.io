# SkinTag: Multi-Dataset, Domain-Robust, Fairness-Aware Skin Lesion Triage (Dual Classification)

## Vision

An AI-powered screening tool for low-resource dermatological settings — helping people with limited access to dermatologists get early risk assessment from consumer phone photos. Not a diagnostic tool; a triage aid that tells users when to seek professional care urgently.

**Target venue**: NeurIPS (with explicit caveats about screening vs. diagnosis).

---

## Problem

1. **Domain bias**: Most dermatology AI is trained on dermoscopic images. Consumer phone photos look completely different. Models learn "dermoscope artifacts = pathological" instead of actual lesion features.
2. **Skin tone bias**: Training data is overwhelmingly Fitzpatrick skin types I-III (lighter skin). Models have lower sensitivity on types IV-VI (darker skin) — exactly the populations with worst access to dermatologists.
3. **Binary triage gap**: Users don't need a 114-class differential diagnosis. They need: "Is this urgent enough to see a doctor?"

## Approach

- **Multi-dataset training** across five complementary datasets spanning dermoscopic, clinical, and smartphone domains
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
| **BCN20000** | 18,946 (12,413 labeled) | Dermoscopic | No | 8 classes (adds SCC) | [ISIC Archive](https://api.isic-archive.com/collections/249/) |

### Why these five

- **HAM10000**: Large, well-labeled dermoscopic dataset. The baseline.
- **DDI**: Only dataset with balanced representation across all Fitzpatrick types. Gold standard for fairness evaluation. Biopsy-proven labels.
- **Fitzpatrick17k**: Adds volume of clinical images with per-image Fitzpatrick annotation. The `three_partition_label` column provides reliable binary mapping without fuzzy condition name matching. **Non-neoplastic conditions are included as benign** (see Label Taxonomy Decision below).
- **PAD-UFES-20**: The critical smartphone-domain dataset. Breaks the dermoscope-pathology correlation. Also provides Fitzpatrick annotations, age, gender, and rich clinical metadata.
- **BCN20000**: Hospital Clinic Barcelona (2010-2016). Adds SCC as an explicit class and additional dermoscopic volume (~12k labeled images). Publicly available via the ISIC Archive with no access request. **Reference**: Hernández-Pérez et al., "BCN20000: Dermoscopic Lesions in the Wild." *Scientific Data* (2024). [DOI: 10.1038/s41597-024-03387-w](https://www.nature.com/articles/s41597-024-03387-w)

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

**BCN20000** (`bcn20000_metadata.csv`):
- `image` or `isic_id` (image identifier), `diagnosis` (melanoma/basal cell carcinoma/squamous cell carcinoma/actinic keratosis/melanocytic nevus/seborrheic keratosis/dermatofibroma/vascular lesion), `age`, `sex`, `anatom_site_general`

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
- All five datasets already have lesion-only images; no "normal skin" dataset exists for a 3rd class
- Confidence-based triage tiers (already in `triage.py`) handle ambiguity without extra classes
- HAM10000's large `nv` class (6,705 melanocytic nevi) already serves as "normal mole" representation

**Normal skin class not needed** because:
- Users upload photos of specific lesions they're concerned about (pre-selected input)
- No major public dataset includes "normal skin" images
- An OOD detector can flag non-lesion inputs without requiring a trained class

**Key references**: Daneshjou et al. (DDI, 2022), Groh et al. (Fitz17k, 2021), ISIC 2024 Challenge (binary pAUC metric validates binary triage approach).

### Dual Classification Targets

The pipeline trains **two sets of classifiers**:

1. **Binary triage** (benign=0, malignant=1) — the primary output for user-facing triage.
2. **Condition estimation** (10 categories) — secondary output showing the most likely specific condition.

Both targets are derived from a unified condition taxonomy defined in `src/data/taxonomy.py`.

### Unified Condition Taxonomy (10 Categories)

| ID | Condition | Binary | Present In |
|----|-----------|--------|------------|
| 0 | Melanoma | Malignant | All 5 |
| 1 | Basal Cell Carcinoma | Malignant | All 5 |
| 2 | Squamous Cell Carcinoma | Malignant | PAD, BCN, DDI, Fitz |
| 3 | Actinic Keratosis | Malignant | HAM, PAD, BCN, Fitz |
| 4 | Melanocytic Nevus | Benign | All 5 |
| 5 | Seborrheic Keratosis | Benign | All 5 |
| 6 | Dermatofibroma | Benign | HAM, BCN, DDI, Fitz |
| 7 | Vascular Lesion | Benign | HAM, BCN |
| 8 | Non-Neoplastic | Benign | Fitz, DDI |
| 9 | Other/Unknown | Benign | All (catch-all) |

Each dataset's raw labels are mapped to this taxonomy via per-dataset dictionaries (HAM10000, PAD-UFES, BCN20000) or keyword matchers (DDI, Fitzpatrick17k). The binary label is then derived from `CONDITION_BINARY[condition]`.

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

**BCN20000** (ISIC Archive — no access request needed):
```bash
mkdir -p data/bcn20000/images
# Download from: https://api.isic-archive.com/collections/249/
# Place bcn20000_metadata.csv in data/bcn20000/
# Place all images in data/bcn20000/images/
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
├── pad_ufes/
│   ├── metadata.csv
│   └── images/
│       └── *.png                      # 2,298 smartphone images
└── bcn20000/
    ├── bcn20000_metadata.csv
    └── images/
        └── ISIC_*.jpg                 # ~12,413 labeled dermoscopic images
```

---

## Model Architecture Comparison

| Model | Type | Architecture | Performance (Acc / F1 Macro / AUC) |
|-------|------|-------------|-----------------------------------|
| Naive baseline | Majority class | Always predicts "benign" | 0.791 / 0.442 / 0.500 |
| Logistic Regression | Classical ML | StandardScaler -> LogisticRegression on 1152-d frozen SigLIP embeddings | 0.840 / 0.792 / 0.922 |
| XGBoost (frozen) | Gradient boosting | XGBClassifier on frozen SigLIP embeddings | 0.957 / 0.938 / 0.990 |
| Fine-tuned SigLIP | End-to-end | Unfreezes last 4 transformer layers + 2-layer MLP head | 0.962 / 0.945 / 0.990 |
| **XGBoost (fine-tuned)** | **Two-stage** | **XGBClassifier on fine-tuned SigLIP embeddings** | **0.968 / 0.951 / 0.992** |

**Best overall model: XGBoost on fine-tuned SigLIP embeddings** — combines the embedding quality from fine-tuning with the robustness of gradient boosting.

**Deployment recommendations:**
- **Web app (GPU)**: XGBoost on fine-tuned embeddings (96.8% acc) — extract embeddings with fine-tuned SigLIP, classify with XGBoost
- **Web app (CPU fallback)**: XGBoost on frozen embeddings (95.7% acc) — no fine-tuned model needed, just frozen SigLIP + XGBoost
- **Mobile (offline)**: Distilled lightweight model (see Phase 2B) — target <25 MB on-device, knowledge distillation from the best model

The **deployed model** is ranked by **F1 macro** (the primary metric for imbalanced dermatology data).

### Full Pipeline Results (47,277 samples, 5 datasets)

**Binary Classification (benign/malignant):**

| Model | Embedding Type | Accuracy | F1 Macro | F1 Malignant | AUC |
|-------|----------------|----------|----------|--------------|-----|
| Baseline | N/A | 0.791 | 0.442 | 0.000 | 0.500 |
| Logistic | Frozen SigLIP | 0.840 | 0.792 | 0.692 | 0.922 |
| XGBoost | Frozen SigLIP | 0.957 | 0.938 | 0.903 | 0.990 |
| Fine-tuned SigLIP | End-to-end | 0.962 | 0.945 | 0.914 | 0.990 |
| **XGBoost** | **Fine-tuned SigLIP** | **0.968** | **0.951** | **0.922** | **0.992** |

**Key finding: XGBoost on fine-tuned SigLIP embeddings achieves the best results across all metrics.** This two-stage approach (fine-tune embeddings, then train XGBoost) outperforms both frozen-embedding classifiers and end-to-end fine-tuned SigLIP.

**Why F1 matters in dermatology:** In skin cancer screening, false negatives (missed malignancies) are far more costly than false positives. F1 macro balances precision and recall equally across both classes, while F1 malignant specifically measures performance on the critical cancer-detection task. Accuracy is misleading on imbalanced data (70% benign → 70% accuracy by always predicting benign).

**Fairness (XGBoost on fine-tuned embeddings):**
- Fitzpatrick equalized odds gap: sensitivity=0.044, specificity=0.098
- Sensitivity gap < 5% across all skin tones — critical for equitable healthcare

### Robustness to Image Distortions

Real-world smartphone photos suffer from blur, noise, compression, and poor lighting. We tested model robustness across 12 distortion types on 1000 test images:

| Distortion | XGBoost (Frozen) | XGBoost (Fine-tuned) | Δ |
|------------|------------------|----------------------|---|
| None (clean) | 96.2% | 97.5% | +1.3% |
| Blur (light) | 91.8% | 95.3% | **+3.5%** |
| Blur (heavy) | 89.7% | 92.9% | +3.2% |
| Noise (light) | 79.0% | 78.8% | -0.2% |
| Noise (heavy) | 79.2% | 79.4% | +0.2% |
| Brightness (dark) | 85.9% | 90.1% | **+4.2%** |
| Brightness (bright) | 86.8% | 90.1% | +3.3% |
| Compression (light) | 95.0% | 96.8% | +1.8% |
| Compression (heavy) | 94.8% | 97.0% | +2.2% |
| Rotation (15°) | 90.3% | 94.0% | **+3.7%** |
| Rotation (45°) | 88.9% | 92.7% | +3.8% |
| Combined (realistic) | 84.3% | 85.8% | +1.5% |

**Key insight:** Fine-tuned embeddings improve robustness across nearly all distortion types, with the largest gains in blur, brightness, and rotation — common issues in smartphone photos. Noise remains challenging for both models.

### Model Sizes and Inference Times

| Component | Size | Notes |
|-----------|------|-------|
| Full fine-tuned SigLIP | 3,350 MB | Complete model with all layers |
| SigLIP classification head only | 1.14 MB | For transfer learning |
| XGBoost (frozen embeddings) | 1.77 MB | Deployable without fine-tuned model |
| XGBoost (fine-tuned embeddings) | 1.40 MB | Requires fine-tuned SigLIP for inference |
| Logistic regression | 0.04 MB | Smallest classifier |

| Operation | Time | Hardware |
|-----------|------|----------|
| Fine-tuned SigLIP inference | 48 ms | RTX 4070 Ti SUPER |
| Frozen embedding extraction | 16 ms | RTX 4070 Ti SUPER |
| XGBoost inference | 0.6 ms | CPU |
| XGBoost training (fine-tuned) | 40 s | CPU |

**Deployment trade-off:** For web deployment with GPU, end-to-end fine-tuned SigLIP offers best accuracy. For CPU-only or edge deployment, pre-extract embeddings (16ms GPU) then run XGBoost (0.6ms CPU) for comparable accuracy with lower latency.

### Condition Classification (10-class)

| Model | Test Acc | F1 Macro |
|-------|----------|----------|
| Logistic | 0.684 | 0.596 |
| Deep MLP | 0.680 | 0.631 |

**Per-condition F1 (logistic, evaluation split):**

| Condition | F1 | n |
|-----------|-----|---|
| Melanoma | 0.667 | 329 |
| Basal Cell Carcinoma | 0.745 | 1,174 |
| Squamous Cell Carcinoma | 0.673 | 147 |
| Actinic Keratosis | 0.755 | 220 |
| Melanocytic Nevus | 0.847 | 2,677 |
| Seborrheic Keratosis | 0.584 | 546 |
| Dermatofibroma | 0.816 | 85 |
| Vascular Lesion | 0.864 | 66 |
| Non-Neoplastic | 0.707 | 1,258 |
| Other/Unknown | 0.707 | 2,954 |

**Backbone**: google/siglip-so400m-patch14-384 (878M params, 1152-d embeddings)

### Saved Model Artifacts

The fine-tuned SigLIP model is saved in two locations:
- **Cache** (ephemeral): `results/cache/finetuned_model/` — may be cleared between runs
- **Dedicated backup** (permanent): `models/finetuned_siglip/` — tracked in git (config only; weights gitignored due to size)

Contents of `models/finetuned_siglip/`:
| File | Size | Description |
|------|------|-------------|
| `config.json` | <1 KB | Model architecture config (tracked in git) |
| `head_state.pt` | ~1.2 MB | Classification head weights only |
| `model_state.pt` | ~3.3 GB | Full fine-tuned SigLIP + head weights |

To restore from backup, load with `EndToEndSigLIP` using the config and `model_state.pt`.

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
├── models/
│   └── finetuned_siglip/              # Dedicated backup of fine-tuned model
│       ├── config.json                # Architecture config (tracked in git)
│       ├── head_state.pt              # Classification head only (~1.2 MB)
│       └── model_state.pt             # Full fine-tuned SigLIP (~3.3 GB, gitignored)
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
│       ├── embeddings.pt              # Cached SigLIP embeddings (~46k samples)
│       ├── classifier.pkl             # Default binary model (for app)
│       ├── classifier_baseline.pkl    # Majority class baseline (binary)
│       ├── classifier_logistic.pkl    # Logistic regression (binary)
│       ├── classifier_deep.pkl        # Deep MLP (binary)
│       ├── classifier_condition.pkl   # Default condition model (10-class)
│       ├── classifier_condition_logistic.pkl  # Logistic regression (condition)
│       ├── classifier_condition_deep.pkl      # Deep MLP (condition)
│       ├── finetuned_model/           # End-to-end fine-tuned SigLIP (if trained)
│       ├── metadata.csv               # Full dataset metadata
│       ├── test_metadata.csv          # Test split metadata
│       ├── training_results.json      # Binary training metrics
│       ├── condition_training_results.json    # Condition training metrics
│       └── evaluation_results.json    # Full evaluation (both targets)
├── scripts/
│   ├── train.py                       # Main training (--multi-dataset --model all --domain-balance)
│   ├── train_all_models.py            # Train + compare all model types
│   ├── evaluate.py                    # Full fairness evaluation report
│   └── evaluate_cross_domain.py       # Leave-one-domain-out experiment
├── src/
│   ├── data/
│   │   ├── schema.py                  # SkinSample dataclass (unified schema for all datasets)
│   │   ├── taxonomy.py                # Unified condition taxonomy (10 categories) + per-dataset maps
│   │   ├── loader.py                  # load_ham10000(), load_multi_dataset() orchestrator
│   │   ├── datasets/
│   │   │   ├── __init__.py            # DATASET_LOADERS registry (5 datasets)
│   │   │   ├── ham10000.py            # HAM10000 adapter -> SkinSample
│   │   │   ├── ddi.py                 # DDI adapter (skin_tone -> Fitzpatrick mapping)
│   │   │   ├── fitzpatrick17k.py      # Fitz17k adapter (three_partition_label -> binary)
│   │   │   ├── pad_ufes.py            # PAD-UFES adapter (smartphone domain)
│   │   │   └── bcn20000.py            # BCN20000 adapter (dermoscopic, adds SCC)
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
├── Dockerfile                         # Containerized deployment (CPU)
├── Dockerfile.gpu                     # GPU deployment with CUDA
├── LICENSE
├── Makefile                           # Build targets (install, data, train, app)
├── PLAN.md                            # This file
├── README.md
├── run_pipeline.py                    # Unified pipeline: data → embed → train → eval → app
└── requirements.txt
```

---

## Usage

### Unified Pipeline (Recommended)

The `run_pipeline.py` script runs the entire pipeline end-to-end in one command:

```bash
# Full pipeline: data → embed → train → evaluate → web app
python run_pipeline.py

# Quick smoke test (500 samples, skip web app)
python run_pipeline.py --quick --no-app

# Skip training, re-evaluate existing models
python run_pipeline.py --skip-train

# Just launch the web app (requires prior training)
python run_pipeline.py --app-only
```

The pipeline:
1. **Checks environment** — verifies packages, 5 datasets, model cache
2. **Loads data** — reads metadata and image paths (no images in RAM)
3. **Extracts embeddings** — streams images from disk per-batch through SigLIP
4. **Trains models** — binary (baseline, logistic, deep MLP) + condition (logistic, deep MLP)
5. **Evaluates** — fairness metrics, per-Fitzpatrick sensitivity, equalized odds, condition accuracy
6. **Launches web app** — FastAPI triage + condition estimation interface

Each stage is wrapped in error handling. If a stage fails, the pipeline logs a warning and continues. Internet is only needed on the first run (SigLIP model download).

### Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
# Or:
make install

# For GPU support (CUDA 12.6, requires NVIDIA driver ≥535):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Download HAM10000 (requires Kaggle API credentials)
make data

# Download other datasets (see "Downloading Data Locally" above)
# DDI: requires Stanford AIMI access request
# Fitzpatrick17k: CSV from GitHub, images from URLs
# PAD-UFES-20: from Mendeley Data or Kaggle
```

### Individual Training Scripts
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

# Docker (CPU)
make app-docker

# Docker with GPU
make app-docker-gpu
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

9. **Lazy/streaming image loading** — Dataset loaders store file paths, not PIL images. The embedding extractor loads images per-batch from disk, keeping only a few images in RAM at any time. This reduced data loading from ~393s to ~2s for 29k samples and avoids multi-GB RAM usage.

10. **GPU auto-detection** — Pipeline auto-detects CUDA availability and adjusts batch size (4 for CPU, 16 for GPU). With RTX 4070 Ti SUPER (16GB VRAM), full-dataset embedding extraction runs significantly faster than CPU.

---

## Phase 2: Deployment Roadmap

### 2A. Web Application (Internet-Connected)

A cloud-hosted web app that allows users to upload a photo from any device with an internet connection and receive triage results in real time.

**Architecture:**
- **Backend**: FastAPI server hosting the fine-tuned SigLIP model (or XGBoost on pre-extracted embeddings as a fast fallback)
- **Frontend**: Responsive web UI (mobile-first design) with camera capture and image upload
- **Inference flow**: User uploads image -> server runs SigLIP embedding extraction -> classification head -> triage tier + condition estimate returned to client
- **Hosting options**: Cloud GPU instance (e.g., AWS/GCP with T4), or CPU-only with the lightweight XGBoost pipeline (embeddings + XGBoost, no fine-tuned backbone needed at inference)

**Key tasks:**
1. Refactor `app/main.py` to support both model backends (fine-tuned SigLIP end-to-end vs. XGBoost on frozen embeddings)
2. Add mobile-responsive camera capture (HTML5 `getUserMedia` API) for phone-based photo taking
3. Deploy behind HTTPS with rate limiting and input validation (image size, format checks)
4. Add result explanation panel: triage tier, condition estimate, confidence, and prominent medical disclaimer
5. Containerize with Docker for reproducible deployment (existing `Dockerfile` as starting point)
6. Evaluate hosting: HuggingFace Spaces (free GPU), AWS SageMaker, or GCP Cloud Run

**Model selection for web:**
- **Primary**: Fine-tuned SigLIP (`models/finetuned_siglip/`) — best accuracy (92.3%), requires GPU or beefy CPU
- **Fallback**: XGBoost on frozen SigLIP embeddings — F1 macro 0.938, runs on CPU, only needs the frozen SigLIP encoder (no fine-tuned weights)

### 2B. Offline Mobile Application (No Internet Required)

A native mobile app (Android and/or iOS) that runs inference entirely on-device, enabling use in areas without reliable internet connectivity — critical for the low-resource settings this project targets.

**Core challenge:** The full fine-tuned SigLIP is ~3.3 GB — too large for most mobile devices. A lightweight model is required.

**Model compression strategy:**
1. **Export classification head only** (~1.2 MB, `head_state.pt`) — pair with a smaller on-device vision backbone
2. **Knowledge distillation**: Train a smaller student model (e.g., MobileNetV3, EfficientNet-Lite, or SigLIP-B/16 at 86M params) to mimic the fine-tuned SigLIP's predictions. Target model size: 20-50 MB
3. **ONNX/TFLite export**: Convert the distilled model to ONNX (Android via ONNX Runtime) or TFLite (Android + iOS via TensorFlow Lite / Core ML)
4. **Quantization**: Apply INT8 post-training quantization to further reduce model size and improve inference speed on mobile hardware (ARM NEON / Apple Neural Engine)

**Platform strategy:**

| Platform | Runtime | Format | Framework |
|----------|---------|--------|-----------|
| Android | ONNX Runtime Mobile / TFLite | `.onnx` or `.tflite` | Kotlin/Java + CameraX |
| iOS | Core ML | `.mlmodel` (converted from ONNX) | Swift + AVFoundation |
| Cross-platform | React Native or Flutter | TFLite via plugin | Single codebase for both |

**Key tasks:**
1. Select a lightweight backbone (MobileNetV3-Large or EfficientNet-Lite0 are strong candidates at ~5-20 MB)
2. Implement knowledge distillation: fine-tuned SigLIP as teacher, lightweight model as student, train on the same 47k dataset
3. Validate distilled model achieves acceptable accuracy (target: >85% test acc, >0.75 F1 malignant — within ~5% of the teacher)
4. Export to ONNX and convert to platform-specific formats (TFLite, Core ML)
5. Build minimal mobile app with camera capture, on-device inference, and triage display
6. Handle edge cases: poor lighting, blurry images, non-lesion photos (OOD detection)
7. Package model weights inside the app binary (no download required after install)

**Estimated model sizes after distillation + quantization:**

| Model | FP32 | INT8 Quantized |
|-------|------|----------------|
| MobileNetV3-Large + head | ~22 MB | ~6 MB |
| EfficientNet-Lite0 + head | ~19 MB | ~5 MB |
| SigLIP-B/16 (small) + head | ~350 MB | ~90 MB |

MobileNetV3-Large is the recommended starting point: small enough for any phone, well-supported by TFLite/Core ML, and proven effective for medical imaging transfer learning.

#### Phase 2B Results (Completed 2026-02-04)

**Training completed** on the full 47,277 sample dataset using knowledge distillation from fine-tuned SigLIP.

| Model | Parameters | Size | Accuracy | F1 Macro | F1 Malignant | AUC |
|-------|------------|------|----------|----------|--------------|-----|
| SigLIP (Teacher) | 878M | 3.4 GB | 92.30% | 0.887 | 0.824 | ~0.96 |
| **MobileNetV3-Large** | 3.2M | **12.5 MB** | **92.44%** | 0.884 | 0.816 | 0.959 |
| **EfficientNet-B0** | 4.3M | **16.8 MB** | **92.65%** | 0.887 | 0.820 | 0.960 |

**Key achievements:**
- Both distilled models match teacher performance (within 0.4% F1)
- Model size reduced by **200-270x** (from 3.4 GB to 12-17 MB)
- Targets exceeded: >85% accuracy, >0.75 F1 malignant, <25 MB size
- ONNX exports ready for Core ML (iOS) and TFLite (Android) conversion

**Artifacts:**
- `models/mobilenet_distilled/` — MobileNetV3-Large weights + ONNX
- `models/efficientnet_distilled/` — EfficientNet-B0 weights + ONNX
- `mobile/ios/SkinTag/` — Complete iOS SwiftUI app
- `mobile/flutter/skin_tag/` — Complete Flutter cross-platform app
- `docs/MOBILE_REPORT.md` — Full deployment report
