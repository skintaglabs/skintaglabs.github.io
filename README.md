# SkinTag

Robust skin lesion classification using MedSigLIP embeddings with augmentations targeting real-world imaging variations.

## Problem

Medical images are captured under inconsistent conditions — different cameras, lighting, compression artifacts, and noise levels. Models trained on clean clinical images often fail when deployed on images from varied sources. We build a classifier robust to these real-world variations.

## Approach

1. **Pre-trained Model**: MedSigLIP (400M vision encoder trained on medical images)
2. **Transfer Learning**: Extract embeddings once, train lightweight sklearn classifier
3. **Augmentations**: Lighting variation, image noise, compression artifacts

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local

```bash
make install   # Install dependencies
make data      # Download HAM10000 dataset
make train     # Run training
make evaluate  # Run evaluation
```

### Model Management

```bash
make upload-model MODEL=results/models/classifier.pt TAG=v1.0.0    # Upload to GitHub release
make download-model TAG=v1.0.0 OUTPUT=results/models/classifier.pt # Download from release
```

### GitHub Actions

1. Go to **Actions → Train Model → Run workflow**
2. Dataset URL is pre-filled with HAM10000
3. Results appear in the workflow summary

## Augmentation Strategy

| Variation | Technique | Real-World Scenario |
|-----------|-----------|---------------------|
| Lighting | Brightness, contrast, gamma | Different exam room lighting |
| Noise | Gaussian, ISO noise | Low-light photos, sensor noise |
| Quality | JPEG compression | Telemedicine, compressed uploads |

## Structure

```
├── notebooks/demo.ipynb    # Presentation notebook
├── src/
│   ├── data/               # Data loading and augmentations
│   ├── model/              # MedSigLIP embeddings + classifier
│   └── evaluation/         # Robustness metrics
├── configs/config.yaml     # Hyperparameters
└── results/                # Figures and metrics
```
