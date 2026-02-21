---
title: SkinTag Inference
emoji: 🏥
colorFrom: red
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# SkinTag

AI-powered skin lesion triage for early melanoma detection, designed for equitable healthcare access.

[![Deploy Frontend](https://github.com/skintaglabs/skintaglabs.github.io/actions/workflows/deploy-webapp.yml/badge.svg)](https://github.com/skintaglabs/skintaglabs.github.io/actions/workflows/deploy-webapp.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=flat)](https://www.python.org/downloads/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow?style=flat)](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier)

https://skintaglabs.github.io/

## Overview

SkinTag provides preliminary screening for skin lesions using a fine-tuned SigLIP vision-language model trained on 47,277 images from five complementary datasets spanning dermoscopic, clinical, and smartphone domains. The system:
- Classifies lesions as benign/malignant with three-tier urgency triage (LOW/MODERATE/HIGH)
- Estimates specific skin conditions (10-class)
- Provides actionable recommendations with inflammatory condition auto-promotion
- Works on mobile devices with camera support

## Key Features

- **Multi-dataset training:** 47k images from 5 datasets (HAM10000, DDI, BCN20000, Fitzpatrick17k, PAD-UFES-20)
- **Fairness-aware:** Fitzpatrick-balanced sampling, equalized odds gap <5% across skin tones
- **Domain robustness:** Trained on clinical, dermoscopic, and smartphone photos with field condition augmentations
- **React frontend:** React 19 + TypeScript + Vite + Tailwind CSS

## Quick Start

### Inference (CPU, Pre-trained Models)

```bash
# Setup
make venv                # Create venv + install dependencies

# Run API server
make app                 # Launch at http://localhost:8000

# Frontend (separate terminal)
make preview             # React dev server
```

### Training (NVIDIA GPU Required)

```bash
# Setup
make install-gpu         # Install dependencies with CUDA

# Download datasets
make data                # HAM10000 from Kaggle (requires kaggle CLI)

# Train models
make pipeline            # Full pipeline: data → embed → train → eval
# OR
make train               # Train individual models
make evaluate            # Run fairness evaluation
```

## Model Performance

Best model: XGBoost on fine-tuned SigLIP embeddings (two-stage training).

| Metric | Value |
|--------|-------|
| Accuracy | 96.8% |
| F1 Macro | 0.951 |
| F1 Malignant | 0.922 |
| AUC | 0.992 |
| Fitzpatrick Sensitivity Gap | <0.05 |

Fine-tuned from `google/siglip-so400m-patch14-384` (878M parameters) on 47,277 images across 5 datasets.

Model hosted at [skintaglabs/siglip-skin-lesion-classifier](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier).

## Project Structure

```
SkinTag/
├── README.md                  <- This file
├── requirements.txt           <- Python dependencies
├── Makefile                   <- Setup and run from command line (install, data, train, app)
├── run_pipeline.py            <- Unified pipeline: data -> embed -> train -> eval -> app
├── app/
│   └── main.py                <- FastAPI backend (upload -> SigLIP -> triage)
├── webapp-react/              <- React 19 + TypeScript + Vite + Tailwind CSS frontend
├── src/
│   ├── data/                  <- Dataset loaders, taxonomy, sampling, augmentations
│   ├── model/                 <- SigLIP embeddings, classifiers, triage system
│   └── evaluation/            <- Fairness metrics, cross-domain evaluation
├── scripts/                   <- Training, evaluation, and utility scripts
│   ├── train.py               <- Main training script
│   └── evaluate.py            <- Fairness evaluation
├── models/                    <- Trained model artifacts (gitignored weights)
├── data/                      <- Raw datasets (gitignored, 5 sources)
├── notebooks/                 <- Exploration notebooks (EDA, demos)
├── configs/                   <- YAML configuration files
├── .gitignore
└── Dockerfile                 <- Containerized deployment
```

## Training

Models trained on NVIDIA RTX 4070 Ti Super.

```bash
make install      # Install dependencies
make data         # Download datasets
make train        # Train all models
make evaluate     # Run evaluation
```

Uploads to Hugging Face:
```bash
cd results/cache/finetuned_model
huggingface-cli upload YourOrg/YourModel . --repo-type model
```

## Deployment

| Layer | Service | Notes |
|-------|---------|-------|
| Inference backend | [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK) | `app_port: 8000`, free tier |
| Frontend | [GitHub Pages](https://pages.github.com) | Auto-deployed via `deploy-webapp.yml` on push to `main` |

### Backend: Hugging Face Spaces

The `README.md` already contains the Space YAML frontmatter required by HF (`sdk: docker`, `app_port: 8000`).

1. Create a new Space under the `skintaglabs` org (Docker SDK) and push this repo to it
2. In the Space **Settings → Variables and secrets**, add:
   - `USE_HF_MODELS` = `true`
   - `HF_TOKEN` = your HF token (only needed if the model repo is private)
   - `HF_REPO_ID` = `skintaglabs/siglip-skin-lesion-classifier` (or override)
3. The Space builds from the `Dockerfile` and starts the FastAPI server on port 8000

### Frontend: GitHub Pages + pointing at the Space

The frontend is deployed automatically by `deploy-webapp.yml` on push to `main`. To point it at the HF Space backend:

1. In this repo's **Settings → Secrets and variables → Actions**, add:
   - `API_URL` = `https://skintaglabs-skintag-inference.hf.space` (your Space URL)
2. Re-run the `Deploy Frontend` workflow (or push any change to `webapp-react/`)

The workflow already reads `secrets.API_URL` as a fallback when no manual URL is provided.

### Local development

```bash
make venv    # Create venv + install deps
make app     # FastAPI at http://localhost:8000
make preview # React dev server
```

### Docker (self-hosted)

```bash
docker build -t skintag .
docker run -p 8000:8000 -e USE_HF_MODELS=true -e HF_TOKEN=your_token skintag
```

## Citation

If you use this work, please cite:

```bibtex
@misc{skintag2026,
  title={SkinTag: Domain-Robust and Fairness-Aware Skin Lesion Triage via Fine-Tuned Vision-Language Models},
  author={Tanzillo, Dominic and Neves, Jonas and Gill, Roshan},
  year={2026},
  url={https://github.com/skintaglabs/skintaglabs.github.io}
}
```

**Base model (SigLIP):**
```bibtex
@misc{zhai2023sigmoid,
  title={Sigmoid Loss for Language Image Pre-Training},
  author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
  year={2023},
  eprint={2303.15343},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowledgments

- **Base Model:** [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) by Google (Apache 2.0)
- **Datasets:** HAM10000, DDI (Stanford), Fitzpatrick17k, PAD-UFES-20, BCN20000

## License

MIT
