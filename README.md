# SkinTag

AI-powered skin lesion triage for early melanoma detection, designed for equitable healthcare access.

[![Deploy Frontend](https://github.com/skintaglabs/main/actions/workflows/deploy-webapp.yml/badge.svg)](https://github.com/skintaglabs/main/actions/workflows/deploy-webapp.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier)
[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://skintaglabs.github.io/main/)

## Overview

SkinTag provides preliminary screening for skin lesions using a fine-tuned SigLIP vision model. The system:
- Classifies lesions as benign/malignant (binary triage)
- Estimates specific skin conditions (10-class)
- Provides actionable recommendations
- Works on mobile devices with camera support

## Key Features

- **Multi-dataset training:** 47k images from 5 datasets (HAM10000, ISIC, BCN20000, Fitzpatrick17k, PAD-UFES-20)
- **Fairness-aware:** Tested across skin tones (Fitzpatrick scale)
- **Domain robustness:** Trained on clinical, dermoscopic, and mobile photos
- **Serverless inference:** Free GitHub Actions deployment
- **PWA support:** Installable on mobile devices

## Quick Start

### Deploy Full Stack (Free)

```bash
# 1. Add Hugging Face token
gh secret set HF_TOKEN

# 2. Start inference server
gh workflow run inference-server.yml

# Frontend auto-deploys with tunnel URL
# Visit: https://skintaglabs.github.io/main/
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for details.

### Local Development

```bash
# Backend
make venv
make app

# Frontend
cd webapp
python -m http.server 8080
# Visit: http://localhost:8080?api=http://localhost:8000
```

## Model Performance

| Metric | Binary Triage | Condition (10-class) |
|--------|--------------|---------------------|
| F1 Score | 0.878 | 0.736 |
| AUC | 0.980 | - |
| Fairness Gap | <0.10 | <0.15 |

Fine-tuned from `google/siglip-so400m-patch14-384` (878M parameters).

Model hosted at [skintaglabs/siglip-skin-lesion-classifier](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier).

## Architecture

```
webapp/                 # Single-file frontend (PWA)
app/main.py            # FastAPI backend
src/
├── data/              # Dataset loaders + augmentations
├── model/             # SigLIP embeddings + classifiers
├── evaluation/        # Cross-domain + fairness metrics
└── utils/             # Hugging Face model downloader
scripts/               # Training pipeline
configs/config.yaml    # Training configuration
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

## Deployment Options

1. **GitHub Actions** (recommended, free)
   - Serverless inference via Cloudflare tunnel
   - Auto-restarts every 5 hours
   - See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

2. **Docker**
   ```bash
   docker build -t skintag .
   docker run -p 8000:8000 skintag
   ```

3. **Local**
   ```bash
   make app
   ```

## Research

Detailed technical writeup: [writeup/main.tex](writeup/main.tex)

NeurIPS-style paper with full methodology, results, and fairness analysis.

## Citation

If you use this work, please cite:

```bibtex
@misc{skintag2026,
  title={SkinTag: Multi-Dataset, Domain-Robust, Fairness-Aware Skin Lesion Triage},
  author={SkinTag Labs},
  year={2026},
  url={https://github.com/skintaglabs/main}
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
