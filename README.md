# SkinTag

Robust skin lesion classification using MedSigLIP embeddings with augmentations targeting demographic and environmental variations.

## Approach

1. **Pre-trained Model**: MedSigLIP (400M vision encoder trained on dermatology images)
2. **Transfer Learning**: Extract embeddings, train lightweight classifier
3. **Augmentations**: Skin tone simulation, lighting variation, image noise

## Setup

```bash
pip install -r requirements.txt

# Download HAM10000 dataset from Kaggle
pip install kaggle
kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip
```

## Usage

```bash
# Training
python scripts/train.py

# Evaluation
python scripts/evaluate.py
```

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
